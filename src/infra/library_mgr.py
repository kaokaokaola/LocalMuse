"""
LocalMuse V2 — LibraryManager (v3.0 SQLite catalog)

Storage layout (unchanged on disk):
  <name>.library/
    metadata.json          — library header (name, version, marker)
    tags.json              — global tag list (fast access, kept in sync)
    catalog.db             — SQLite: one row per image (NEW)
    images/
      <UID>.info/
        metadata.json      — legacy / backup per-image record (kept for safety)
        thumbnail.png      — fixed name, O(1) lookup
        <original file>    — copy of the source image

Migration:
  On open_library(), if catalog.db is missing rows but .info/ folders exist,
  a one-time migration reads every legacy metadata.json and inserts rows
  into catalog.db. Legacy files are kept (not deleted) for backward compat.

Performance targets (SSD, 100 K images):
  open_library()           < 10 ms   (DB connection + header read)
  list_page(offset, limit) < 20 ms   (indexed SELECT)
  save_image_entry()       < 1  ms   (single INSERT / REPLACE)
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional, Tuple

from PIL import Image

# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #

LIBRARY_EXTENSION = ".library"
METADATA_FILENAME  = "metadata.json"
TAGS_FILENAME      = "tags.json"
IMAGES_SUBFOLDER   = "images"
CATALOG_FILENAME   = "catalog.db"
THUMBNAIL_FILENAME = "thumbnail.png"
TRASH_SUBFOLDER    = ".trash"
TRASH_MANIFEST     = "manifest.json"

SLOT_NAMES = ("semantic", "sketch", "sketch_fit", "sketch_crop", "depth", "pose")

LIBRARY_VERSION = "3.0"
LIBRARY_MARKER  = "localmuse_library"

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".tif", ".tiff", ".gif",
}


def _hash_distance(a: str, b: str) -> Optional[int]:
    try:
        if not a or not b or len(a) != len(b):
            return None
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except Exception:
        return None


def _best_duplicate(current, kind: str, distance: int):
    if current is None:
        return kind, distance
    cur_kind, cur_distance = current
    priority = {"exact": 0, "flipped": 1, "similar": 2}
    if distance < cur_distance:
        return kind, distance
    if distance == cur_distance and priority.get(kind, 9) < priority.get(cur_kind, 9):
        return kind, distance
    return current

# ------------------------------------------------------------------ #
#  SQLite schema
# ------------------------------------------------------------------ #

_DDL = """
CREATE TABLE IF NOT EXISTS images (
    uid                 TEXT PRIMARY KEY,
    name                TEXT NOT NULL DEFAULT '',
    ext                 TEXT NOT NULL DEFAULT '',
    width               INTEGER NOT NULL DEFAULT 0,
    height              INTEGER NOT NULL DEFAULT 0,
    file_size           INTEGER NOT NULL DEFAULT 0,
    dominant_color      TEXT NOT NULL DEFAULT '[128,128,128]',
    color_palette       TEXT NOT NULL DEFAULT '[]',
    color_palette_ratio TEXT NOT NULL DEFAULT '[]',
    tags                TEXT NOT NULL DEFAULT '[]',
    annotation          TEXT NOT NULL DEFAULT '',
    source_path         TEXT NOT NULL DEFAULT '',
    original_filename   TEXT NOT NULL DEFAULT '',
    ocr_text            TEXT NOT NULL DEFAULT '',
    exif                TEXT NOT NULL DEFAULT '{}',
    indexed_slots       TEXT NOT NULL DEFAULT '{}',
    file_hash           TEXT NOT NULL DEFAULT '',
    phash               TEXT NOT NULL DEFAULT '',
    phash_flip          TEXT NOT NULL DEFAULT '',
    duplicate_group     TEXT NOT NULL DEFAULT '',
    duplicate_kind      TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL DEFAULT '',
    modified_at         TEXT NOT NULL DEFAULT '',
    import_at           TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_import_at  ON images(import_at  DESC);
CREATE INDEX IF NOT EXISTS idx_name       ON images(name        COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_created_at ON images(created_at  DESC);
"""


# ------------------------------------------------------------------ #
#  Image record factory (API unchanged)
# ------------------------------------------------------------------ #

def make_image_record(
    uid: str,
    name: str,
    ext: str,
    width: int,
    height: int,
    file_size: int,
    dominant_color: Tuple[int, int, int],
    color_palette: List[Tuple[int, int, int]],
    tags: List[str] = None,
    annotation: str = "",
    source_path: str = "",
    ocr_text: str = "",
    original_filename: str = "",
    exif: dict = None,
    color_palette_ratio: list = None,
    indexed_slots: dict = None,
    file_hash: str = "",
    phash: str = "",
    phash_flip: str = "",
    duplicate_group: str = "",
    duplicate_kind: str = "",
) -> dict:
    now = datetime.utcnow().isoformat()
    return {
        "id":                   uid,
        "name":                 name,
        "ext":                  ext,
        "width":                width,
        "height":               height,
        "file_size":            file_size,
        "dominant_color":       list(dominant_color),
        "color_palette":        [list(c) for c in (color_palette or [])],
        "color_palette_ratio":  color_palette_ratio or [],
        "exif":                 exif or {},
        "tags":                 tags or [],
        "annotation":           annotation,
        "source_path":          source_path,
        "original_filename":    original_filename,
        "ocr_text":             ocr_text,
        "indexed_slots":        indexed_slots or {},
        "file_hash":            file_hash or "",
        "phash":                phash or "",
        "phash_flip":           phash_flip or "",
        "duplicate_group":      duplicate_group or "",
        "duplicate_kind":       duplicate_kind or "",
        "created_at":           now,
        "modified_at":          now,
        "import_at":            now,
    }


# ------------------------------------------------------------------ #
#  LibraryManager
# ------------------------------------------------------------------ #

class LibraryManager:
    def __init__(self):
        self._library_path: Optional[Path] = None
        self._meta: dict = {}
        self._tags: List[str] = []
        self._db: Optional[sqlite3.Connection] = None

    # ---------------------------------------------------------------- #
    #  Properties
    # ---------------------------------------------------------------- #

    @property
    def is_open(self) -> bool:
        return self._library_path is not None

    @property
    def library_path(self) -> Optional[Path]:
        return self._library_path

    @property
    def library_name(self) -> str:
        return self._library_path.stem if self._library_path else ""

    @property
    def index_dir(self) -> Optional[Path]:
        return self._library_path

    @property
    def _catalog_path(self) -> Optional[Path]:
        return self._library_path / CATALOG_FILENAME if self._library_path else None

    def has_index(self) -> bool:
        if self._library_path is None:
            return False
        if any((self._library_path / f"index_{s}.faiss").exists() for s in SLOT_NAMES):
            return True
        return (self._library_path / "vectors.index").exists()

    # ---------------------------------------------------------------- #
    #  Library lifecycle
    # ---------------------------------------------------------------- #

    def create_library(self, parent_dir: str, name: str) -> Path:
        lib_path = Path(parent_dir) / f"{name}{LIBRARY_EXTENSION}"
        if lib_path.exists():
            raise FileExistsError(f"Library already exists: {lib_path}")
        lib_path.mkdir(parents=True)
        (lib_path / IMAGES_SUBFOLDER).mkdir()
        meta = {
            "marker":     LIBRARY_MARKER,
            "version":    LIBRARY_VERSION,
            "name":       name,
            "created_at": datetime.utcnow().isoformat(),
        }
        self._write_json(lib_path / METADATA_FILENAME, meta)
        self._write_json(lib_path / TAGS_FILENAME, [])
        self._library_path = lib_path
        self._meta = meta
        self._tags = []
        self._open_db()
        return lib_path

    def open_library(self, path: str) -> None:
        lib_path = Path(path)
        meta_file = lib_path / METADATA_FILENAME
        if not lib_path.is_dir() or not meta_file.exists():
            raise ValueError(f"Not a valid library: {path}")
        meta = self._read_json(meta_file)
        if meta.get("marker") != LIBRARY_MARKER:
            raise ValueError(f"Not a LocalMuse library: {path}")
        self._library_path = lib_path
        self._meta = meta
        self._tags = self._read_json(lib_path / TAGS_FILENAME, default=[])
        self._open_db()
        self._maybe_migrate()

    def close_library(self) -> None:
        if self._db:
            self._db.close()
            self._db = None
        self._library_path = None
        self._meta = {}
        self._tags = []

    # ---------------------------------------------------------------- #
    #  SQLite connection
    # ---------------------------------------------------------------- #

    def _open_db(self) -> None:
        if self._db:
            self._db.close()
        self._db = sqlite3.connect(
            str(self._catalog_path), check_same_thread=False
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA cache_size=-32000")
        self._db.execute("PRAGMA temp_store=MEMORY")
        self._db.executescript(_DDL)
        self._ensure_schema_upgrades()
        self._db.commit()

    def _ensure_schema_upgrades(self) -> None:
        """Add columns introduced after the original v3 catalog schema."""
        columns = {
            row["name"]
            for row in self._db.execute("PRAGMA table_info(images)").fetchall()
        }
        additions = {
            "file_hash": "TEXT NOT NULL DEFAULT ''",
            "phash": "TEXT NOT NULL DEFAULT ''",
            "phash_flip": "TEXT NOT NULL DEFAULT ''",
            "duplicate_group": "TEXT NOT NULL DEFAULT ''",
            "duplicate_kind": "TEXT NOT NULL DEFAULT ''",
        }
        for name, ddl in additions.items():
            if name not in columns:
                self._db.execute(f"ALTER TABLE images ADD COLUMN {name} {ddl}")
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_hash ON images(file_hash)"
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_phash ON images(phash)")
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_phash_flip ON images(phash_flip)"
        )

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            yield self._db
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

    # ---------------------------------------------------------------- #
    #  One-time migration: legacy .info/metadata.json → catalog.db
    # ---------------------------------------------------------------- #

    def _maybe_migrate(self) -> None:
        count = self._db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        if count > 0:
            return
        images_dir = self._library_path / IMAGES_SUBFOLDER
        if not images_dir.exists():
            return
        info_dirs = [
            d for d in images_dir.iterdir()
            if d.is_dir() and d.name.endswith(".info")
        ]
        if not info_dirs:
            return
        print(f"[LibraryManager] Migrating {len(info_dirs)} legacy records to catalog.db …")
        rows = []
        for d in info_dirs:
            uid = d.stem
            meta_file = d / METADATA_FILENAME
            if not meta_file.exists():
                continue
            try:
                rec = self._read_json(meta_file)
                rows.append(self._record_to_row(uid, rec))
            except Exception as e:
                print(f"[LibraryManager] Skip {uid}: {e}")
        with self._tx():
            self._db.executemany(self._insert_sql(), rows)
        print(f"[LibraryManager] Migration complete: {len(rows)} rows inserted.")

    # ---------------------------------------------------------------- #
    #  UID and .info folder
    # ---------------------------------------------------------------- #

    def generate_uid(self) -> str:
        return uuid.uuid4().hex[:13].upper()

    def _info_dir(self, uid: str) -> Path:
        assert self._library_path is not None
        return self._library_path / IMAGES_SUBFOLDER / f"{uid}.info"

    def create_info_folder(self, uid: str) -> Path:
        info_dir = self._info_dir(uid)
        info_dir.mkdir(parents=True, exist_ok=True)
        return info_dir

    def info_folder_exists(self, uid: str) -> bool:
        return self._info_dir(uid).exists()

    # ---------------------------------------------------------------- #
    #  Image entry persistence
    # ---------------------------------------------------------------- #

    def save_image_entry(
        self,
        uid: str,
        src_image_path: str,
        thumbnail: Image.Image,
        record: dict,
    ) -> None:
        info_dir = self._info_dir(uid)
        info_dir.mkdir(parents=True, exist_ok=True)

        src = Path(src_image_path)
        dest_image = info_dir / src.name
        if not dest_image.exists():
            shutil.copy2(src, dest_image)

        thumb_path = info_dir / THUMBNAIL_FILENAME
        thumbnail.save(str(thumb_path), format="PNG", optimize=True)

        if not record.get("original_filename"):
            record["original_filename"] = src.name

        # Backup JSON
        self._write_json(info_dir / METADATA_FILENAME, record)

        # Primary: SQLite
        with self._tx():
            self._db.execute(self._upsert_sql(), self._record_to_row(uid, record))

    # ---------------------------------------------------------------- #
    #  Metadata access
    # ---------------------------------------------------------------- #

    def load_image_metadata(self, uid: str) -> Optional[dict]:
        row = self._db.execute(
            "SELECT * FROM images WHERE uid=?", (uid,)
        ).fetchone()
        if row:
            return self._row_to_record(row)
        # Legacy fallback
        meta_file = self._info_dir(uid) / METADATA_FILENAME
        if meta_file.exists():
            return self._read_json(meta_file)
        return None

    def save_image_metadata(self, uid: str, record: dict) -> None:
        record["modified_at"] = datetime.utcnow().isoformat()
        with self._tx():
            self._db.execute(self._upsert_sql(), self._record_to_row(uid, record))
        meta_file = self._info_dir(uid) / METADATA_FILENAME
        if meta_file.exists():
            self._write_json(meta_file, record)

    # ---------------------------------------------------------------- #
    #  Paginated listing  (NEW — primary performance API)
    # ---------------------------------------------------------------- #

    def list_page(
        self,
        offset: int = 0,
        limit: int = 200,
        tag: str = "",
        sort: str = "import_at",
    ) -> List[dict]:
        """
        Return one page of image metadata dicts, directly from SQLite.

        Args:
            offset: row offset (0-based)
            limit:  rows per page
            tag:    if non-empty, filter to images containing this tag
            sort:   'import_at' | 'name' | 'created_at'
        """
        sort_col = sort if sort in ("import_at", "name", "created_at") else "import_at"
        if tag:
            rows = self._db.execute(
                f"SELECT * FROM images WHERE tags LIKE ?"
                f" ORDER BY {sort_col} DESC LIMIT ? OFFSET ?",
                (f'%"{tag}"%', limit, offset),
            ).fetchall()
        else:
            rows = self._db.execute(
                f"SELECT * FROM images ORDER BY {sort_col} DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def total_count(self, tag: str = "") -> int:
        if tag:
            return self._db.execute(
                "SELECT COUNT(*) FROM images WHERE tags LIKE ?",
                (f'%"{tag}"%',),
            ).fetchone()[0]
        return self._db.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    # ---------------------------------------------------------------- #
    #  Legacy enumeration (kept for FAISS indexer compatibility)
    # ---------------------------------------------------------------- #

    def list_all_ids(self) -> List[str]:
        if self._db:
            count = self._db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            if count > 0:
                rows = self._db.execute(
                    "SELECT uid FROM images ORDER BY import_at DESC"
                ).fetchall()
                return [r["uid"] for r in rows]
        return self._list_all_ids_fs()

    def _list_all_ids_fs(self) -> List[str]:
        if self._library_path is None:
            return []
        images_dir = self._library_path / IMAGES_SUBFOLDER
        if not images_dir.exists():
            return []
        dirs = [
            d for d in images_dir.iterdir()
            if d.is_dir() and d.name.endswith(".info")
        ]
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return [d.stem for d in dirs]

    def image_count(self) -> int:
        if self._db:
            return self._db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        return len(self._list_all_ids_fs())

    def list_recent_ids(self, n: int = 50) -> List[str]:
        if self._db:
            rows = self._db.execute(
                "SELECT uid FROM images ORDER BY import_at DESC LIMIT ?", (n,)
            ).fetchall()
            return [r["uid"] for r in rows]
        return self._list_all_ids_fs()[:n]

    def list_all_records(self) -> List[dict]:
        """Return all catalog records, newest first."""
        if not self._db:
            return []
        rows = self._db.execute(
            "SELECT * FROM images ORDER BY import_at DESC"
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # ---------------------------------------------------------------- #
    #  Path helpers
    # ---------------------------------------------------------------- #

    def get_thumbnail_path(self, uid: str) -> Optional[str]:
        thumb = self._info_dir(uid) / THUMBNAIL_FILENAME
        if thumb.exists():
            return str(thumb)
        info_dir = self._info_dir(uid)
        if info_dir.exists():
            for f in info_dir.iterdir():
                if f.name.endswith("_thumbnail.png"):
                    return str(f)
        return None

    def get_original_image_path(self, uid: str) -> Optional[str]:
        info_dir = self._info_dir(uid)
        if not info_dir.exists():
            return None
        if self._db:
            row = self._db.execute(
                "SELECT original_filename FROM images WHERE uid=?", (uid,)
            ).fetchone()
            if row and row["original_filename"]:
                candidate = info_dir / row["original_filename"]
                if candidate.exists():
                    return str(candidate)
        for f in info_dir.iterdir():
            if (f.suffix.lower() in SUPPORTED_EXTENSIONS
                    and f.name != THUMBNAIL_FILENAME
                    and not f.name.endswith("_thumbnail.png")):
                return str(f)
        return None

    # ---------------------------------------------------------------- #
    #  Tags
    # ---------------------------------------------------------------- #

    def list_tags(self) -> List[str]:
        return list(self._tags)

    def add_global_tag(self, tag: str) -> None:
        if tag and tag not in self._tags:
            self._tags.append(tag)
            if self._library_path:
                self._write_json(self._library_path / TAGS_FILENAME, self._tags)

    def update_image_tags(self, uid: str, tags: List[str]) -> None:
        record = self.load_image_metadata(uid)
        if record is not None:
            record["tags"] = tags
            self.save_image_metadata(uid, record)
            for t in tags:
                self.add_global_tag(t)
        self._rebuild_global_tags()

    def _rebuild_global_tags(self) -> None:
        if not self._db:
            return
        rows = self._db.execute("SELECT DISTINCT tags FROM images").fetchall()
        all_tags: List[str] = []
        for row in rows:
            try:
                for t in json.loads(row["tags"] or "[]"):
                    if t and t not in all_tags:
                        all_tags.append(t)
            except Exception:
                pass
        self._tags = all_tags
        if self._library_path:
            self._write_json(self._library_path / TAGS_FILENAME, self._tags)

    # ---------------------------------------------------------------- #
    #  Removal
    # ---------------------------------------------------------------- #

    def remove_image(self, uid: str, delete_source: bool = False) -> dict:
        record = self.load_image_metadata(uid) or {}
        source_path = record.get("source_path", "")
        result = {
            "uid": uid,
            "library_deleted": False,
            "source_deleted": False,
            "source_path": source_path,
            "source_error": "",
        }

        if delete_source and source_path:
            try:
                src = Path(source_path)
                if src.exists() and src.is_file() and not self._is_in_library(src):
                    src.unlink()
                    result["source_deleted"] = True
            except Exception as e:
                result["source_error"] = str(e)

        info_dir = self._info_dir(uid)
        if info_dir.exists():
            shutil.rmtree(info_dir)
            result["library_deleted"] = True
        if self._db:
            with self._tx():
                self._db.execute("DELETE FROM images WHERE uid=?", (uid,))
        self._rebuild_global_tags()
        return result

    def _is_in_library(self, path: Path) -> bool:
        if self._library_path is None:
            return False
        try:
            path.resolve().relative_to(self._library_path.resolve())
            return True
        except Exception:
            return False

    # ---------------------------------------------------------------- #
    #  Trash / soft-delete  (recoverable removal + undo)
    # ---------------------------------------------------------------- #
    #
    #  Soft delete moves an image's <UID>.info folder into a hidden
    #  {library}/.trash/ area and drops its catalog row, so it disappears
    #  from every listing and search immediately but nothing is lost on
    #  disk. A JSON manifest records the original record (plus any pending
    #  "also delete source" intent) so the operation can be fully undone,
    #  or later made permanent via purge_trashed().

    def _trash_dir(self) -> Optional[Path]:
        if self._library_path is None:
            return None
        return self._library_path / TRASH_SUBFOLDER

    def _trash_manifest_path(self) -> Optional[Path]:
        td = self._trash_dir()
        return td / TRASH_MANIFEST if td else None

    def _read_trash_manifest(self) -> dict:
        mp = self._trash_manifest_path()
        if mp is None or not mp.exists():
            return {}
        data = self._read_json(mp, default={})
        return data if isinstance(data, dict) else {}

    def _write_trash_manifest(self, manifest: dict) -> None:
        td = self._trash_dir()
        mp = self._trash_manifest_path()
        if td is None or mp is None:
            return
        td.mkdir(parents=True, exist_ok=True)
        self._write_json(mp, manifest)

    def soft_delete_image(self, uid: str, delete_source: bool = False) -> Optional[dict]:
        """
        Move an image into the trash and drop its catalog row.

        The source file on disk is NOT touched here; if ``delete_source`` is
        requested the intent is recorded and only honoured when the entry is
        permanently purged. Returns the archived record (for restore), or
        None if the image does not exist.
        """
        if self._library_path is None:
            return None
        record = self.load_image_metadata(uid)
        if record is None:
            return None

        td = self._trash_dir()
        if td is None:
            return None
        td.mkdir(parents=True, exist_ok=True)

        info_dir = self._info_dir(uid)
        dest_dir = td / f"{uid}.info"
        try:
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            if info_dir.exists():
                shutil.move(str(info_dir), str(dest_dir))
        except Exception:
            return None

        # Drop the catalog row so it vanishes from listings / search.
        if self._db:
            with self._tx():
                self._db.execute("DELETE FROM images WHERE uid=?", (uid,))
        self._rebuild_global_tags()

        manifest = self._read_trash_manifest()
        manifest[uid] = {
            "record":        record,
            "delete_source": bool(delete_source),
            "source_path":   record.get("source_path", ""),
            "name":          record.get("name", uid),
            "file_size":     int(record.get("file_size", 0) or 0),
            "deleted_at":    datetime.utcnow().isoformat(),
        }
        self._write_trash_manifest(manifest)
        return record

    def restore_trashed_image(self, uid: str) -> Optional[dict]:
        """
        Undo a soft delete: move the folder back and re-insert the catalog row.
        Returns the restored record, or None if the trash entry is missing.
        """
        if self._library_path is None:
            return None
        manifest = self._read_trash_manifest()
        entry = manifest.get(uid)
        if not entry:
            return None
        record = entry.get("record") or {}

        td = self._trash_dir()
        src_dir = (td / f"{uid}.info") if td else None
        dest_dir = self._info_dir(uid)
        try:
            if src_dir and src_dir.exists():
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.move(str(src_dir), str(dest_dir))
        except Exception:
            return None

        if self._db and record:
            with self._tx():
                self._db.execute(self._upsert_sql(), self._record_to_row(uid, record))
        self._rebuild_global_tags()

        manifest.pop(uid, None)
        self._write_trash_manifest(manifest)
        return record

    def purge_trashed_image(self, uid: str) -> dict:
        """
        Permanently delete a trashed image. Honours the ``delete_source``
        intent captured at soft-delete time. Returns a small status dict.
        """
        result = {
            "uid": uid,
            "purged": False,
            "source_deleted": False,
            "source_error": "",
        }
        manifest = self._read_trash_manifest()
        entry = manifest.get(uid)
        if entry is None:
            return result

        if entry.get("delete_source") and entry.get("source_path"):
            try:
                src = Path(entry["source_path"])
                if src.exists() and src.is_file() and not self._is_in_library(src):
                    src.unlink()
                    result["source_deleted"] = True
            except Exception as e:
                result["source_error"] = str(e)

        td = self._trash_dir()
        trashed_dir = (td / f"{uid}.info") if td else None
        try:
            if trashed_dir and trashed_dir.exists():
                shutil.rmtree(trashed_dir)
            result["purged"] = True
        except Exception as e:
            result["source_error"] = result["source_error"] or str(e)

        manifest.pop(uid, None)
        self._write_trash_manifest(manifest)
        return result

    def list_trashed(self) -> List[dict]:
        """Return trash entries (newest first) for a recycle-bin view."""
        manifest = self._read_trash_manifest()
        items = []
        for uid, entry in manifest.items():
            items.append({
                "uid":           uid,
                "name":          entry.get("name", uid),
                "file_size":     entry.get("file_size", 0),
                "delete_source": bool(entry.get("delete_source")),
                "deleted_at":    entry.get("deleted_at", ""),
            })
        items.sort(key=lambda x: x.get("deleted_at", ""), reverse=True)
        return items

    def trash_count(self) -> int:
        return len(self._read_trash_manifest())

    # ---------------------------------------------------------------- #
    #  Duplicate lookup helpers
    # ---------------------------------------------------------------- #

    def find_duplicate_candidates(
        self,
        file_hash: str = "",
        phash: str = "",
        phash_flip: str = "",
        max_distance: int = 10,
        exclude_uid: str = "",
        limit: int = 12,
    ) -> List[dict]:
        """Find existing exact/similar/flipped candidates for a new image."""
        if not self._db:
            return []
        rows = self._db.execute(
            "SELECT * FROM images WHERE file_hash<>'' OR phash<>'' OR phash_flip<>''"
        ).fetchall()
        candidates = []
        seen = set()
        for row in rows:
            rec = self._row_to_record(row)
            uid = rec.get("id", "")
            if not uid or uid == exclude_uid:
                continue
            best = None
            if file_hash and rec.get("file_hash") == file_hash:
                best = ("exact", 0)
            dist = _hash_distance(phash, rec.get("phash", ""))
            if dist is not None and dist <= max_distance:
                best = _best_duplicate(best, "similar", dist)
            flip_dists = [
                _hash_distance(phash, rec.get("phash_flip", "")),
                _hash_distance(phash_flip, rec.get("phash", "")),
            ]
            flip_dists = [d for d in flip_dists if d is not None]
            if flip_dists:
                dist = min(flip_dists)
                if dist <= max_distance:
                    best = _best_duplicate(best, "flipped", dist)
            if best and uid not in seen:
                seen.add(uid)
                candidates.append({
                    "uid": uid,
                    "kind": best[0],
                    "distance": best[1],
                    "name": rec.get("name", uid),
                })
        candidates.sort(key=lambda c: (c["distance"], c["kind"], c["uid"]))
        return candidates[:limit]

    # ---------------------------------------------------------------- #
    #  SQLite row ↔ record helpers
    # ---------------------------------------------------------------- #

    @staticmethod
    def _record_to_row(uid: str, rec: dict) -> tuple:
        def j(v, default="[]"):
            if v is None:
                return default
            return json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
        return (
            uid,
            rec.get("name", ""),
            rec.get("ext", ""),
            int(rec.get("width", 0)),
            int(rec.get("height", 0)),
            int(rec.get("file_size", 0)),
            j(rec.get("dominant_color", [128, 128, 128])),
            j(rec.get("color_palette", [])),
            j(rec.get("color_palette_ratio", [])),
            j(rec.get("tags", [])),
            rec.get("annotation", ""),
            rec.get("source_path", ""),
            rec.get("original_filename", ""),
            rec.get("ocr_text", ""),
            j(rec.get("exif", {}), default="{}"),
            j(rec.get("indexed_slots", {}), default="{}"),
            rec.get("file_hash", ""),
            rec.get("phash", ""),
            rec.get("phash_flip", ""),
            rec.get("duplicate_group", ""),
            rec.get("duplicate_kind", ""),
            rec.get("created_at", ""),
            rec.get("modified_at", ""),
            rec.get("import_at", ""),
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> dict:
        def p(v, default):
            try:
                return json.loads(v) if v else default
            except Exception:
                return default
        return {
            "id":                   row["uid"],
            "name":                 row["name"],
            "ext":                  row["ext"],
            "width":                row["width"],
            "height":               row["height"],
            "file_size":            row["file_size"],
            "dominant_color":       p(row["dominant_color"], [128, 128, 128]),
            "color_palette":        p(row["color_palette"], []),
            "color_palette_ratio":  p(row["color_palette_ratio"], []),
            "tags":                 p(row["tags"], []),
            "annotation":           row["annotation"],
            "source_path":          row["source_path"],
            "original_filename":    row["original_filename"],
            "ocr_text":             row["ocr_text"],
            "exif":                 p(row["exif"], {}),
            "indexed_slots":        p(row["indexed_slots"], {}),
            "file_hash":            row["file_hash"],
            "phash":                row["phash"],
            "phash_flip":           row["phash_flip"],
            "duplicate_group":      row["duplicate_group"],
            "duplicate_kind":       row["duplicate_kind"],
            "created_at":           row["created_at"],
            "modified_at":          row["modified_at"],
            "import_at":            row["import_at"],
        }

    @staticmethod
    def _insert_sql() -> str:
        return """
        INSERT OR IGNORE INTO images
          (uid,name,ext,width,height,file_size,
           dominant_color,color_palette,color_palette_ratio,
           tags,annotation,source_path,original_filename,ocr_text,
           exif,indexed_slots,file_hash,phash,phash_flip,
           duplicate_group,duplicate_kind,created_at,modified_at,import_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """

    @staticmethod
    def _upsert_sql() -> str:
        return """
        INSERT INTO images
          (uid,name,ext,width,height,file_size,
           dominant_color,color_palette,color_palette_ratio,
           tags,annotation,source_path,original_filename,ocr_text,
           exif,indexed_slots,file_hash,phash,phash_flip,
           duplicate_group,duplicate_kind,created_at,modified_at,import_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(uid) DO UPDATE SET
          name=excluded.name, ext=excluded.ext,
          width=excluded.width, height=excluded.height,
          file_size=excluded.file_size,
          dominant_color=excluded.dominant_color,
          color_palette=excluded.color_palette,
          color_palette_ratio=excluded.color_palette_ratio,
          tags=excluded.tags, annotation=excluded.annotation,
          source_path=excluded.source_path,
          original_filename=excluded.original_filename,
          ocr_text=excluded.ocr_text, exif=excluded.exif,
          indexed_slots=excluded.indexed_slots,
          file_hash=excluded.file_hash,
          phash=excluded.phash,
          phash_flip=excluded.phash_flip,
          duplicate_group=excluded.duplicate_group,
          duplicate_kind=excluded.duplicate_kind,
          modified_at=excluded.modified_at,
          import_at=excluded.import_at
        """

    # ---------------------------------------------------------------- #
    #  JSON helpers
    # ---------------------------------------------------------------- #

    @staticmethod
    def _write_json(path: Path, data) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            tmp.replace(path)
        except Exception:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            raise

    @staticmethod
    def _read_json(path: Path, default=None):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default if default is not None else {}

    @staticmethod
    def is_supported_image(path: str) -> bool:
        return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS
