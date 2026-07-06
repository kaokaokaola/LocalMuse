"""
Per-slot FAISS index persistence inside the library folder.

Architecture (v4 — slot-independent files):
  {library}/index_semantic.faiss  — FAISS serialized bytes (IndexFlatIP, 512D)
  {library}/index_semantic.map   — JSON UID list  ["uid1", "uid2", ...]
  {library}/index_sketch.faiss   — FAISS serialized bytes (IndexFlatL2, 4096D)
  {library}/index_sketch.map     — JSON UID list

  v3 精简：depth / pose 槽位已移除。旧库中的 index_depth.* /
  index_pose.* 文件会被忽略（不加载、不删除）。

Design principles:
  • Each slot is fully independent — replacing one model only requires
    rebuilding that slot's two files; all other slots stay untouched.
  • Atomic writes (write to .tmp → rename) prevent corruption.
  • Backward compatible with v3 single-file vectors.index:
    If index_semantic.faiss is absent but vectors.index exists, the old
    pickle is loaded and immediately migrated to the new per-slot format.
"""

from __future__ import annotations
import json
import pickle
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.vector_math import VectorIndex

# Slot specs: name → (dim, metric)
_INDEX_SPECS: Dict[str, Tuple[int, str]] = {
    "semantic": (512, "ip"),       # CLIP cosine
    "sketch": (4096, "l2"),        # Legacy Canny 64x64
    "sketch_fit": (16384, "l2"),   # Canny 128x128, long-side fit + pad
    "sketch_crop": (16384, "l2"),  # Canny 128x128, short-side fit + crop
}

_LEGACY_FILENAME = "vectors.index"   # v3 single-file pickle


def slot_faiss_path(lib_dir: Path, name: str) -> Path:
    return lib_dir / f"index_{name}.faiss"


def slot_map_path(lib_dir: Path, name: str) -> Path:
    return lib_dir / f"index_{name}.map"


class IndexStore:
    """
    Thread-safe wrapper around four independent named VectorIndex instances.

    Each slot is stored as two files inside the library directory:
      index_{name}.faiss — raw FAISS binary (from faiss.serialize_index)
      index_{name}.map   — JSON array of UIDs (parallel to FAISS row order)

    This allows model upgrades on a single slot: rebuild only
    index_{name}.faiss + index_{name}.map without touching any other slot.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._indices: Dict[str, VectorIndex] = {
            name: VectorIndex(dim, metric)
            for name, (dim, metric) in _INDEX_SPECS.items()
        }
        self._lib_dir: Optional[Path] = None

    # ---------------------------------------------------------------- #
    #  Persistence
    # ---------------------------------------------------------------- #

    def load(self, path: str) -> None:
        """
        Load slot index files from the library.

        ``path`` can be either:
          • The library directory itself (new usage)
          • The legacy ``vectors.index`` file path (old usage — auto-migrates)

        Safe to call if no index files exist (starts with empty indices).
        """
        p = Path(path)
        lib_dir = p if p.is_dir() else p.parent
        self._lib_dir = lib_dir

        # Prefer new per-slot format
        if any(slot_faiss_path(lib_dir, n).exists() for n in _INDEX_SPECS):
            self._load_slots(lib_dir)
        else:
            legacy = lib_dir / _LEGACY_FILENAME
            if legacy.exists():
                self._migrate_legacy(lib_dir, legacy)
            else:
                self._reset_indices()

    def save(self, path: Optional[str] = None) -> None:
        """
        Persist all slots to their individual .faiss + .map files.

        ``path`` can be either:
          • The library directory (new usage)
          • The legacy ``vectors.index`` file path (old usage — parent is used)

        Each slot is written atomically (temp-file + rename).
        """
        if path is not None:
            p = Path(path)
            lib_dir = p if p.is_dir() else p.parent
        elif self._lib_dir is not None:
            lib_dir = self._lib_dir
        else:
            raise ValueError("No library directory specified for IndexStore.save().")

        self._lib_dir = lib_dir
        lib_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            for name in _INDEX_SPECS:
                self._save_slot(lib_dir, name)

    # ---------------------------------------------------------------- #
    #  Persistence internals
    # ---------------------------------------------------------------- #

    def _load_slots(self, lib_dir: Path) -> None:
        """Load each slot from its .faiss + .map file pair."""
        with self._lock:
            for name, (dim, metric) in _INDEX_SPECS.items():
                fp = slot_faiss_path(lib_dir, name)
                mp = slot_map_path(lib_dir, name)
                if fp.exists() and mp.exists():
                    try:
                        fbytes  = fp.read_bytes()
                        uid_map = json.loads(mp.read_text(encoding="utf-8"))
                        self._indices[name] = VectorIndex.deserialize(
                            fbytes, uid_map, dim, metric
                        )
                    except Exception:
                        self._indices[name] = VectorIndex(dim, metric)
                else:
                    self._indices[name] = VectorIndex(dim, metric)

    def _migrate_legacy(self, lib_dir: Path, legacy_path: Path) -> None:
        """
        Load old v3 vectors.index pickle and immediately migrate to the
        new per-slot format. The legacy file is NOT deleted (kept as backup).
        """
        try:
            with open(legacy_path, "rb") as f:
                data = pickle.load(f)
            with self._lock:
                for name, (dim, metric) in _INDEX_SPECS.items():
                    b_key = f"{name}_bytes"
                    m_key = f"{name}_map"
                    # v2 backward compat: "structure" → "sketch" rename
                    if name == "sketch" and b_key not in data:
                        b_key, m_key = "structure_bytes", "structure_map"
                    fbytes = data.get(b_key)
                    fmap   = data.get(m_key, [])
                    if fbytes and fmap:
                        self._indices[name] = VectorIndex.deserialize(
                            fbytes, fmap, dim, metric
                        )
                    else:
                        self._indices[name] = VectorIndex(dim, metric)
            # Write new format immediately so next startup uses the fast path
            self.save(str(lib_dir))
        except Exception:
            self._reset_indices()

    def _save_slot(self, lib_dir: Path, name: str) -> None:
        """Atomically write one slot's .faiss and .map files."""
        fbytes, fmap, _, _ = self._indices[name].serialize()

        # FAISS binary
        faiss_path = slot_faiss_path(lib_dir, name)
        tmp_f = faiss_path.with_suffix(".faiss.tmp")
        tmp_f.write_bytes(fbytes)
        tmp_f.replace(faiss_path)

        # UID map (human-readable JSON)
        map_path = slot_map_path(lib_dir, name)
        tmp_m = map_path.with_suffix(".map.tmp")
        tmp_m.write_text(
            json.dumps(fmap, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_m.replace(map_path)

    # ---------------------------------------------------------------- #
    #  Mutation
    # ---------------------------------------------------------------- #

    def add_vectors(
        self,
        uid:          str,
        semantic_vec: np.ndarray,
        sketch_vec:   np.ndarray,
        sketch_fit_vec: Optional[np.ndarray] = None,
        sketch_crop_vec: Optional[np.ndarray] = None,
    ) -> None:
        """Add all slot vectors for one image. Vectors must already be normalized."""
        with self._lock:
            self._indices["semantic"].add(uid, semantic_vec)
            self._indices["sketch"].add(uid,   sketch_vec)
            if sketch_fit_vec is not None:
                self._indices["sketch_fit"].add(uid, sketch_fit_vec)
            if sketch_crop_vec is not None:
                self._indices["sketch_crop"].add(uid, sketch_crop_vec)

    def add_structure_vectors(
        self,
        uid: str,
        sketch_fit_vec: Optional[np.ndarray] = None,
        sketch_crop_vec: Optional[np.ndarray] = None,
    ) -> None:
        """Supplement the scheme-B structure slots for an existing image."""
        with self._lock:
            if sketch_fit_vec is not None:
                self._indices["sketch_fit"].add(uid, sketch_fit_vec)
            if sketch_crop_vec is not None:
                self._indices["sketch_crop"].add(uid, sketch_crop_vec)

    def remove_uid(self, uid: str, save: bool = True) -> Dict[str, int]:
        """
        Remove a UID from every slot and persist the updated index files.

        FAISS flat indexes do not have cheap per-row deletion with our parallel
        UID maps, so each slot is rebuilt without the removed rows.
        """
        with self._lock:
            removed = {
                name: self._indices[name].remove_uid(uid)
                for name in _INDEX_SPECS
            }
        if save and self._lib_dir is not None and any(removed.values()):
            self.save(str(self._lib_dir))
        return removed

    def remove_uids(self, uids, save: bool = True) -> Dict[str, int]:
        """
        Remove several UIDs from every slot in a single rebuild per slot.

        Much cheaper than calling :meth:`remove_uid` in a loop, because each
        flat FAISS slot is reconstructed only once regardless of how many
        UIDs are removed. Returns {slot_name: removed_row_count}.
        """
        uid_set = set(uids)
        if not uid_set:
            return {name: 0 for name in _INDEX_SPECS}
        with self._lock:
            removed = {
                name: self._indices[name].remove_uids(uid_set)
                for name in _INDEX_SPECS
            }
        if save and self._lib_dir is not None and any(removed.values()):
            self.save(str(self._lib_dir))
        return removed

    # ---------------------------------------------------------------- #
    #  Capture / restore — used by soft-delete + undo
    # ---------------------------------------------------------------- #

    def capture_vectors(self, uid: str) -> Dict[str, np.ndarray]:
        """
        Reconstruct and return every stored vector for a UID, keyed by slot.

        Called just before a soft delete so the exact vectors can be re-added
        on undo without recomputing them from the image. Slots that do not
        contain the UID are simply omitted.
        """
        captured: Dict[str, np.ndarray] = {}
        with self._lock:
            for name in _INDEX_SPECS:
                vec = self._indices[name].get_vector(uid)
                if vec is not None:
                    captured[name] = np.array(vec, dtype=np.float32)
        return captured

    def restore_vectors(
        self,
        uid: str,
        vectors: Dict[str, np.ndarray],
        save: bool = True,
    ) -> bool:
        """
        Re-add previously captured vectors for a UID (undo of a soft delete).

        Returns True if at least one slot vector was restored. Skips slots
        that already contain the UID so a double-undo cannot duplicate rows.
        """
        if not vectors:
            return False
        added = False
        with self._lock:
            for name, vec in vectors.items():
                if name not in self._indices:
                    continue
                if uid in set(self._indices[name].all_ids()):
                    continue
                self._indices[name].add(uid, np.array(vec, dtype=np.float32))
                added = True
        if added and save and self._lib_dir is not None:
            self.save(str(self._lib_dir))
        return added

    # ---------------------------------------------------------------- #
    #  Search — one method per slot
    # ---------------------------------------------------------------- #

    def search_semantic(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        with self._lock:
            return self._indices["semantic"].search(query, k)

    def search_sketch(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        with self._lock:
            return self._indices["sketch"].search(query, k)

    def search_sketch_fit(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        with self._lock:
            return self._indices["sketch_fit"].search(query, k)

    def search_sketch_crop(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        with self._lock:
            return self._indices["sketch_crop"].search(query, k)

    # ---------------------------------------------------------------- #
    #  Stats
    # ---------------------------------------------------------------- #

    def semantic_size(self) -> int:
        return self._indices["semantic"].size()

    def all_ids(self) -> List[str]:
        return self._indices["semantic"].all_ids()

    def is_empty(self) -> bool:
        return self._indices["semantic"].size() == 0

    def slot_sizes(self) -> Dict[str, int]:
        """Number of indexed vectors per slot (useful for diagnostics)."""
        with self._lock:
            return {name: self._indices[name].size() for name in _INDEX_SPECS}

    def slot_ids(self, name: str) -> List[str]:
        """UID map for one slot, used by diagnostics and supplement jobs."""
        with self._lock:
            if name not in self._indices:
                return []
            return self._indices[name].all_ids()

    def semantic_vector(self, uid: str) -> Optional[np.ndarray]:
        """Reconstruct the stored 512-D CLIP vector for one image (or None)."""
        with self._lock:
            return self._indices["semantic"].get_vector(uid)

    def has_uid(self, name: str, uid: str) -> bool:
        """Return whether a slot already contains a UID."""
        with self._lock:
            if name not in self._indices:
                return False
            return uid in set(self._indices[name].all_ids())

    # ---------------------------------------------------------------- #
    #  Reset
    # ---------------------------------------------------------------- #

    def reset(self) -> None:
        with self._lock:
            self._reset_indices()
            self._lib_dir = None

    def _reset_indices(self) -> None:
        self._indices = {
            name: VectorIndex(dim, metric)
            for name, (dim, metric) in _INDEX_SPECS.items()
        }
