"""
LocalMuse V3 — FastAPI server（精简版）.

检索三模态：semantic（CLIP 文本/图像）、structure（草图轮廓）、color。
反馈机制：moodboard 收藏 + 会话内排除 + append-only 行为日志。

Communication:
  - REST endpoints for request/response operations
  - WebSocket (/ws) for server-push events
"""

from __future__ import annotations
import asyncio
import base64
import io
import json
import os
import threading
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.config.settings import Settings
from src.core import clip_model as clip_module
from src.core import image_proc
from src.infra.library_mgr import LibraryManager
from src.infra.index_store import IndexStore
from src.services import auto_tagger
from src.services.indexer import IndexWorker
from src.services.searcher import (
    SearchWorker,
    make_query_context,
    _load_query_image,
    _to_rgb_on_white,
)
from src.services.session_tracker import get_session, reset_session

# ------------------------------------------------------------------ #
#  Module-level state (injected from main.py via initialize())
# ------------------------------------------------------------------ #

_lib: Optional[LibraryManager] = None
_idx: Optional[IndexStore]     = None
_cfg: Optional[Settings]       = None

_loop: Optional[asyncio.AbstractEventLoop] = None
_ws_clients: Set[WebSocket] = set()
_indexing  = False
_searching = False
_dataset_running = False
_duplicate_running = False
_tagging_running = False

# Captured FAISS vectors for soft-deleted images, keyed by uid. Lets an undo
# restore the exact vectors without recomputing them. Populated on batch
# delete, drained on restore / purge.
_trash_vectors: Dict[str, Dict[str, "object"]] = {}

_HTML_PATH = Path(__file__).parent / "ui" / "frontend" / "app.html"
_APP_ROOT  = Path(__file__).parent.parent  # project root directory


def initialize(lib: LibraryManager, idx: IndexStore, cfg: Settings) -> None:
    """Called once from main.py to inject shared state into this module."""
    global _lib, _idx, _cfg
    _lib = lib
    _idx = idx
    _cfg = cfg
    # 若 main.py 已恢复上次图库，则同步反馈日志路径
    try:
        if lib.is_open and lib.library_path:
            get_session().set_feedback_log_path(str(lib.library_path))
    except Exception:
        pass


# ------------------------------------------------------------------ #
#  FastAPI app
# ------------------------------------------------------------------ #

app = FastAPI(title="LocalMuse V2")


@app.on_event("startup")
async def _startup() -> None:
    global _loop
    _loop = asyncio.get_running_loop()
    threading.Thread(target=_preload_clip, daemon=True).start()


# ------------------------------------------------------------------ #
#  WebSocket — server-push events
# ------------------------------------------------------------------ #

def _push(event: str, data) -> None:
    """Thread-safe: broadcast an event to all connected browser clients."""
    if _loop is None:
        return
    msg = json.dumps({"event": event, "data": data})
    asyncio.run_coroutine_threadsafe(_broadcast(msg), _loop)


async def _broadcast(msg: str) -> None:
    dead: Set[WebSocket] = set()
    for ws in list(_ws_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    _ws_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        _ws_clients.discard(ws)
    except Exception:
        _ws_clients.discard(ws)


# ------------------------------------------------------------------ #
#  Frontend & file serving
# ------------------------------------------------------------------ #

@app.get("/")
async def index():
    return FileResponse(
        _HTML_PATH,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/thumb/{uid}")
async def get_thumbnail(uid: str):
    path = _lib.get_thumbnail_path(uid)
    if path and os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/original/{uid}")
async def get_original(uid: str):
    path = _lib.get_original_image_path(uid)
    if path and os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "not found"}, status_code=404)


# ------------------------------------------------------------------ #
#  Library management
# ------------------------------------------------------------------ #

@app.get("/api/library/info")
async def library_info() -> dict:
    return _lib_info()


@app.post("/api/library/open-dialog")
async def open_library_dialog():
    """Open native folder picker, then load the selected library."""
    loop = asyncio.get_running_loop()
    path = await loop.run_in_executor(None, _pick_folder, "Open Library", "")
    if not path:
        _push("dialog_event", {"type": "cancelled"})
        return {"cancelled": True}
    try:
        _lib.open_library(path)
        _idx.reset()
        if _lib.has_index():
            _idx.load(str(_lib.index_dir))
        _cfg.last_library_path = path
        get_session().set_feedback_log_path(str(_lib.library_path))
        info = _lib_info()
        _push("library_changed", info)
        _push("status", f"Library '{_lib.library_name}' opened "
                        f"({_idx.semantic_size()} images indexed).")
        _do_show_all_push()
        return info
    except Exception as e:
        _push("status", f"Error opening library: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/library/close")
async def close_library():
    _lib.close_library()
    _idx.reset()
    _push("library_changed", {"is_open": False, "name": "", "count": 0,
                               "path": "", "tags": []})
    _push("all_images", [])
    _push("status", "Library closed.")
    return {"ok": True}


class _CreateLibReq(BaseModel):
    name: str
    location: str


@app.post("/api/library/create")
async def create_library(req: _CreateLibReq):
    try:
        lib_path = _lib.create_library(req.location, req.name)
        _idx.reset()
        _cfg.last_library_path = str(lib_path)
        get_session().set_feedback_log_path(str(lib_path))
        info = _lib_info()
        _push("library_changed", info)
        _push("status", f"Library '{req.name}' created.")
        _push("all_images", [])
        return info
    except FileExistsError:
        return JSONResponse(
            {"error": "A library with that name already exists at that location."},
            status_code=409,
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/api/library/dataset/check")
async def library_dataset_check():
    if not _lib.is_open:
        return JSONResponse({"error": "No library open"}, status_code=400)
    report = _build_dataset_report()
    _push("dataset_check", report)
    _push("status", _dataset_report_summary(report))
    return report


@app.post("/api/library/dataset/supplement")
async def library_dataset_supplement():
    global _dataset_running
    if not _lib.is_open:
        return JSONResponse({"error": "No library open"}, status_code=400)
    if _indexing:
        return JSONResponse({"error": "Indexing is running"}, status_code=409)
    if _dataset_running:
        return {"busy": True}
    _dataset_running = True
    threading.Thread(target=_run_structure_supplement, daemon=True).start()
    return {"started": True}


@app.get("/api/library/duplicates/check")
async def library_duplicates_check(min_similarity: int = 84):
    """Report duplicate groups; ``min_similarity`` is a 50–100 percentage."""
    if not _lib.is_open:
        return JSONResponse({"error": "No library open"}, status_code=400)
    report = _build_duplicate_report(min_similarity)
    # NOTE: no WS push here — the caller already gets the report in the
    # response. A broadcast would re-render the panel asynchronously and
    # can arrive mid-bulk-delete, corrupting the list.
    _push("status", _duplicate_report_summary(report))
    return report


@app.post("/api/library/duplicates/scan")
async def library_duplicates_scan():
    global _duplicate_running
    if not _lib.is_open:
        return JSONResponse({"error": "No library open"}, status_code=400)
    if _indexing:
        return JSONResponse({"error": "Indexing is running"}, status_code=409)
    if _duplicate_running:
        return {"busy": True}
    _duplicate_running = True
    threading.Thread(target=_run_duplicate_scan, daemon=True).start()
    return {"started": True}


@app.get("/api/app-root")
async def get_app_root():
    return {"path": str(_APP_ROOT.resolve())}


@app.post("/api/dialog/browse-folder")
async def browse_folder():
    loop = asyncio.get_running_loop()
    path = await loop.run_in_executor(
        None, _pick_folder, "Select Folder", str(_APP_ROOT.resolve())
    )
    if path:
        _push("dialog_event", {"type": "browse_location", "path": path})
        return {"path": path}
    _push("dialog_event", {"type": "browse_location", "path": ""})
    return {"path": ""}


# ------------------------------------------------------------------ #
#  Browse
# ------------------------------------------------------------------ #

@app.get("/api/library/images")
async def show_all():
    """
    Legacy WebSocket-push endpoint (kept for compatibility).
    Now only pushes the first page and total count.
    The frontend uses GET /api/images for pagination.
    """
    if not _lib.is_open:
        _push("all_images", [])
        return {"items": [], "total": 0}
    total = _lib.total_count()
    records = _lib.list_page(offset=0, limit=200)
    items = [_make_item_from_record(r) for r in records]
    _push("all_images", items)
    _push("library_total", {"total": total, "page_size": 200})
    _push("status", f"Library: {total} images.")
    return {"items": items, "total": total}


@app.get("/api/images")
async def get_images_page(
    offset: int = 0,
    limit: int = 200,
    tag: str = "",
    sort: str = "import_at",
):
    """
    Paginated image listing — primary endpoint for the grid.

    Returns JSON:
      { "items": [...], "total": int, "offset": int, "limit": int }
    """
    if not _lib.is_open:
        return {"items": [], "total": 0, "offset": offset, "limit": limit}
    limit  = min(limit, 500)   # cap per-request size
    total  = _lib.total_count(tag=tag)
    records = _lib.list_page(offset=offset, limit=limit, tag=tag, sort=sort)
    items  = [_make_item_from_record(r) for r in records]
    return {"items": items, "total": total, "offset": offset, "limit": limit}


@app.get("/api/library/recent")
async def show_recent(n: int = 50):
    if not _lib.is_open:
        return {"items": [], "total": 0}
    records = _lib.list_page(offset=0, limit=n, sort="import_at")
    items   = [_make_item_from_record(r) for r in records]
    _push("all_images", items)
    return {"items": items, "total": len(items)}


@app.get("/api/library/filter")
async def filter_by_tag(tag: str = ""):
    """Tag filter — now delegates to paginated /api/images."""
    if not _lib.is_open:
        return {"items": [], "total": 0}
    total   = _lib.total_count(tag=tag)
    records = _lib.list_page(offset=0, limit=200, tag=tag)
    items   = [_make_item_from_record(r) for r in records]
    _push("all_images", items)
    _push("library_total", {"total": total, "page_size": 200, "tag": tag})
    return {"items": items, "total": total}


# ------------------------------------------------------------------ #
#  Import / indexing
# ------------------------------------------------------------------ #

@app.post("/api/import/dialog")
async def import_dialog():
    global _indexing
    if not _lib.is_open:
        _push("status", "error:No library open. Create or open a library first.")
        return {"error": "No library open"}
    if _indexing:
        return {"error": "Already indexing"}

    loop = asyncio.get_running_loop()
    paths = await loop.run_in_executor(None, _pick_files, "Import Images")
    if not paths:
        return {"cancelled": True}

    threading.Thread(target=_run_indexing, args=(list(paths),), daemon=True).start()
    return {"started": True, "count": len(paths)}


@app.post("/api/import/folder-dialog")
async def import_folder_dialog():
    """Open a folder picker; recursively collect all images and index them."""
    global _indexing
    if not _lib.is_open:
        _push("status", "error:No library open. Create or open a library first.")
        return {"error": "No library open"}
    if _indexing:
        return {"error": "Already indexing"}

    loop = asyncio.get_running_loop()
    folder = await loop.run_in_executor(None, _pick_folder, "Select Image Folder", "")
    if not folder:
        return {"cancelled": True}

    _push("status", f"Scanning folder: {folder} …")
    paths = await loop.run_in_executor(None, _scan_folder_images, folder)
    if not paths:
        _push("status", f"No images found in: {folder}")
        return {"found": 0}

    _push("status", f"Found {len(paths)} images — indexing…")
    threading.Thread(target=_run_indexing, args=(paths,), daemon=True).start()
    return {"started": True, "count": len(paths), "folder": folder}


def _run_indexing(paths: List[str]) -> None:
    global _indexing
    _indexing = True
    _push("indexing_progress", {"current": 0, "total": len(paths), "msg": "Starting..."})

    worker = IndexWorker(
        lib=_lib,
        idx=_idx,
        settings=_cfg,
        on_progress=lambda c, t, m: _push("indexing_progress",
                                          {"current": c, "total": t, "msg": m}),
        on_item_indexed=_on_item_indexed,
        on_finished=_on_index_finished,
        on_error=_on_index_error,
    )
    worker.run(paths)


def _on_item_indexed(uid: str) -> None:
    meta = _lib.load_image_metadata(uid)
    if meta:
        thumb = _lib.get_thumbnail_path(uid)
        item  = _make_item(uid, meta, thumb)
        _push("dialog_event", {"type": "item_indexed", "item": item})


def _on_index_finished(added: int, skipped: int) -> None:
    global _indexing
    _indexing = False
    _push("indexing_finished", {"added": added, "skipped": skipped})
    _push("library_changed", _lib_info())
    _push("status", f"Import complete: {added} added, {skipped} skipped.")


def _on_index_error(msg: str) -> None:
    global _indexing
    _indexing = False
    _push("indexing_error", msg)


# ------------------------------------------------------------------ #
#  Search
# ------------------------------------------------------------------ #

class _SearchReq(BaseModel):
    text: str = ""
    color: Optional[List[int]] = None
    sketch_path: Optional[str] = None
    sketch_data_url: str = ""
    enabled_modalities: Optional[List[str]] = None
    top_k: Optional[int] = 200


@app.post("/api/search")
async def search(req: _SearchReq):
    global _searching

    if not _lib.is_open:
        _push("status", "error:No library open. Please open or create a library first.")
        _push("search_results", [])
        return {"error": "no_library"}

    if _indexing:
        _push("status", "Indexing in progress — please wait and try again.")
        _push("search_results", [])
        return {"error": "indexing"}

    if _searching:
        _push("status", "Search in progress — please wait...")
        return {"busy": True}

    text = (req.text or "").strip()
    color = tuple(req.color) if req.color else None

    if not any([text, color, req.sketch_path, req.sketch_data_url]):
        _do_show_all_push()
        return []

    _searching = True
    _push("status", "Searching...")

    ctx = make_query_context(
        text=text,
        color=color,
        sketch_path=req.sketch_path,
        sketch_data_url=req.sketch_data_url,
        enabled_modalities=req.enabled_modalities,
        top_k=req.top_k,
    )
    threading.Thread(target=_run_search, args=(ctx, text), daemon=True).start()
    return {"started": True}


def _run_search(ctx: dict, raw_text: str) -> None:
    global _searching
    try:
        session = get_session()
        session.record_search(raw_text, list(ctx.get("enabled_modalities") or []))

        worker  = SearchWorker(_lib, _idx, _cfg)
        results = worker.run_sync(ctx)
        items   = [
            _make_item(
                r["uid"],
                r.get("meta", {}),
                r.get("thumbnail_path"),
                r.get("score", 0.0),
                r.get("per_modal", {}),
            )
            for r in results
            if not session.is_excluded(r["uid"])
        ]
        _searching = False

        _push("search_results", items)
        _push("status", _search_status_message(ctx, len(items)))

    except Exception as e:
        _searching = False
        _push("search_results", [])
        _push("status", f"Search error: {e}")


@app.post("/api/search/sketch")
async def search_by_sketch(payload: dict):
    data_url: str = payload.get("data_url", "")
    if not _lib.is_open or _idx.is_empty():
        _push("sketch_results", [])
        return []
    try:
        ctx = make_query_context(
            sketch_data_url=data_url,
            enabled_modalities=["sketch"],
            top_k=max(_cfg.top_k, 200),
        )
        worker = SearchWorker(_lib, _idx, _cfg)
        results = worker.run_sync(ctx)
        items = [
            _make_item(
                r["uid"],
                r.get("meta", {}),
                r.get("thumbnail_path"),
                r.get("score", 0.0),
                r.get("per_modal", {}),
            )
            for r in results
        ]

        _push("sketch_results", items)
        _push("status", f"Sketch search: {len(items)} matches.")
        return items
    except Exception as e:
        _push("sketch_results", [])
        _push("status", f"Sketch search error: {e}")
        return []


# ------------------------------------------------------------------ #
#  Moodboard
# ------------------------------------------------------------------ #

@app.get("/api/moodboard")
async def get_moodboard():
    session = get_session()
    return {"items": session.moodboard}


class _MoodboardAddReq(BaseModel):
    uid: str
    item: dict


@app.post("/api/moodboard/add")
async def moodboard_add(req: _MoodboardAddReq):
    session = get_session()
    session.record_save(req.uid, req.item)
    _push("moodboard_updated", {"items": session.moodboard})
    return {"ok": True, "count": len(session.moodboard)}


@app.delete("/api/moodboard/{uid}")
async def moodboard_remove(uid: str):
    session = get_session()
    session.record_remove_from_moodboard(uid)
    _push("moodboard_updated", {"items": session.moodboard})
    return {"ok": True, "count": len(session.moodboard)}


# ------------------------------------------------------------------ #
#  Behavior signals（只保留显式行为：expand / exclude）
# ------------------------------------------------------------------ #

class _BehaviorReq(BaseModel):
    uid: str
    signal: str          # "expand" | "exclude"


@app.post("/api/behavior")
async def record_behavior(req: _BehaviorReq):
    """Record an explicit user behavior signal (append-only feedback log)."""
    session = get_session()
    if req.signal == "expand":
        session.record_expand(req.uid)
    elif req.signal == "exclude":
        session.record_exclude(req.uid)
        _push("moodboard_updated", {"items": session.moodboard})
    return {"ok": True}


# ------------------------------------------------------------------ #
#  Session state
# ------------------------------------------------------------------ #

@app.get("/api/session")
async def get_session_state():
    """Return current session state."""
    return get_session().get_session_state()


@app.post("/api/session/reset")
async def session_reset():
    session = reset_session()
    if _lib and _lib.is_open and _lib.library_path:
        session.set_feedback_log_path(str(_lib.library_path))
    return {"ok": True}


# ------------------------------------------------------------------ #
#  Settings
# ------------------------------------------------------------------ #

@app.get("/api/settings")
async def get_settings() -> dict:
    return _build_settings_dict()


@app.post("/api/settings")
async def save_settings(data: dict):
    for key, val in data.items():
        _cfg.set(key, val)
    _push("status", "Settings saved.")
    result = _build_settings_dict()
    _push("settings_data", result)
    return result


# ------------------------------------------------------------------ #
#  Image operations
# ------------------------------------------------------------------ #

@app.post("/api/image/{uid}/open-location")
async def open_image_location(uid: str):
    orig = _lib.get_original_image_path(uid)
    if orig:
        import subprocess, sys
        folder = os.path.dirname(orig)
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
    return {"ok": True}


@app.delete("/api/image/{uid}")
async def remove_image(uid: str, delete_source: bool = False):
    if not _lib.is_open:
        return JSONResponse({"error": "No library open"}, status_code=400)
    if _lib.load_image_metadata(uid) is None:
        return JSONResponse({"error": "Image not found"}, status_code=404)
    result = _lib.remove_image(uid, delete_source=delete_source)
    index_removed = _idx.remove_uid(uid) if _idx else {}
    get_session().record_remove_from_moodboard(uid)
    _push("library_changed", _lib_info())
    _push("moodboard_updated", {"items": get_session().moodboard})
    _push("image_removed", {"uid": uid, "index_removed": index_removed, **result})
    msg = "Image removed from library."
    if result.get("source_deleted"):
        msg = "Image removed from library and source file."
    elif result.get("source_error"):
        msg = f"Image removed from library. Source delete failed: {result['source_error']}"
    _push("status", msg)
    return {"ok": True, "index_removed": index_removed, **result}


class _BatchDeleteReq(BaseModel):
    uids: List[str]
    delete_source: bool = False


@app.post("/api/images/delete")
async def batch_delete_images(req: _BatchDeleteReq):
    """
    Soft-delete a batch of images (recoverable). Each image is moved to the
    library trash and removed from every FAISS slot in a single rebuild;
    the exact vectors are captured so the operation can be undone.

    Response:
      {ok, deleted:[{uid,name,file_size}], failed:[uid], freed_bytes, trash_count}
    """
    if not _lib.is_open:
        return JSONResponse({"error": "No library open"}, status_code=400)

    uids = [u for u in (req.uids or []) if u]
    deleted: List[dict] = []
    failed: List[str] = []
    freed_bytes = 0
    removed_uids: List[str] = []

    for uid in uids:
        meta = _lib.load_image_metadata(uid)
        if meta is None:
            failed.append(uid)
            continue
        # Capture vectors BEFORE the folder/row disappears so undo is exact.
        vecs = _idx.capture_vectors(uid) if _idx else {}
        record = _lib.soft_delete_image(uid, delete_source=req.delete_source)
        if record is None:
            failed.append(uid)
            continue
        _trash_vectors[uid] = vecs
        removed_uids.append(uid)
        freed_bytes += int(record.get("file_size", 0) or 0)
        deleted.append({
            "uid":       uid,
            "name":      record.get("name", uid),
            "file_size": int(record.get("file_size", 0) or 0),
        })
        get_session().record_remove_from_moodboard(uid)

    # One rebuild per FAISS slot for the whole batch, then persist once.
    if removed_uids and _idx:
        _idx.remove_uids(removed_uids, save=True)

    # Keep every open client in sync (main grid, moodboard, etc.).
    for uid in removed_uids:
        _push("image_removed", {"uid": uid, "soft": True})
    if removed_uids:
        _push("library_changed", _lib_info())
        _push("moodboard_updated", {"items": get_session().moodboard})

    _push("images_deleted", {
        "deleted":      deleted,
        "failed":       failed,
        "freed_bytes":  freed_bytes,
        "trash_count":  _lib.trash_count(),
    })
    return {
        "ok":          True,
        "deleted":     deleted,
        "failed":      failed,
        "freed_bytes": freed_bytes,
        "trash_count": _lib.trash_count(),
    }


class _RestoreReq(BaseModel):
    uids: List[str]


@app.post("/api/images/restore")
async def restore_images(req: _RestoreReq):
    """Undo a soft delete: move images back and re-add their vectors."""
    if not _lib.is_open:
        return JSONResponse({"error": "No library open"}, status_code=400)

    restored: List[dict] = []
    failed: List[str] = []
    for uid in (req.uids or []):
        record = _lib.restore_trashed_image(uid)
        if record is None:
            failed.append(uid)
            continue
        # Fast path: re-add the captured vectors. Fallback: recompute from
        # the restored image copy (e.g. after a server restart cleared cache).
        vecs = _trash_vectors.pop(uid, None)
        if not vecs and _idx:
            vecs = _recompute_vectors_for(uid)
        if vecs and _idx:
            _idx.restore_vectors(uid, vecs, save=True)
        restored.append({"uid": uid, "name": record.get("name", uid)})

    for entry in restored:
        _push("image_restored", {"uid": entry["uid"]})
    if restored:
        _push("library_changed", _lib_info())

    _push("images_restored", {
        "restored":    restored,
        "failed":      failed,
        "trash_count": _lib.trash_count(),
    })
    return {"ok": True, "restored": restored, "failed": failed,
            "trash_count": _lib.trash_count()}


@app.get("/api/trash")
async def list_trash():
    if not _lib.is_open:
        return JSONResponse({"error": "No library open"}, status_code=400)
    return {"items": _lib.list_trashed(), "count": _lib.trash_count()}


class _PurgeReq(BaseModel):
    uids: Optional[List[str]] = None


@app.post("/api/trash/purge")
async def purge_trash(req: _PurgeReq):
    """Permanently delete trashed images (all, if no uids given)."""
    if not _lib.is_open:
        return JSONResponse({"error": "No library open"}, status_code=400)
    targets = req.uids if req.uids else [e["uid"] for e in _lib.list_trashed()]
    results = []
    for uid in targets:
        results.append(_lib.purge_trashed_image(uid))
        _trash_vectors.pop(uid, None)
    _push("trash_updated", {"count": _lib.trash_count()})
    return {"ok": True, "purged": results, "trash_count": _lib.trash_count()}


def _recompute_vectors_for(uid: str) -> dict:
    """
    Rebuild a soft-deleted image's slot vectors from its restored file copy.
    Used only when the in-memory capture is gone (e.g. after a restart).
    Mirrors the indexer's vector construction.
    """
    try:
        import numpy as np
        from src.core.vector_math import normalize
        path = _lib.get_original_image_path(uid)
        if not path or not os.path.exists(path):
            return {}
        img = Image.open(path).convert("RGB")
        clip_model = clip_module.get_clip_model()
        vecs: dict = {"semantic": normalize(clip_model.encode_image(img))}
        try:
            edge_img = image_proc.extract_edges(img)
            edge_gray = edge_img.convert("L").resize((64, 64))
            edge_arr = np.array(edge_gray, dtype=np.float32) / 255.0
            vecs["sketch"] = normalize(edge_arr.flatten())
            fit_vec, crop_vec = image_proc.structure_dual_vectors_from_edges(edge_img)
            vecs["sketch_fit"] = fit_vec
            vecs["sketch_crop"] = crop_vec
        except Exception:
            pass
        return vecs
    except Exception:
        return {}


@app.post("/api/image/{uid}/tags")
async def update_tags(uid: str, payload: dict):
    tags = payload.get("tags", [])
    _lib.update_image_tags(uid, tags)
    _push("library_changed", _lib_info())
    return {"ok": True}


# ------------------------------------------------------------------ #
#  Sketch local match — 局部边缘匹配可视化
# ------------------------------------------------------------------ #

class _SketchMatchReq(BaseModel):
    sketch_data_url: str = ""


@app.post("/api/image/{uid}/sketch-match")
def sketch_match(uid: str, req: _SketchMatchReq):
    """
    Compare the user's sketch against the target image's Canny edge map.
    Returns the edge map (data URL) plus the best-matching local region.

    Sync ``def`` — FastAPI runs it in a threadpool, so CPU work is fine.
    """
    if not _lib or not _lib.is_open:
        return JSONResponse({"error": "No library open"}, status_code=400)
    if not req.sketch_data_url:
        return JSONResponse({"error": "No sketch provided"}, status_code=400)

    path = _lib.get_original_image_path(uid)
    if not path or not os.path.exists(path):
        path = _lib.get_thumbnail_path(uid)
    if not path or not os.path.exists(path):
        return JSONResponse({"error": "Image not found"}, status_code=404)

    try:
        with Image.open(path) as im:
            img = im.convert("RGB")

        # Matching always runs at 512 (algorithm behaviour + speed unchanged).
        edge_img = image_proc.extract_edges(img)
        sketch = _to_rgb_on_white(_load_query_image(None, req.sketch_data_url))
        match  = image_proc.sketch_local_match(edge_img, sketch)

        # Display edge map: sharper, up to 1024 (never upscale past the original).
        disp_side = max(512, min(1024, max(img.size)))
        disp_img = (
            image_proc.extract_edges(img, long_side=disp_side)
            if disp_side != 512 else edge_img
        )

        buf = io.BytesIO()
        disp_img.save(buf, format="PNG")
        edge_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        result = {
            "uid": uid,
            "edge_url": edge_url,
            "width": disp_img.width,
            "height": disp_img.height,
            "bbox": None,
            "score": 0.0,
            "full_score": 0.0,
            "sketch_crop": None,
        }
        if match:
            result.update(match)
            if result.get("bbox") and disp_img is not edge_img:
                # Rescale match-space (512) bbox into display-space coordinates.
                sx = disp_img.width / edge_img.width
                sy = disp_img.height / edge_img.height
                bx, by, bw, bh = result["bbox"]
                result["bbox"] = [
                    int(round(bx * sx)), int(round(by * sy)),
                    int(round(bw * sx)), int(round(bh * sy)),
                ]
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------------------------------------ #
#  Auto tagging — 建筑限定词表 zero-shot 批量打标
# ------------------------------------------------------------------ #

@app.post("/api/library/auto-tag")
async def library_auto_tag(overwrite: bool = False):
    """Batch zero-shot tagging using stored CLIP vectors (no re-encoding)."""
    global _tagging_running
    if not _lib.is_open:
        return JSONResponse({"error": "No library open"}, status_code=400)
    if _indexing:
        return JSONResponse({"error": "Indexing is running"}, status_code=409)
    if _tagging_running:
        return {"busy": True}
    _tagging_running = True
    threading.Thread(target=_run_auto_tag, args=(overwrite,), daemon=True).start()
    return {"started": True}


def _run_auto_tag(overwrite: bool = False) -> None:
    global _tagging_running
    tagged = 0
    skipped = 0
    try:
        uids = _idx.slot_ids("semantic")
        total = len(uids)
        _push("tagging_progress", {
            "current": 0, "total": total, "msg": "Starting auto-tag",
        })
        for i, uid in enumerate(uids, start=1):
            try:
                record = _lib.load_image_metadata(uid)
                if record is None:
                    skipped += 1
                    continue
                if record.get("tags") and not overwrite:
                    skipped += 1
                    continue
                vec = _idx.semantic_vector(uid)
                if vec is None:
                    skipped += 1
                    continue
                tags = auto_tagger.tags_for_vector(vec)
                if not tags:
                    skipped += 1
                    continue
                record["tags"] = tags
                _lib.save_image_metadata(uid, record)
                tagged += 1
            except Exception:
                skipped += 1
            if i == total or i % 25 == 0:
                _push("tagging_progress", {
                    "current": i, "total": total, "msg": uid,
                })

        # Rebuild the global tag list once at the end (O(N), not O(N²))
        try:
            _lib._rebuild_global_tags()
        except Exception:
            pass

        _push("tagging_finished", {"tagged": tagged, "skipped": skipped})
        _push("library_changed", _lib_info())
        _push("status", f"Auto-tag complete: {tagged} tagged, {skipped} skipped.")
    except Exception as e:
        _push("tagging_error", str(e))
        _push("status", f"Auto-tag error: {e}")
    finally:
        _tagging_running = False


# ------------------------------------------------------------------ #
#  Internal helpers
# ------------------------------------------------------------------ #

_DATASET_SLOTS = ["semantic", "sketch", "sketch_fit", "sketch_crop"]
_STRUCTURE_SLOTS = ["sketch_fit", "sketch_crop"]


def _build_dataset_report() -> dict:
    uids = _lib.list_all_ids() if _lib and _lib.is_open else []
    total = len(uids)
    slot_sizes = _idx.slot_sizes() if _idx else {}
    slot_id_sets = {
        slot: set(_idx.slot_ids(slot)) if _idx else set()
        for slot in _DATASET_SLOTS
    }
    index_unique_counts = {
        slot: len(slot_id_sets[slot])
        for slot in _DATASET_SLOTS
    }
    index_duplicate_counts = {
        slot: max(0, int(slot_sizes.get(slot, 0)) - index_unique_counts[slot])
        for slot in _DATASET_SLOTS
    }

    metadata_counts = {slot: 0 for slot in _DATASET_SLOTS}
    index_missing_counts = {slot: 0 for slot in _DATASET_SLOTS}
    metadata_missing_counts = {slot: 0 for slot in _DATASET_SLOTS}
    structure_missing = []

    for uid in uids:
        meta = _lib.load_image_metadata(uid) if _lib else None
        indexed = (meta or {}).get("indexed_slots", {}) or {}
        if not isinstance(indexed, dict):
            indexed = {}
        missing_structure = False
        for slot in _DATASET_SLOTS:
            if slot in indexed:
                metadata_counts[slot] += 1
            else:
                metadata_missing_counts[slot] += 1
            if uid not in slot_id_sets[slot]:
                index_missing_counts[slot] += 1
            if slot in _STRUCTURE_SLOTS and (
                slot not in indexed or uid not in slot_id_sets[slot]
            ):
                missing_structure = True
        if missing_structure and len(structure_missing) < 12:
            structure_missing.append(uid)

    structure_complete = (
        total > 0
        and all(metadata_counts[s] == total for s in _STRUCTURE_SLOTS)
        and all(index_unique_counts[s] >= total for s in _STRUCTURE_SLOTS)
    )

    return {
        "is_open": bool(_lib and _lib.is_open),
        "total": total,
        "slot_sizes": slot_sizes,
        "index_unique_counts": index_unique_counts,
        "index_duplicate_counts": index_duplicate_counts,
        "metadata_counts": metadata_counts,
        "missing_counts": {
            slot: {
                "metadata": metadata_missing_counts[slot],
                "index": index_missing_counts[slot],
            }
            for slot in _DATASET_SLOTS
        },
        "structure": {
            "complete": structure_complete,
            "missing_count": max(
                metadata_missing_counts["sketch_fit"],
                metadata_missing_counts["sketch_crop"],
                index_missing_counts["sketch_fit"],
                index_missing_counts["sketch_crop"],
            ),
            "sample_missing": structure_missing,
            "fit": "long-side 128 + white pad to 128x128",
            "crop": "short-side 128 + center crop to 128x128",
            "dim": 16384,
        },
    }


def _dataset_report_summary(report: dict) -> str:
    total = report.get("total", 0)
    missing = (report.get("structure") or {}).get("missing_count", 0)
    if total <= 0:
        return "Dataset check: no images."
    if missing <= 0:
        return f"Dataset check: structure info complete for {total} images."
    return f"Dataset check: {missing}/{total} images missing structure info."


def _structure_slot_meta(today: str) -> dict:
    return {
        "sketch_fit": {
            "model": "canny-longside-pad-128x128",
            "version": "2.0",
            "date": today,
        },
        "sketch_crop": {
            "model": "canny-shortside-crop-128x128",
            "version": "2.0",
            "date": today,
        },
    }


def _run_structure_supplement() -> None:
    global _dataset_running
    added_vectors = 0
    updated_records = 0
    skipped = 0
    try:
        uids = _lib.list_all_ids()
        fit_ids = set(_idx.slot_ids("sketch_fit"))
        crop_ids = set(_idx.slot_ids("sketch_crop"))
        tasks = []
        for uid in uids:
            meta = _lib.load_image_metadata(uid)
            indexed = (meta or {}).get("indexed_slots", {}) or {}
            if not isinstance(indexed, dict):
                indexed = {}
            needs_fit = uid not in fit_ids
            needs_crop = uid not in crop_ids
            needs_meta = (
                "sketch_fit" not in indexed or "sketch_crop" not in indexed
            )
            if needs_fit or needs_crop or needs_meta:
                tasks.append((uid, needs_fit, needs_crop, needs_meta))

        total = len(tasks)
        _push("dataset_progress", {
            "current": 0,
            "total": total,
            "msg": "Starting structure supplement",
        })
        today = date.today().isoformat()
        meta_template = _structure_slot_meta(today)

        for i, (uid, needs_fit, needs_crop, needs_meta) in enumerate(tasks, start=1):
            try:
                if needs_fit or needs_crop:
                    path = _lib.get_original_image_path(uid)
                    if not path or not os.path.exists(path):
                        skipped += 1
                        continue
                    with Image.open(path) as img:
                        fit_vec, crop_vec = image_proc.structure_dual_vectors(
                            img.convert("RGB")
                        )
                    _idx.add_structure_vectors(
                        uid,
                        sketch_fit_vec=fit_vec if needs_fit else None,
                        sketch_crop_vec=crop_vec if needs_crop else None,
                    )
                    if needs_fit:
                        fit_ids.add(uid)
                        added_vectors += 1
                    if needs_crop:
                        crop_ids.add(uid)
                        added_vectors += 1

                record = _lib.load_image_metadata(uid)
                if record is not None and (needs_meta or uid in fit_ids or uid in crop_ids):
                    indexed = record.get("indexed_slots", {}) or {}
                    if not isinstance(indexed, dict):
                        indexed = {}
                    if uid in fit_ids:
                        indexed["sketch_fit"] = dict(meta_template["sketch_fit"])
                    if uid in crop_ids:
                        indexed["sketch_crop"] = dict(meta_template["sketch_crop"])
                    record["indexed_slots"] = indexed
                    _lib.save_image_metadata(uid, record)
                    updated_records += 1
            except Exception:
                skipped += 1
            if i == total or i % 25 == 0:
                _push("dataset_progress", {
                    "current": i,
                    "total": total,
                    "msg": uid,
                })

        if _lib.index_dir and added_vectors > 0:
            _idx.save(str(_lib.index_dir))

        report = _build_dataset_report()
        _push("dataset_finished", {
            "added_vectors": added_vectors,
            "updated_records": updated_records,
            "skipped": skipped,
            "report": report,
        })
        _push(
            "status",
            "Dataset supplement complete: "
            f"{added_vectors} vectors, {updated_records} records, {skipped} skipped.",
        )
    except Exception as e:
        _push("dataset_error", str(e))
        _push("status", f"Dataset supplement error: {e}")
    finally:
        _dataset_running = False


_DUP_STRICT_DISTANCE = 5
_DUP_POSSIBLE_DISTANCE = 10          # legacy default radius (≈84% similarity)
_DUP_GROUP_LIMIT = 80
_PHASH_BITS = 64                     # pHash length; similarity% = (1-dist/64)*100
_DUP_DEFAULT_MIN_SIMILARITY = 84


def _phash_similarity(distance: int) -> int:
    """Hamming distance → pixel-similarity percentage (0–100)."""
    return max(0, min(100, round((1.0 - distance / _PHASH_BITS) * 100)))


def _similarity_radius(min_similarity: int) -> int:
    """Similarity percentage (50–100) → BK-tree Hamming search radius."""
    min_similarity = max(50, min(100, int(min_similarity)))
    return max(0, min(_PHASH_BITS // 2,
                      round(_PHASH_BITS * (100 - min_similarity) / 100)))


class _BKNode:
    __slots__ = ("value", "uids", "children")

    def __init__(self, value: int, uid: str):
        self.value = value
        self.uids = [uid]
        self.children: Dict[int, "_BKNode"] = {}


class _BKTree:
    def __init__(self):
        self.root: Optional[_BKNode] = None

    def add(self, value: int, uid: str) -> None:
        if self.root is None:
            self.root = _BKNode(value, uid)
            return
        node = self.root
        while True:
            dist = (value ^ node.value).bit_count()
            if dist == 0:
                node.uids.append(uid)
                return
            child = node.children.get(dist)
            if child is None:
                node.children[dist] = _BKNode(value, uid)
                return
            node = child

    def query(self, value: int, radius: int) -> List[Tuple[str, int]]:
        if self.root is None:
            return []
        results: List[Tuple[str, int]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            dist = (value ^ node.value).bit_count()
            if dist <= radius:
                results.extend((uid, dist) for uid in node.uids)
            lo = dist - radius
            hi = dist + radius
            for edge_dist, child in node.children.items():
                if lo <= edge_dist <= hi:
                    stack.append(child)
        return results


def _run_duplicate_scan() -> None:
    global _duplicate_running
    updated = 0
    skipped = 0
    try:
        records = _lib.list_all_records()
        total = len(records)
        _push("duplicate_progress", {
            "current": 0,
            "total": total,
            "msg": "Starting duplicate scan",
        })
        for i, record in enumerate(records, start=1):
            uid = record.get("id", "")
            if not uid:
                skipped += 1
                continue
            if not _record_needs_duplicate_hash(record):
                if i == total or i % 50 == 0:
                    _push("duplicate_progress", {
                        "current": i,
                        "total": total,
                        "msg": uid,
                    })
                continue

            path = _lib.get_original_image_path(uid)
            if not path or not os.path.exists(path):
                skipped += 1
                continue
            try:
                record["file_hash"] = image_proc.compute_file_sha256(path)
                with Image.open(path) as img:
                    phash, phash_flip = image_proc.duplicate_hashes(
                        img.convert("RGB")
                    )
                record["phash"] = phash
                record["phash_flip"] = phash_flip
                _lib.save_image_metadata(uid, record)
                updated += 1
            except Exception:
                skipped += 1

            if i == total or i % 25 == 0:
                _push("duplicate_progress", {
                    "current": i,
                    "total": total,
                    "msg": uid,
                })

        report = _build_duplicate_report()
        _push("duplicate_finished", {
            "updated": updated,
            "skipped": skipped,
            "report": report,
        })
        _push(
            "status",
            f"Duplicate scan complete: {len(report.get('groups', []))} groups, "
            f"{updated} updated, {skipped} skipped.",
        )
    except Exception as e:
        _push("duplicate_error", str(e))
        _push("status", f"Duplicate scan error: {e}")
    finally:
        _duplicate_running = False


def _record_needs_duplicate_hash(record: dict) -> bool:
    return not (
        record.get("file_hash")
        and record.get("phash")
        and record.get("phash_flip")
    )


def _build_duplicate_report(min_similarity: int = _DUP_DEFAULT_MIN_SIMILARITY) -> dict:
    records = _lib.list_all_records() if _lib and _lib.is_open else []
    by_uid = {r.get("id", ""): r for r in records if r.get("id")}
    missing = sum(1 for r in records if _record_needs_duplicate_hash(r))
    radius = _similarity_radius(min_similarity)
    edges = _duplicate_edges(records, radius)
    groups = _duplicate_groups_from_edges(edges, by_uid)
    return {
        "is_open": bool(_lib and _lib.is_open),
        "total": len(records),
        "missing_hashes": missing,
        "group_count": len(groups),
        "groups": groups[:_DUP_GROUP_LIMIT],
        "limited": len(groups) > _DUP_GROUP_LIMIT,
        "strict_distance": _DUP_STRICT_DISTANCE,
        "possible_distance": radius,
        "min_similarity": max(50, min(100, int(min_similarity))),
    }


def _duplicate_edges(records: List[dict],
                     radius: int = _DUP_POSSIBLE_DISTANCE) -> List[dict]:
    edges: Dict[Tuple[str, str, str], dict] = {}

    exact: Dict[str, List[str]] = {}
    normal_tree = _BKTree()
    flipped_tree = _BKTree()
    normal_values: Dict[str, int] = {}
    flipped_values: Dict[str, int] = {}

    for record in records:
        uid = record.get("id", "")
        if not uid:
            continue
        file_hash = record.get("file_hash", "")
        if file_hash:
            exact.setdefault(file_hash, []).append(uid)
        phash_value = _phash_int(record.get("phash", ""))
        if phash_value is not None:
            normal_values[uid] = phash_value
            normal_tree.add(phash_value, uid)
        flip_value = _phash_int(record.get("phash_flip", ""))
        if flip_value is not None:
            flipped_values[uid] = flip_value
            flipped_tree.add(flip_value, uid)

    for uids in exact.values():
        if len(uids) < 2:
            continue
        for i, a in enumerate(uids):
            for b in uids[i + 1:]:
                _add_duplicate_edge(edges, a, b, "exact", 0)

    for uid, value in normal_values.items():
        for other_uid, dist in normal_tree.query(value, radius):
            if uid == other_uid:
                continue
            _add_duplicate_edge(edges, uid, other_uid, "similar", dist)
        for other_uid, dist in flipped_tree.query(value, radius):
            if uid == other_uid:
                continue
            _add_duplicate_edge(edges, uid, other_uid, "flipped", dist)

    return list(edges.values())


def _add_duplicate_edge(
    edges: Dict[Tuple[str, str, str], dict],
    uid_a: str,
    uid_b: str,
    kind: str,
    distance: int,
) -> None:
    a, b = sorted((uid_a, uid_b))
    key = (a, b, kind)
    existing = edges.get(key)
    if existing is None or distance < existing["distance"]:
        edges[key] = {
            "uid_a": a,
            "uid_b": b,
            "kind": kind,
            "distance": int(distance),
            "similarity": _phash_similarity(int(distance)),
            "confidence": (
                "high" if distance <= _DUP_STRICT_DISTANCE else "possible"
            ),
        }


def _duplicate_groups_from_edges(edges: List[dict], by_uid: Dict[str, dict]) -> List[dict]:
    parent: Dict[str, str] = {}

    def find(uid: str) -> str:
        parent.setdefault(uid, uid)
        if parent[uid] != uid:
            parent[uid] = find(parent[uid])
        return parent[uid]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for edge in edges:
        union(edge["uid_a"], edge["uid_b"])

    buckets: Dict[str, dict] = {}
    for edge in edges:
        root = find(edge["uid_a"])
        bucket = buckets.setdefault(root, {"uids": set(), "edges": []})
        bucket["uids"].update([edge["uid_a"], edge["uid_b"]])
        bucket["edges"].append(edge)

    groups = []
    for idx, bucket in enumerate(buckets.values(), start=1):
        uids = [uid for uid in bucket["uids"] if uid in by_uid]
        if len(uids) < 2:
            continue
        edge_list = bucket["edges"]
        kinds = sorted({e["kind"] for e in edge_list})
        if "flipped" in kinds and len(kinds) > 1:
            kind = "mixed"
        elif "exact" in kinds:
            kind = "exact"
        elif "flipped" in kinds:
            kind = "flipped"
        else:
            kind = "similar"
        best_distance = min(e["distance"] for e in edge_list)
        confidence = (
            "high"
            if any(e["distance"] <= _DUP_STRICT_DISTANCE for e in edge_list)
            else "possible"
        )
        items = [_make_item_from_record(by_uid[uid]) for uid in uids]
        items.sort(key=lambda item: item.get("file_size", 0), reverse=True)
        groups.append({
            "group_id": f"dup_{idx}",
            "kind": kind,
            "kinds": kinds,
            "confidence": confidence,
            "best_distance": best_distance,
            "similarity": _phash_similarity(best_distance),
            "items": items,
            "edges": sorted(edge_list, key=lambda e: e["distance"])[:12],
        })

    groups.sort(key=lambda g: (g["best_distance"], -len(g["items"]), g["group_id"]))
    return groups


def _phash_int(value: str) -> Optional[int]:
    try:
        return int(value, 16) if value else None
    except Exception:
        return None


def _duplicate_report_summary(report: dict) -> str:
    total = report.get("total", 0)
    missing = report.get("missing_hashes", 0)
    groups = report.get("group_count", len(report.get("groups", [])))
    if total <= 0:
        return "Duplicate check: no images."
    if missing:
        return (
            f"Duplicate check: {groups} groups found; "
            f"{missing}/{total} images need hash scan."
        )
    min_sim = report.get("min_similarity")
    suffix = f" (≥{min_sim}% similarity)" if min_sim else ""
    return f"Duplicate check: {groups} duplicate/flipped groups found{suffix}."


def _lib_info() -> dict:
    return {
        "is_open": _lib.is_open,
        "name":    _lib.library_name,
        "count":   _lib.image_count() if _lib.is_open else 0,
        "path":    str(_lib.library_path) if _lib.library_path else "",
        "tags":    _lib.list_tags() if _lib.is_open else [],
    }


def _uids_to_items(uids: List[str]) -> list:
    """Legacy helper used by search results (uids come from FAISS)."""
    items = []
    for uid in uids:
        meta = _lib.load_image_metadata(uid)
        if meta:
            items.append(_make_item(uid, meta, _lib.get_thumbnail_path(uid)))
    return items


def _make_item_from_record(record: dict, score: float = 0.0, per_modal: Optional[dict] = None) -> dict:
    """
    Build a frontend item dict directly from a SQLite record dict.
    Faster than _make_item() because it skips a second DB lookup.
    """
    uid = record.get("id", "")
    thumb_path = _lib.get_thumbnail_path(uid) if _lib.is_open else None
    thumb_url  = f"/thumb/{uid}" if (thumb_path and os.path.exists(thumb_path)) else ""
    orig_url   = f"/original/{uid}"   # lazy — browser hits the endpoint only if needed
    return {
        "uid":                  uid,
        "score":                score,
        "match_pct":            int(score * 100) if score > 0.01 else 0,
        "per_modal":            per_modal or {},
        "name":                 record.get("name", uid),
        "ext":                  record.get("ext", ""),
        "width":                record.get("width", 0),
        "height":               record.get("height", 0),
        "file_size":            record.get("file_size", 0),
        "dominant_color":       record.get("dominant_color", [128, 128, 128]),
        "color_palette":        record.get("color_palette", []),
        "color_palette_ratio":  record.get("color_palette_ratio", []),
        "tags":                 record.get("tags", []),
        "annotation":           record.get("annotation", ""),
        "file_hash":            record.get("file_hash", ""),
        "phash":                record.get("phash", ""),
        "phash_flip":           record.get("phash_flip", ""),
        "duplicate_group":      record.get("duplicate_group", ""),
        "duplicate_kind":       record.get("duplicate_kind", ""),
        "import_at":            (record.get("import_at", "") or "")[:10],
        "thumbnail_url":        thumb_url,
        "original_url":         orig_url,
    }


def _make_item(
    uid: str,
    meta: Optional[dict],
    thumb_path: Optional[str],
    score: float = 0.0,
    per_modal: Optional[dict] = None,
) -> dict:
    meta = meta or {}
    thumb_url = f"/thumb/{uid}"    if (thumb_path and os.path.exists(thumb_path)) else ""
    orig      = _lib.get_original_image_path(uid) if _lib.is_open else None
    orig_url  = f"/original/{uid}" if (orig and os.path.exists(orig))              else ""
    return {
        "uid":                  uid,
        "score":                score,
        "match_pct":            int(score * 100) if score > 0.01 else 0,
        "per_modal":            per_modal or {},
        "name":                 meta.get("name", uid),
        "ext":                  meta.get("ext", ""),
        "width":                meta.get("width", 0),
        "height":               meta.get("height", 0),
        "file_size":            meta.get("file_size", 0),
        "dominant_color":       meta.get("dominant_color", [128, 128, 128]),
        "color_palette":        meta.get("color_palette", []),
        "color_palette_ratio":  meta.get("color_palette_ratio", []),
        "tags":                 meta.get("tags", []),
        "annotation":           meta.get("annotation", ""),
        "file_hash":            meta.get("file_hash", ""),
        "phash":                meta.get("phash", ""),
        "phash_flip":           meta.get("phash_flip", ""),
        "duplicate_group":      meta.get("duplicate_group", ""),
        "duplicate_kind":       meta.get("duplicate_kind", ""),
        "import_at":            (meta.get("import_at", "") or "")[:10],
        "thumbnail_url":        thumb_url,
        "original_url":         orig_url,
    }


def _do_show_all_push() -> None:
    """Push first page + total to frontend; remaining pages loaded on scroll."""
    total   = _lib.total_count()
    records = _lib.list_page(offset=0, limit=200)
    items   = [_make_item_from_record(r) for r in records]
    _push("all_images", items)
    _push("library_total", {"total": total, "page_size": 200})
    _push("status", f"Library: {total} images.")


def _build_settings_dict() -> dict:
    s = _cfg
    return {
        "language":          s.language,
        "semantic_enabled":  s.semantic_enabled,
        "color_enabled":     s.color_enabled,
        "structure_enabled": s.structure_enabled,
        "semantic_weight":   s.semantic_weight,
        "color_weight":      s.color_weight,
        "structure_weight":  s.structure_weight,
        "top_k":             s.top_k,
        "effective_weights": s.effective_weights(),
    }


def _search_status_message(ctx: dict, count: int) -> str:
    return f"Found {count} images."


def _preload_clip() -> None:
    import time
    time.sleep(1.5)
    try:
        model = clip_module.get_clip_model()
        if model.multilingual:
            _push("status", "CLIP ready — multilingual search enabled (Chinese / 50+ languages).")
        else:
            _push("status", "CLIP ready — English search (install multilingual-clip for multilingual support).")
    except Exception as e:
        _push("status", f"CLIP load failed: {e}")


# ------------------------------------------------------------------ #
#  Native file dialogs (tkinter)
# ------------------------------------------------------------------ #

def _pick_folder(title: str = "Select Folder", initialdir: str = "") -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        kwargs: dict = {"title": title}
        if initialdir and os.path.isdir(initialdir):
            kwargs["initialdir"] = initialdir
        path = filedialog.askdirectory(**kwargs)
        root.destroy()
        return str(Path(path)) if path else ""
    except Exception:
        return ""


def _pick_files(title: str = "Select Files") -> List[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        paths = filedialog.askopenfilenames(
            title=title,
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.gif"),
                ("All Files", "*.*"),
            ],
        )
        root.destroy()
        return list(paths)
    except Exception:
        return []


_IMAGE_EXTS: set = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"}


def _scan_folder_images(folder: str) -> List[str]:
    """Recursively collect all image paths under *folder* (sorted)."""
    results: List[str] = []
    for root_dir, _dirs, files in os.walk(folder):
        for fname in files:
            if Path(fname).suffix.lower() in _IMAGE_EXTS:
                results.append(os.path.join(root_dir, fname))
    results.sort()
    return results
