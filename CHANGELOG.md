# Changelog

All notable changes to LocalMuse are documented here. This project loosely follows [Semantic Versioning](https://semver.org/).

## [v2.1.0] — 2026-07-06

### Highlights
The duplicate‑detection workflow was reworked into a proper product experience. All changes are scoped to the **Duplicates panel** — the main grid and lightbox delete behavior are intentionally unchanged.

### Added
- **Recycle bin (soft delete).** Deleting a duplicate moves its `.info` folder to a per‑library `.trash/` folder and records a manifest entry, instead of hard‑deleting it. Original source files on disk are preserved unless *"Also delete source"* is ticked.
- **Undo.** A 7‑second Undo toast appears after every delete. Undo restores both the catalog entries and their exact FAISS vectors, captured in memory before deletion (with an on‑disk recompute fallback for restores after a restart).
- **Cross‑group batch selection.** Per‑image checkboxes across groups, a group‑level *"select smaller"* toggle, and a global *"Select smaller (all)"* action. An action bar shows the selected count and the disk space that will be freed.
- **Local refresh.** After a delete the panel re‑renders instantly from the cached report model; groups that fall below two items are dropped automatically — no full library re‑scan.
- **Batch API endpoints:**
  - `POST /api/images/delete` — batch soft delete (one FAISS rebuild per slot for the whole batch), returns `{deleted, failed, freed_bytes, trash_count}`.
  - `POST /api/images/restore` — undo a soft delete.
  - `GET /api/trash` — list the recycle bin.
  - `POST /api/trash/purge` — permanently empty the recycle bin (honors the *"also delete source"* intent).
- WebSocket events (`image_removed`, `image_restored`, `images_deleted`, `images_restored`, `trash_updated`) keep every open client in sync.

### Changed
- The Compare & Select overlay and the "keep largest" shortcuts now route through the same soft‑delete + local‑refresh + undo engine, replacing blocking confirmation dialogs and per‑item delete calls.
- Infrastructure: `IndexStore` gained `remove_uids` (batched removal), `capture_vectors`, and `restore_vectors`; `LibraryManager` gained the trash lifecycle (`soft_delete_image`, `restore_trashed_image`, `purge_trashed_image`, `list_trashed`, `trash_count`).

### Notes
- Existing hard‑delete paths (`DELETE /api/image/{uid}`) remain unchanged for the lightbox and main grid.
- No new external dependencies were introduced.

## [v2.0] — earlier
- Initial public V2 implementation (companion to the CDRF 2026 paper): CLIP semantic search, sketch/structure/depth search, YOLOv8 pose search, AI auto‑annotation, optional M‑CLIP multilingual search, FastAPI/uvicorn local web app.
