# LocalMuse Modification Guide

Read this file before every future code write or edit in this folder.

## Current Search Rules (v3 simplified)

- Only three modalities exist: `semantic` (text), `structure` (sketch), `color`.
- The top `Search` button is the single user-facing trigger for all modalities.
- Slot chips decide which data is sent. Do not keep separate per-slot search buttons as the primary workflow.
- Search runs directly — there is NO intent-candidate gate. The intent engine, pairwise prompts, diversity nudge, adaptive session weights, and weight visualization panel were all removed in v3 and must not be reintroduced unless explicitly requested.
- Depth, Pose, OCR, and EXIF are removed from indexing and search and should not be reintroduced unless explicitly requested.

## Live Search & Sketch Match (v3.1)

- A `Live` checkbox next to `Search` enables auto re-search on every input change (sketch strokes, text typing, color changes, slot toggles) via `scheduleLiveSearch()` with a 0.5 s debounce. If the server returns `{busy:true}`, one retry is scheduled. Any new input path must call `scheduleLiveSearch()`.
- `POST /api/image/{uid}/sketch-match` (sync `def`, threadpooled) returns the image's Canny edge map as a data URL plus the best-matching local region: `{edge_url, width, height, bbox, score, full_score, sketch_crop}`. Coordinates are in edge-map pixels.
- The matcher is `image_proc.sketch_local_match(edge_img, sketch_img)`: orientation-aware chamfer matching. Key invariants (each fixes a specific observed failure — do not remove):
  - Orientation bins come from the STRUCTURE TENSOR, not raw Sobel angles — raw gradients cancel to zero on stroke centerlines and dump everything into bin 0, destroying orientation selectivity.
  - Per-channel distance transforms give smooth falloff with misalignment (pixel-overlap/F1 scoring was tried and let clutter windows win).
  - Forward score (sketch→image) is balanced across orientation bins (sqrt-count weights) so every stroke direction must find support; reverse score (image→sketch, 0.7/0.3 mix) penalises clutter and is only computed for the top-40 forward candidates.
  - Windows are searched at 5 scales × 3 aspect ratios (0.72/1.0/1.38 × sketch aspect) + refinement pass, so a slightly wider/narrower shape still matches.
  - Reference discrimination on synthetic bench: matching tower ≈0.89, wider-tower ≈0.84, no-shape ≈0.43, wrong-orientation ≈0.28; ~200 ms per image.
- The Detail pane renders this via `updateSketchMatch()` → `drawSketchMatch()` on `#d-match-canvas` (edge map + red-tinted sketch strokes drawn into `bbox`, red rectangle around the region). Only triggered when the sketch modality is active and a sketch exists. Uses `_matchToken` to discard stale responses.

## Auto Tagging (v3.1)

- Tag vocabulary is HUMAN-CURATED and limited to ~38 architecture terms in `src/services/auto_tagger.py` (`TAG_PROMPTS`: 类型/场景/材料/风格). Never switch to an open vocabulary; extend only by adding curated entries.
- Tagging is CLIP zero-shot cosine against the stored 512-D image vector — never re-encode images. Bulk tagging reconstructs vectors via `IndexStore.semantic_vector(uid)` (FAISS flat reconstruct).
- New imports are tagged automatically in `indexer.py`. `POST /api/library/auto-tag` (+ `Auto Tag` button) batch-tags images with empty tags (`?overwrite=true` re-tags all); worker uses `_tagging_running` flag, pushes `tagging_progress`/`tagging_finished`/`tagging_error`, calls `save_image_metadata` per record and `_rebuild_global_tags()` ONCE at the end (never `update_image_tags` in a loop — it is O(N) per call).
- Limits: max 4 tags per image, min cosine 0.21, always keep the best-1 tag.

## Duplicate Detection (v3.1)

- `GET /api/library/duplicates/check?min_similarity=84` accepts a user threshold in 50–100 (%). Similarity is pHash-based: `similarity% = (1 - hamming/64) * 100`; the query radius is `round(64 * (100 - min_sim) / 100)`, capped at 32 (= 50%). Thresholds below ~70% get noisy for pHash — the UI slider floor is 50 by design, do not lower it.
- Both the normal and horizontally-flipped BK-trees are queried with the same radius. Edges and groups carry a `similarity` percent; the report carries `min_similarity` and `possible_distance` (the radius actually used). Strict distance ≤5 still marks "high" confidence.
- Frontend: the Duplicates dialog header has a `#dup-sim` range slider (50–100, default 84) whose `onchange` calls `refreshDuplicates()` (re-queries without reopening the overlay). Group items are sorted by `file_size` descending server-side; the first item gets a green "Largest" keep-badge, and every item has its own "Delete This Version" button — this per-item button is the version-selection mechanism, do not replace it with a bulk "delete all duplicates" action.
- **Keep Largest shortcuts**: each group title has a "Keep Largest" button (`keepLargestInGroup(gi)`) and the dialog header has "Keep Largest (All)" (`keepLargestAll()`). Both rely on the server-side largest-first sort — they delete `items.slice(1)` via the shared `_bulkDeleteDuplicates(uids)` helper (library-only delete; source files always stay on disk for these shortcuts, by design). The global variant dedupes uids across groups and warns about "possible"-confidence groups in its confirm.
- **Rendering rules (bug-hardened)**: `renderDuplicateReport` builds the panel with DOM APIs (`createElement`/`textContent`), NOT innerHTML templates — file names may contain quotes/`<` and must never be interpolated into HTML strings. Thumbnails use the fallback chain thumbnail → original → `.dup-noimg` placeholder (`makeDupThumb`); groups with empty/missing `items` are skipped. Duplicate cells carry `data-uid`, so `removeItemLocally` clears them live during bulk deletes. The GET `/duplicates/check` endpoint must NOT `_push("duplicate_check", ...)` — the response already carries the report, and a WS broadcast can arrive mid-bulk-delete and corrupt the open panel.
- The `#dup-sim` slider is fully custom-styled (webkit + moz track/thumb rules); its filled-track effect uses the `--sim-pct` CSS var updated by `onDupSimInput(el)` — keep that oninput hook if you change the markup.
- **Compare & Select** (`#dup-compare-overlay`, z-index 530, sits above the Duplicates dialog at 520): each group title has a "Compare & Select" button → `openDupCompare(groupIndex)` renders all versions side-by-side at full resolution (`original_url`, `onerror` falls back to `thumbnail_url`). Clicking a card toggles it in `_dupCompareSel` (a Set of uids); `toggleDupCompareSel` refuses to select all items — at least one version must remain unselected. The checkbox label has `pointer-events:none` on purpose (it is a visual mirror only; clicks are handled at the card level to avoid the label/input double-fire). "Delete Selected" shows ONE confirm listing all names, honors the "Also delete source files" checkbox, calls `DELETE /api/image/{uid}?delete_source=` per uid directly (bypassing `deleteImageByUid`'s per-item confirm), then `removeItemLocally` each, closes the compare overlay, and `refreshDuplicates()`.

## Edge-Map Display Resolution (v3.1)

- `extract_edges(image, long_side=512)`: the `long_side` parameter exists ONLY for display. Anything feeding CLIP encoding or `sketch_local_match` MUST stay at the default 512 — changing it silently invalidates every stored structure vector and the chamfer benchmark numbers.
- `/api/image/{uid}/sketch-match` runs matching at 512 but returns a display edge map at `max(512, min(1024, original_long_side))` (never upscales past the original). When the display map differs from the match map, the endpoint rescales `bbox` into display coordinates server-side; `sketch_crop` remains in sketch space. The frontend draws in `res.width/height` space and needs no scaling logic.
- `#d-match-canvas` is click-to-zoom: `enlargeMatchCanvas()` pushes `canvas.toDataURL()` into the existing `#lightbox`.

## UI Theme (v3.1)

- Light RMIT theme: white base, RMIT red accent `#e60028` (`--accent`), darker hover `#b3001f` (`--accent2`). All colors flow through the `:root` CSS variables in `app.html`; do not hardcode dark-theme colors.

## Feedback Mechanism (v3)

- All feedback is non-intrusive: moodboard save/remove, per-session exclude, expand (detail view).
- Every explicit action (`search` / `save` / `remove` / `expand` / `exclude`) is appended to `{library}/feedback_log.jsonl` by `SessionTracker`. The log only records — it never changes ranking.
- Excluded UIDs are filtered out of search results server-side for the current session only.
- `/api/behavior` accepts only `{uid, signal}` with signal `expand` or `exclude`. Do not add hover/skip/pairwise signals.

## Slot Index Facts

- `semantic`: CLIP image/text, 512-D, inner product.
- `sketch`: legacy Canny 64x64, 4096-D L2, retained only as fallback.
- `sketch_fit`: scheme-B structure feature, Canny long side to 128 then white padding, 16384-D L2.
- `sketch_crop`: scheme-B structure feature, Canny short side to 128 then center crop, 16384-D L2.
- Old libraries may contain `index_depth.*` / `index_pose.*` files; `IndexStore` ignores them (does not load or delete). Old records may contain `ocr_text`/`exif` columns; `library_mgr.py` keeps those columns for backward compatibility — do not drop them via migration.
- `src/core/vector_math.py` keeps optional `depth_scores`/`pose_scores` kwargs on `fuse_scores` as a generic utility; callers must not pass them.

## Existing Library Migration

- Do not silently rebuild large existing libraries during startup.
- Use the Library `Check` button for read-only diagnostics.
- Use the Library `Fill Info` button to supplement missing `sketch_fit` and `sketch_crop` metadata/index rows.
- Avoid duplicate FAISS rows when supplementing: only add vectors for UIDs missing from that specific slot.

## Frontend Payload Checks

- `buildSearchPayload(rawText, mods, validate)` is the source of truth for `/api/search`.
- A selected slot must send its required input:
  - `semantic`: `text`.
  - `color`: `color` (RGB array).
  - `sketch`: `sketch_data_url`.
- If a selected non-semantic slot has no input, show a clear status error before calling `/api/search`.
- The right panel has only `moodboard` and `detail` tabs (timeline removed).

## Verification Before Finishing

- Compile edited Python files with `python -m py_compile` (the bundled venv is Windows-only; use the system interpreter when working in a Linux sandbox).
- Parse the frontend `<script>` block with `node --check` whenever `app.html` changes.
- Import `src.server` with `PYTHONDONTWRITEBYTECODE=1` and the project venv when server code changes (only possible on Windows where the venv works).
- After removals, grep `src/` for `depth|pose|ocr|pairwise|intent|hover|diversity` to confirm no stale references (backward-compat columns in `library_mgr.py`, generic kwargs in `vector_math.py`, and explanatory docstrings are the only acceptable hits).
- Do not run full dataset supplementation unless the user explicitly asks or clicks the UI button.
