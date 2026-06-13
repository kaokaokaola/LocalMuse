# LocalMuse Modification Guide

Read this file before every future code write or edit in this folder.

## Current Search Rules

- The top `Search` button is the single user-facing trigger for all modalities.
- Slot chips decide which data is sent. Do not keep separate per-slot search buttons as the primary workflow.
- If `Semantic` is active, text can trigger intent candidates. If only non-semantic slots are active, search must run directly.
- Intent candidates must not be static architecture presets. They should be derived from frequent words in the currently opened library.
- OCR is removed from search and should not be reintroduced unless explicitly requested.

## Slot Index Facts

- `semantic`: CLIP image/text, 512-D.
- `sketch`: legacy Canny 64x64, 4096-D, retained only as fallback.
- `sketch_fit`: scheme-B structure feature, Canny long side to 128 then white padding, 16384-D.
- `sketch_crop`: scheme-B structure feature, Canny short side to 128 then center crop, 16384-D.
- `depth`: real depth vectors are valid only when metadata `indexed_slots.depth` exists. Old libraries may contain zero FAISS rows that must not be treated as valid. If no valid depth vectors exist, pure Depth search falls back to a depth-contour proxy over `sketch_fit/sketch_crop`, and the score is still reported under the `depth` modality.
- `pose`: currently only valid when metadata `indexed_slots.pose` exists. Old libraries may contain zero FAISS rows that must not be treated as valid.

## Annotation Model Facts

- Image annotation should use `src/core/annotation_model.py`, not CLIP.
- Canonical thumbnails are 1024px on the long edge. Annotation must first ensure the thumbnail is regenerated to that size, then annotate the 1024px thumbnail.
- New import/index preprocessing should use the 1024px thumbnail as the working image for semantic, color, sketch/structure, depth, and pose features. Keep original files for EXIF, file hash, and duplicate pHash only.
- The preferred annotation model is `Qwen/Qwen3-VL-7B-Instruct`, with Qwen2.5-VL fallbacks.
- Current Transformers builds expose `Qwen3VLForConditionalGeneration` and `Qwen2_5_VLForConditionalGeneration`; do not import the stale `Qwen2_5VLForConditionalGeneration` name directly.
- The annotation venv needs CUDA torch plus `qwen-vl-utils` and `accelerate`. `setup_annotation.bat` should skip reinstall when CUDA is already available, and force reinstall only when replacing CPU torch.
- On Windows, `flash_attn` may import successfully while still failing inside Qwen VL generation. Keep annotation on `attn_implementation="sdpa"` by default. Only enable `flash_attention_2` when explicitly testing with `LOCALMUSE_ENABLE_FLASH_ATTN=1`.

## Dependency Management

- `requirements.txt` (required) and `requirements-optional.txt` (optional /
  graceful-degradation) are the single source of truth for pip package
  versions. When bumping or adding a dependency, edit ONLY these two files.
  Do not hardcode package versions in `setup.bat` or `setup_annotation.bat`.
- The only exceptions, handled directly in `setup.bat`:
  - `torch` / `torchvision` (Step 3/6) - installed first from a GPU-specific
    index: CUDA 12.4 (`--index-url .../whl/cu124`) if `nvidia-smi` succeeds,
    otherwise CPU-only (`--index-url .../whl/cpu`). A single requirements.txt
    line can't express "pick the index based on hardware", so this stays in
    the script.
  - OpenAI CLIP (Step 6/6) - installed via
    `git+https://github.com/openai/CLIP.git` because it isn't on PyPI, and
    only if Git is detected.
- `numpy` must stay `<2.0` (OpenAI CLIP needs the numpy 1.x API) and
  `transformers` must stay `>=4.51.0` (required for
  `Qwen3VLForConditionalGeneration`). Both constraints live in the
  requirements files - keep them there if the files are edited again.
- `setup_annotation.bat` is now a repair/verify tool, not a required step for
  fresh installs: `setup.bat` already installs CUDA torch when a GPU is
  detected and installs `qwen-vl-utils`/`accelerate` via
  `requirements-optional.txt`. Run `setup_annotation.bat` only to force CUDA
  torch onto a venv that ended up CPU-only, or to repair/verify an older venv.
- `install_flash_attn.bat` is pinned to `torch==2.6.0+cu124` (its version
  check and `WHEEL_URL` both encode this). If `setup.bat`'s CUDA index or the
  torch version ever changes, update both of those together in
  `install_flash_attn.bat` or the flash-attn install will silently target the
  wrong wheel.
- `python\` is a portable Python runtime (built by `build_portable_python.bat`
  from `venv\`) that `LocalMuse.bat` prefers over `venv\` when present. After
  verifying `python\` works standalone, `cleanup_venv.bat` can reclaim the
  ~6 GB used by `venv\` (two-stage rename-then-delete, run by the user).

## Existing Library Migration

- Do not silently rebuild large existing libraries during startup.
- Use the Library `Check` button for read-only diagnostics.
- Use the Library `Fill Info` button to supplement missing `sketch_fit` and `sketch_crop` metadata/index rows.
- Avoid duplicate FAISS rows when supplementing: only add vectors for UIDs missing from that specific slot.

## Frontend Payload Checks

- `buildSearchPayload()` is the source of truth for `/api/search`.
- A selected slot must send its required input:
  - `semantic`: text or selected intent query.
  - `color`: `color`.
  - `sketch`: `sketch_data_url`.
  - `depth`: `depth_image_data_url` plus `depth_is_map`.
  - `pose`: `pose_image_data_url`.
- If a selected non-semantic slot has no input, show a clear status error before calling `/api/search`.

## Verification Before Finishing

- Compile edited Python files with `compile(...)` or the venv interpreter.
- Parse the frontend `<script>` with Node when `app.html` changes.
- Import `src.server` with `PYTHONDONTWRITEBYTECODE=1` and the project venv when server code changes.
- Do not run full dataset supplementation unless the user explicitly asks or clicks the UI button.
