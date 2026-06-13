# LocalMuse V2

Multimodal local image search for designers — search your reference library by
**meaning, sketch, structure/depth, and pose**, entirely on your own machine.
No cloud, no upload, no subscription.

> Companion project for a paper presented at **CDRF 2026**. This repository
> hosts the V2 implementation.

## Features

- **Semantic search** — CLIP-based text-to-image and image-to-image search.
- **Sketch search** — query by rough composition / structure (Canny-based
  structure features).
- **Depth & structure** — match spatial depth and massing.
- **Pose search** — retrieve images by body pose (YOLOv8-Pose).
- **AI auto-annotation** — local vision-language model (Qwen3-VL /
  Qwen2.5-VL) tags and describes your library automatically.
- **Multilingual search** (optional) — M-CLIP for Chinese/Japanese/Korean and
  50+ languages.
- Runs as a local web app: `python main.py` starts a FastAPI/uvicorn server
  on `http://localhost:17788` and opens it in your browser. No Qt, no
  cloud services.

## Requirements

- Windows 10/11, x64
- Python 3.10–3.12 (installer will look for these via the `py` launcher or
  `PATH`)
- Git (only needed to install OpenAI CLIP from source — `setup.bat` will
  warn and continue without it if missing)
- NVIDIA GPU + driver supporting CUDA 12.4 — **optional**, but strongly
  recommended for AI annotation and faster indexing. CPU-only works for
  search.

## Quick start

```bat
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
setup.bat        REM one-time dependency install into .\venv\
LocalMuse.bat    REM starts the server and opens the browser
```

`setup.bat` is idempotent — safe to re-run any time to repair a broken
environment. See [`requirements.txt`](requirements.txt) and
[`requirements-optional.txt`](requirements-optional.txt) for what gets
installed and why.

## Project layout

```
main.py                  Entry point (FastAPI/uvicorn server)
LocalMuse.bat             Launcher (prefers .\venv\, falls back to .\python\)
setup.bat                  One-time / repair dependency installer -> .\venv\
requirements.txt           Required dependencies (single source of truth)
requirements-optional.txt  Optional features (pose, OCR, multilingual, VLM annotation)
src/
  core/                    Model wrappers (CLIP, depth, pose, annotation, etc.)
  infra/                   Index storage, library management
  services/                Indexing, search, intent engine
  config/                  Settings
  server.py                FastAPI app
  ui/frontend/             Web UI (single-page app)
test_search.py            Search smoke test
```

## Optional / advanced scripts

These are **not required** for a normal install — `setup.bat` +
`LocalMuse.bat` is enough for most users.

| Script | Purpose | When to use |
|---|---|---|
| `setup_annotation.bat` | Force-reinstall CUDA torch / re-check VLM annotation deps | Only for **older venvs** created before `setup.bat` had GPU auto-detection, or venvs missing `requirements-optional.txt` packages. On a fresh install this is a no-op. |
| `install_flash_attn.bat` | Installs a prebuilt FlashAttention 2 wheel | Optional speed-up for VLM annotation. Pinned to `torch==2.6.0+cu124` — if you change the PyTorch/CUDA version in `setup.bat`, this script's wheel URL must be updated too. |
| `build_portable_python.bat` | Builds a self-contained `.\python\` runtime (~6 GB) from `.\venv\` | **Maintainers only** — used to produce a no-install distributable for users without Python/internet. See [`PORTABLE_DEPLOYMENT.md`](PORTABLE_DEPLOYMENT.md). |
| `cleanup_venv.bat` | Removes `.\venv\` after `.\python\` is verified to work | Maintainers only, used after `build_portable_python.bat`. |

## Data & privacy

All indexing and search run locally. Model weights (CLIP, Qwen3-VL,
EasyOCR, depth/pose models — several GB total) are downloaded automatically
from Hugging Face on first use and cached under your user profile
(`~/.cache/huggingface`, etc.) — they are **not** part of this repository.
Your image library stays wherever you point LocalMuse to; nothing is
uploaded.

## Development notes

See [`LOCALMUSE_MODIFICATION_GUIDE.md`](LOCALMUSE_MODIFICATION_GUIDE.md) for
search-slot definitions, the annotation model setup, and dependency
management rules — read this before modifying search or annotation code.

## License

[MIT](LICENSE)
