

https://github.com/user-attachments/assets/3b468fb9-9911-4ee7-abfb-8bec16612bbe

# LocalMuse V2.1

Multimodal local image search for designers — search your reference library by **meaning, sketch, structure/depth, and pose**, entirely on your own machine. No cloud, no upload, no subscription.

> Companion project for a paper presented at **CDRF 2026**. This repository hosts the V2 implementation. **v2.1** is the current release — see [What's New](#whats-new-in-v21) below.

https://github.com/user-attachments/assets/3b468fb9-9911-4ee7-abfb-8bec16612bbe
---

## What's New in v2.1

**v2.1 makes the duplicate‑detection workflow feel like a real product.** The rework is scoped to the Duplicates panel only — the main grid and lightbox delete behavior are unchanged.

- **Recycle bin (soft delete).** Removing a duplicate now moves it to a per‑library `.trash/` folder instead of deleting it outright. Your original source files on disk are left untouched unless you explicitly tick *"Also delete source"*.
- **One‑click Undo.** Every delete raises a 7‑second Undo toast. Undo restores the images *and* their exact search vectors instantly, so a mistaken cleanup costs nothing.
- **Cross‑group batch selection.** Tick individual images across any groups, use a group‑level *"select smaller"* toggle, or *"Select smaller (all)"* to mark every non‑largest version in one shot. The action bar shows how many are selected and how much space they will free.
- **Local refresh — no full re‑scan.** After a delete the panel re‑renders instantly from the cached model (groups that drop below two items disappear automatically). No more waiting for the whole library to be re‑scanned.
- **Progress & result feedback.** Deletes report how many images were removed, how much space was freed, and how many (if any) failed.
- **New batch API endpoints.** `POST /api/images/delete`, `POST /api/images/restore`, `GET /api/trash`, and `POST /api/trash/purge` power the recycle bin; FAISS slots are rebuilt once per batch and the exact vectors are captured so an undo is lossless.

Full details in [`CHANGELOG.md`](CHANGELOG.md).

---

## Features

- **Semantic search** — CLIP‑based text‑to‑image and image‑to‑image search.
- **Sketch search** — query by rough composition / structure (Canny‑based structure features).
- **Depth & structure** — match spatial depth and massing.
- **Pose search** — retrieve images by body pose (YOLOv8‑Pose).
- **AI auto‑annotation** — a local vision‑language model (Qwen3‑VL / Qwen2.5‑VL) tags and describes your library automatically.
- **Multilingual search** (optional) — M‑CLIP for Chinese / Japanese / Korean and 50+ languages, reusing the same indexed image vectors (no re‑indexing needed).
- **Duplicate & flipped‑image detection** with a recycle bin, batch selection, and undo (new in v2.1).
- **Runs as a local web app** — `python main.py` starts a FastAPI / uvicorn server on `http://localhost:17788` and opens it in your browser. No Qt, no cloud services.

---

## Requirements

- **Windows** (the `.bat` launchers target Windows; the Python code itself is cross‑platform).
- **Python 3.10 – 3.12** (add Python to PATH during install).
- **Git** — required to install OpenAI CLIP from source (semantic search). Optional but strongly recommended.
- Runs **CPU‑only** — no CUDA / GPU required.
- ~2–4 GB of free disk for model weights and dependencies.

Core dependencies (installed automatically by `setup.bat`): FastAPI, uvicorn, PyTorch (CPU), torchvision, Pillow, OpenCV (headless), `numpy<2.0`, faiss‑cpu, ftfy, regex, tqdm, timm. Optional: `ultralytics` (pose), `easyocr` (in‑image OCR), `transformers` + `multilingual-clip` (multilingual search).

---

## Installation

```bat
:: 1. One‑time setup — creates a self‑contained virtual environment in .\venv\
::    and installs every dependency (PyTorch CPU, FastAPI, faiss, CLIP, M‑CLIP…).
setup.bat
```

`setup.bat` is idempotent: if the environment is already complete it skips straight to verification, and re‑running it repairs a broken install. If Git is missing, semantic search is skipped with clear instructions to enable it later.

---

## Usage

```bat
:: Launch the app (uses .\venv\, then an embedded .\python\, then system Python)
LocalMuse.bat
```

Then open **http://127.0.0.1:17788** in your browser (it opens automatically). Press `Ctrl+C` in the console to quit. If the app crashes, details are written to `localmuse_crash.log`.

---

## Project structure

```
main.py                     Entry point — launches the FastAPI/uvicorn server
requirements.txt            Python dependencies (CLIP installed separately from Git)
setup.bat                   One‑time dependency installer → .\venv\
LocalMuse.bat               Launcher (venv → embedded python → system python)
src/
  core/                     CLIP model, image processing, vector math
  infra/                    Library catalog (SQLite) + FAISS index store + recycle bin
  services/                 Indexer, searcher, auto‑tagger, session tracker
  server.py                 FastAPI REST + WebSocket API
  ui/frontend/app.html      Single‑file web UI
```

---

## Privacy

Everything runs locally. Your images, the index, and all search happen on your own machine — nothing is uploaded, and no account or subscription is required.

## License

MIT — see [`LICENSE`](LICENSE).
