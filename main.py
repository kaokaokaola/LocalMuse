"""
LocalMuse V2 — Entry point (HTTP server mode).

Starts a FastAPI/uvicorn server on localhost and opens the browser automatically.
No Qt window — the UI is served as a standard web page at http://localhost:PORT.

Usage:
    python main.py              # default port 17788
    python main.py --port 8080  # custom port
"""

import argparse
import os
import sys
import threading
import traceback
import webbrowser
from pathlib import Path

# --- Project root -------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))

# --- Encoding safety ----------------------------------------------------
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# --- Crash log ----------------------------------------------------------
_CRASH_LOG = _ROOT / "localmuse_crash.log"

DEFAULT_PORT = 17788


def main() -> None:
    parser = argparse.ArgumentParser(description="LocalMuse V2 — local image search")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open the browser automatically")
    parser.add_argument("--language", choices=["en", "zh"], default=None,
                        help="Override UI language: 'en' or 'zh' (default: use saved setting)")
    args = parser.parse_args()

    # --- Import server & dependencies -----------------------------------
    from src.config.settings import Settings
    from src.infra.library_mgr import LibraryManager
    from src.infra.index_store import IndexStore
    from src.server import app, initialize

    # --- Initialize shared state ----------------------------------------
    cfg = Settings()
    # Override language from command-line if provided
    if args.language is not None:
        cfg.language = args.language
    lib = LibraryManager()
    idx = IndexStore()

    # Try to restore the last-used library so it's ready when the page loads
    last = cfg.last_library_path
    if last and Path(last).exists():
        try:
            lib.open_library(last)
            # Load per-slot FAISS indices (v4 format) or migrate from v3
            if lib.has_index():
                idx.load(str(lib.index_dir))
            print(f"[LocalMuse] Restored library: {lib.library_name} "
                  f"({lib.image_count()} images, "
                  f"indexed={idx.semantic_size()})")
        except Exception as e:
            print(f"[LocalMuse] Could not restore last library: {e}")

    initialize(lib, idx, cfg)

    # --- Open browser after a short delay -------------------------------
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        def _open_browser() -> None:
            import time
            time.sleep(1.2)          # wait for uvicorn to be ready
            webbrowser.open(url)
        threading.Thread(target=_open_browser, daemon=True).start()

    print(f"\n{'=' * 55}")
    print(f"  LocalMuse V2")
    print(f"  Listening on:  {url}")
    print(f"  Press Ctrl+C to quit")
    print(f"{'=' * 55}\n")

    # --- Start uvicorn --------------------------------------------------
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",   # suppress routine request logs
    )


if __name__ == "__main__":
    # Clean up any stale crash log
    try:
        if _CRASH_LOG.exists():
            _CRASH_LOG.unlink()
    except OSError:
        pass

    try:
        main()
    except KeyboardInterrupt:
        print("\n[LocalMuse] Shutting down.")
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        print("\n" + "=" * 60, file=sys.stderr, flush=True)
        print("  LocalMuse CRASH — startup error", file=sys.stderr, flush=True)
        print("=" * 60, file=sys.stderr, flush=True)
        print(tb, file=sys.stderr, flush=True)
        try:
            _CRASH_LOG.write_text(tb, encoding="utf-8")
            print(f"Crash log: {_CRASH_LOG}", file=sys.stderr, flush=True)
        except OSError:
            pass
        sys.exit(1)
