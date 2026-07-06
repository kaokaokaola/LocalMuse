"""
LocalMuse V2 — Retrieval Test Suite
Run from project root:  python test_search.py

Tests 8 query combinations and prints ranked results + timing.
Library path is hardcoded below — change if needed.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

LIBRARY_PATH = r"E:\30,000test.library"
TOP_K = 10  # results to show per test

# ── bootstrap ────────────────────────────────────────────────────────────────
from src.config.settings import Settings
from src.infra.library_mgr import LibraryManager
from src.infra.index_store import IndexStore
from src.services.searcher import SearchWorker, make_query_context
from src.core import clip_model as clip_module

print("=" * 64)
print("  LocalMuse V2 — Retrieval Test Suite")
print("=" * 64)

cfg = Settings()
cfg.top_k = TOP_K

lib = LibraryManager()
print(f"\n[1/3] Opening library: {LIBRARY_PATH}")
t0 = time.perf_counter()
lib.open_library(LIBRARY_PATH)
print(f"      {lib.image_count():,} images  ({time.perf_counter()-t0:.2f}s)")

idx = IndexStore()
print("[2/3] Loading FAISS indices …")
t0 = time.perf_counter()
idx.load(str(lib.index_dir))
sizes = idx.slot_sizes()
print(f"      Slots: { {k:v for k,v in sizes.items() if v>0} }  ({time.perf_counter()-t0:.2f}s)")

print("[3/3] Loading CLIP model …")
t0 = time.perf_counter()
clip_module.load_model()
print(f"      CLIP ready ({time.perf_counter()-t0:.2f}s)")

searcher = SearchWorker(lib, idx, cfg)

# ── helper ────────────────────────────────────────────────────────────────────
def run(label, ctx, settings_patch=None):
    if settings_patch:
        for k, v in settings_patch.items():
            setattr(cfg, k, v)
    t0 = time.perf_counter()
    results = searcher.run_sync(ctx)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\n{'─'*64}")
    print(f"  TEST: {label}")
    print(f"  Time: {elapsed:.0f} ms   |   Results: {len(results)}")
    print(f"{'─'*64}")
    for i, r in enumerate(results[:TOP_K], 1):
        name  = r.get("meta", {}).get("name", r["uid"])[:40]
        score = r.get("score", 0)
        pm    = r.get("per_modal", {})
        pm_str = "  ".join(
            f"{k[:3]}={v:.2f}" for k, v in pm.items() if v > 0.01
        )
        print(f"  {i:>2}. [{score:.3f}] {name:<42}  {pm_str}")
    # reset any patched settings
    if settings_patch:
        for k in settings_patch:
            setattr(cfg, k, Settings._DEFAULTS.get(k, getattr(Settings(), k, None)) 
                    if hasattr(Settings, '_DEFAULTS') else None)

# ── reset weights helper ──────────────────────────────────────────────────────
def reset_weights():
    cfg.semantic_enabled  = True;  cfg.semantic_weight   = 1.0
    cfg.color_enabled     = True;  cfg.color_weight      = 0.6
    cfg.structure_enabled = True;  cfg.structure_weight  = 0.8
    cfg.depth_enabled     = False
    cfg.pose_enabled      = False

# ════════════════════════════════════════════════════════════════════════════════
#  TEST CASES
# ════════════════════════════════════════════════════════════════════════════════

# T1: Text only — broad architectural query
reset_weights()
cfg.color_enabled = False; cfg.structure_enabled = False
run("Text only — 'concrete brutalist heavy mass'",
    make_query_context(text="concrete brutalist heavy mass"))

# T2: Text only — Japanese minimalism
reset_weights()
cfg.color_enabled = False; cfg.structure_enabled = False
run("Text only — 'Tadao Ando light shadow concrete'",
    make_query_context(text="Tadao Ando light shadow concrete"))

# T3: Text + Color (warm tones)
reset_weights()
cfg.structure_enabled = False
run("Text + Color — 'timber warm interior' + warm orange",
    make_query_context(text="timber warm interior", color=(180, 120, 60)))

# T4: Text + Color (cool/grey)
reset_weights()
cfg.structure_enabled = False
run("Text + Color — 'glass steel curtain wall' + grey",
    make_query_context(text="glass steel curtain wall", color=(160, 165, 170)))

# T5: Text + Structure (high structure weight)
reset_weights()
cfg.color_enabled     = False
cfg.structure_weight  = 2.0
run("Text + Structure (boosted) — 'minimalist void space'",
    make_query_context(text="minimalist void space"))

# T6: Color only — dark dramatic
reset_weights()
cfg.semantic_enabled  = False
cfg.structure_enabled = False
run("Color only — dark dramatic (20, 20, 30)",
    make_query_context(color=(20, 20, 30)))

# T7: Color only — soft white
reset_weights()
cfg.semantic_enabled  = False
cfg.structure_enabled = False
run("Color only — soft white (240, 238, 232)",
    make_query_context(color=(240, 238, 232)))

# T8: Full multimodal — Text + Color + Structure
reset_weights()
run("Full multimodal — 'organic biophilic courtyard' + green + structure",
    make_query_context(text="organic biophilic courtyard", color=(80, 120, 70)))

print(f"\n{'='*64}")
print("  All tests complete.")
print(f"{'='*64}\n")
