"""
LocalMuse V2 retrieval test suite.

Run from the project root:
    python test_search.py

The test uses in-memory settings so it does not modify user preferences.
Library path is hardcoded below; change it if needed.
"""

from __future__ import annotations

import base64
import os
import sys
import time
from io import BytesIO
from typing import Any, Dict, Iterable, Optional

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

LIBRARY_PATH = r"E:\30,000test.library"
TOP_K = 10

from src.core import clip_model as clip_module
from src.infra.index_store import IndexStore
from src.infra.library_mgr import LibraryManager
from src.services.searcher import SearchWorker, make_query_context


class TestSettings:
    """Small SearchWorker-compatible settings object that never saves to disk."""

    def __init__(self) -> None:
        self.language = "zh"
        self.last_library_path = LIBRARY_PATH
        self.semantic_enabled = True
        self.color_enabled = True
        self.structure_enabled = True
        self.depth_enabled = False
        self.pose_enabled = False
        self.ocr_enabled = False
        self.semantic_weight = 1.0
        self.color_weight = 0.6
        self.structure_weight = 0.8
        self.depth_weight = 0.5
        self.pose_weight = 0.5
        self.top_k = TOP_K
        self.thumbnail_size = 360

    def effective_weights(self) -> Dict[str, float]:
        raw = {
            "semantic": self.semantic_weight if self.semantic_enabled else 0.0,
            "color": self.color_weight if self.color_enabled else 0.0,
            "structure": self.structure_weight if self.structure_enabled else 0.0,
            "depth": self.depth_weight if self.depth_enabled else 0.0,
            "pose": self.pose_weight if self.pose_enabled else 0.0,
        }
        total = sum(raw.values())
        if total <= 0:
            return {
                "semantic": 1.0,
                "color": 0.0,
                "structure": 0.0,
                "depth": 0.0,
                "pose": 0.0,
            }
        return {k: v / total for k, v in raw.items()}


def make_data_url(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def make_structure_sketch() -> str:
    image = Image.new("RGB", (320, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.line((45, 165, 140, 55, 275, 155), fill="black", width=12)
    draw.line((92, 135, 238, 135), fill="black", width=8)
    draw.line((140, 55, 140, 165), fill="black", width=7)
    return make_data_url(image)


def reset_weights(cfg: TestSettings) -> None:
    cfg.semantic_enabled = True
    cfg.color_enabled = True
    cfg.structure_enabled = True
    cfg.depth_enabled = False
    cfg.pose_enabled = False
    cfg.semantic_weight = 1.0
    cfg.color_weight = 0.6
    cfg.structure_weight = 0.8
    cfg.depth_weight = 0.5
    cfg.pose_weight = 0.5


def search_context(
    *,
    enabled: Iterable[str],
    text: str = "",
    color: Optional[tuple[int, int, int]] = None,
    sketch_data_url: str = "",
) -> dict:
    return make_query_context(
        text=text,
        color=color,
        sketch_data_url=sketch_data_url,
        enabled_modalities=list(enabled),
        top_k=TOP_K,
    )


print("=" * 64)
print("  LocalMuse V2 Retrieval Test Suite")
print("=" * 64)

cfg = TestSettings()

lib = LibraryManager()
print(f"\n[1/3] Opening library: {LIBRARY_PATH}")
t0 = time.perf_counter()
lib.open_library(LIBRARY_PATH)
print(f"      {lib.image_count():,} images ({time.perf_counter() - t0:.2f}s)")

idx = IndexStore()
print("[2/3] Loading FAISS indices")
t0 = time.perf_counter()
idx.load(str(lib.index_dir))
sizes = idx.slot_sizes()
print(f"      Slots: { {k: v for k, v in sizes.items() if v > 0} } ({time.perf_counter() - t0:.2f}s)")

print("[3/3] Loading CLIP model")
t0 = time.perf_counter()
clip_module.get_clip_model()
print(f"      CLIP ready ({time.perf_counter() - t0:.2f}s)")

searcher = SearchWorker(lib, idx, cfg)
SKETCH_DATA_URL = make_structure_sketch()


def run(
    label: str,
    ctx: dict,
    settings_patch: Optional[Dict[str, Any]] = None,
) -> None:
    old_values: Dict[str, Any] = {}
    if settings_patch:
        for key, value in settings_patch.items():
            old_values[key] = getattr(cfg, key)
            setattr(cfg, key, value)
    try:
        t0 = time.perf_counter()
        results = searcher.run_sync(ctx)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"\n{'=' * 64}")
        print(f"  TEST: {label}")
        print(f"  Time: {elapsed:.0f} ms   |   Results: {len(results)}")
        print("=" * 64)
        for index, result in enumerate(results[:TOP_K], 1):
            name = result.get("meta", {}).get("name", result["uid"])[:40]
            score = result.get("score", 0.0)
            per_modal = result.get("per_modal", {})
            modal_text = "  ".join(
                f"{key[:3]}={value:.2f}"
                for key, value in per_modal.items()
                if value > 0.01
            )
            print(f"  {index:>2}. [{score:.3f}] {name:<42}  {modal_text}")
    finally:
        for key, value in old_values.items():
            setattr(cfg, key, value)


tests = [
    (
        "Semantic only: concrete brutalist heavy mass",
        search_context(enabled=["semantic"], text="concrete brutalist heavy mass"),
        None,
    ),
    (
        "Semantic only: Tadao Ando light shadow concrete",
        search_context(enabled=["semantic"], text="Tadao Ando light shadow concrete"),
        None,
    ),
    (
        "Semantic + Color: timber warm interior + warm orange",
        search_context(
            enabled=["semantic", "color"],
            text="timber warm interior",
            color=(180, 120, 60),
        ),
        None,
    ),
    (
        "Semantic + Color: glass steel curtain wall + grey",
        search_context(
            enabled=["semantic", "color"],
            text="glass steel curtain wall",
            color=(160, 165, 170),
        ),
        None,
    ),
    (
        "Semantic + Sketch: minimalist void space + synthetic sketch",
        search_context(
            enabled=["semantic", "sketch"],
            text="minimalist void space",
            sketch_data_url=SKETCH_DATA_URL,
        ),
        {"structure_weight": 2.0},
    ),
    (
        "Sketch only: synthetic roof/courtyard sketch",
        search_context(enabled=["sketch"], sketch_data_url=SKETCH_DATA_URL),
        None,
    ),
    (
        "Color only: dark dramatic",
        search_context(enabled=["color"], color=(20, 20, 30)),
        None,
    ),
    (
        "Full multimodal: organic biophilic courtyard + green + sketch",
        search_context(
            enabled=["semantic", "color", "sketch"],
            text="organic biophilic courtyard",
            color=(80, 120, 70),
            sketch_data_url=SKETCH_DATA_URL,
        ),
        None,
    ),
]

for label, ctx, patch in tests:
    reset_weights(cfg)
    run(label, ctx, patch)

print(f"\n{'=' * 64}")
print("  All tests complete.")
print(f"{'=' * 64}\n")
