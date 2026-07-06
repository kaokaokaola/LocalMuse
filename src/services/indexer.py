"""
Indexer service: scans image files, runs the active modality extractors
(semantic CLIP + sketch/structure + color), and persists results into the
library.  Runs in a plain Python thread (no Qt dependency).

v3 简化：移除 depth / pose / OCR / EXIF 提取 —— 只保留对
“快速找到合适图像”有直接贡献的模态。
"""

from __future__ import annotations
from datetime import date
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from PIL import Image

from src.core import clip_model as clip_module
from src.core import image_proc
from src.core.vector_math import normalize
from src.services import auto_tagger
from src.config.settings import Settings
from src.infra.library_mgr import LibraryManager, make_image_record, SUPPORTED_EXTENSIONS
from src.infra.index_store import IndexStore


class IndexWorker:
    """
    Pure-Python indexer — runs via threading.Thread, communicates via callbacks.

    Callbacks (all optional):
        on_progress(current: int, total: int, msg: str)
        on_item_indexed(uid: str)
        on_finished(added: int, skipped: int)
        on_error(message: str)
    """

    def __init__(
        self,
        lib: LibraryManager,
        idx: IndexStore,
        settings: Settings,
        on_progress:     Optional[Callable] = None,
        on_item_indexed: Optional[Callable] = None,
        on_finished:     Optional[Callable] = None,
        on_error:        Optional[Callable] = None,
    ):
        self._lib      = lib
        self._idx      = idx
        self._settings = settings
        self._on_progress     = on_progress     or (lambda *a: None)
        self._on_item_indexed = on_item_indexed or (lambda *a: None)
        self._on_finished     = on_finished     or (lambda *a: None)
        self._on_error        = on_error        or (lambda *a: None)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    # ---------------------------------------------------------------- #
    #  Main pipeline
    # ---------------------------------------------------------------- #

    def run(self, file_paths: List[str]) -> None:
        self._cancelled = False
        added   = 0
        skipped = 0
        total   = len(file_paths)

        # Load CLIP (mandatory)
        try:
            clip_model = clip_module.get_clip_model()
        except Exception as e:
            self._on_error(f"CLIP load failed: {e}")
            return

        for i, path in enumerate(file_paths):
            if self._cancelled:
                break

            filename = Path(path).name
            self._on_progress(i + 1, total, filename)

            if not LibraryManager.is_supported_image(path):
                skipped += 1
                continue

            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                skipped += 1
                continue

            try:
                uid = self._lib.generate_uid()

                # ---- Semantic vector (CLIP image) -------------------
                semantic_vec = normalize(clip_model.encode_image(img))
                sketch_fit_vec = None
                sketch_crop_vec = None

                # ---- Auto tags（限定建筑词表 zero-shot）--------------
                try:
                    auto_tags = auto_tagger.tags_for_vector(semantic_vec)
                except Exception:
                    auto_tags = []

                # ---- Sketch vector (Canny 64×64 raw, L2) -----------
                try:
                    edge_img  = image_proc.extract_edges(img)
                    edge_gray = edge_img.convert("L").resize((64, 64))
                    edge_arr  = np.array(edge_gray, dtype=np.float32) / 255.0
                    sketch_vec = normalize(edge_arr.flatten())
                    sketch_fit_vec, sketch_crop_vec = (
                        image_proc.structure_dual_vectors_from_edges(edge_img)
                    )
                except Exception:
                    sketch_vec = np.zeros(4096, dtype=np.float32)

                # ---- Duplicate detection hashes --------------------
                file_hash = image_proc.compute_file_sha256(path)
                phash, phash_flip = image_proc.duplicate_hashes(img)
                dupes = self._lib.find_duplicate_candidates(
                    file_hash=file_hash,
                    phash=phash,
                    phash_flip=phash_flip,
                    max_distance=10,
                    limit=1,
                )
                duplicate_group = dupes[0]["uid"] if dupes else ""
                duplicate_kind = dupes[0]["kind"] if dupes else ""

                # ---- Color analysis --------------------------------
                stat           = Path(path).stat()
                dominant_color = image_proc.extract_dominant_color(img)
                # Basic palette for backward-compat
                palette        = image_proc.extract_color_palette(img, n=6)
                # Palette with frequency ratios + adaptive sampling
                palette_ratio  = image_proc.extract_color_palette_with_ratio(
                    img, n=12, file_size=stat.st_size
                )

                # ---- Thumbnail -------------------------------------
                thumb = image_proc.generate_thumbnail(img, size=360)

                # ---- Slot version tracking --------------------------
                today = date.today().isoformat()
                semantic_model = (
                    "clip-vit-b32-mclip"
                    if getattr(clip_model, "multilingual", False)
                    else "clip-vit-b32"
                )
                indexed_slots: dict = {
                    "semantic": {
                        "model": semantic_model,
                        "version": "1.0",
                        "date": today,
                    },
                    "sketch": {
                        "model": "canny-64x64",
                        "version": "1.0",
                        "date": today,
                    },
                }
                if sketch_fit_vec is not None and sketch_crop_vec is not None:
                    indexed_slots["sketch_fit"] = {
                        "model": "canny-longside-pad-128x128",
                        "version": "2.0",
                        "date": today,
                    }
                    indexed_slots["sketch_crop"] = {
                        "model": "canny-shortside-crop-128x128",
                        "version": "2.0",
                        "date": today,
                    }

                # ---- Build record ----------------------------------
                record = make_image_record(
                    uid=uid,
                    name=Path(path).stem,
                    ext=Path(path).suffix.lower().lstrip("."),
                    width=img.width,
                    height=img.height,
                    file_size=stat.st_size,
                    tags=auto_tags,
                    dominant_color=dominant_color,
                    color_palette=palette,
                    color_palette_ratio=palette_ratio,
                    original_filename=Path(path).name,   # O(1) path lookup
                    source_path=str(path),
                    indexed_slots=indexed_slots,          # Model version tracking
                    file_hash=file_hash,
                    phash=phash,
                    phash_flip=phash_flip,
                    duplicate_group=duplicate_group,
                    duplicate_kind=duplicate_kind,
                )

                # ---- Persist ---------------------------------------
                self._lib.save_image_entry(
                    uid=uid,
                    src_image_path=path,
                    thumbnail=thumb,
                    record=record,
                )
                self._idx.add_vectors(
                    uid,
                    semantic_vec,
                    sketch_vec,
                    sketch_fit_vec=sketch_fit_vec,
                    sketch_crop_vec=sketch_crop_vec,
                )

                added += 1
                self._on_item_indexed(uid)

            except Exception:
                skipped += 1
                continue

        # Save per-slot FAISS indices to library directory
        if self._lib.index_dir and added > 0:
            try:
                self._idx.save(str(self._lib.index_dir))
            except Exception as e:
                self._on_error(f"Failed to save index: {e}")

        self._on_finished(added, skipped)
