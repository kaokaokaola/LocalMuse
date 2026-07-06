"""
Hybrid retrieval service for LocalMuse.

Each modality is treated as an independent slot. A request can explicitly
select slots with ``enabled_modalities``; only those slots contribute to recall
and re-ranking.  v3 精简后仅保留 semantic / structure(sketch) / color 三个模态。
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from src.config.settings import Settings
from src.core import clip_model as clip_module
from src.core import image_proc
from src.core.vector_math import fuse_scores, normalize, normalize_scores
from src.infra.index_store import IndexStore
from src.infra.library_mgr import LibraryManager


COARSE_K: int = 500


def make_query_context(
    text: str = "",
    color: Optional[Tuple[int, int, int]] = None,
    sketch_path: Optional[str] = None,
    sketch_data_url: str = "",
    enabled_modalities: Optional[List[str]] = None,
    top_k: Optional[int] = None,
) -> dict:
    return {
        "text": text,
        "color": color,
        "sketch_path": sketch_path,
        "sketch_data_url": sketch_data_url,
        "enabled_modalities": enabled_modalities,
        "top_k": top_k,
    }


class SearchWorker:
    """Synchronous search worker. Call from a background thread."""

    def __init__(
        self,
        library_mgr: LibraryManager,
        index_store: IndexStore,
        settings: Settings,
    ):
        self._lib = library_mgr
        self._idx = index_store
        self._settings = settings

    def run_sync(self, query_context: dict) -> List[dict]:
        return self._search(query_context)

    def _search(self, ctx: dict) -> List[dict]:
        text: str = ctx.get("text", "").strip()
        color = ctx.get("color")
        sketch_path = ctx.get("sketch_path")
        sketch_data_url = ctx.get("sketch_data_url", "")
        enabled_modalities = ctx.get("enabled_modalities")
        enabled_set = set(enabled_modalities or [])

        weights = self._weights_for_request(enabled_set if enabled_modalities else None)

        try:
            top_k = int(ctx.get("top_k") or self._settings.top_k)
        except Exception:
            top_k = self._settings.top_k
        top_k = max(1, min(top_k, 500))
        coarse_k = max(COARSE_K, top_k)

        recall_attempted = False

        semantic_scores: Dict[str, float] = {}
        if weights.get("semantic", 0) > 0 and text:
            recall_attempted = True
            try:
                model = clip_module.get_clip_model()
                q_vec = normalize(model.encode_text(text))
                for uid, score in self._idx.search_semantic(q_vec, coarse_k):
                    semantic_scores[uid] = score
            except Exception:
                pass

        sketch_scores: Dict[str, float] = {}
        if weights.get("structure", 0) > 0 and (sketch_path or sketch_data_url):
            recall_attempted = True
            try:
                sk_img = _load_query_image(sketch_path, sketch_data_url)
                edge_img = image_proc.extract_edges(sk_img)
                fit_vec, crop_vec = image_proc.structure_dual_vectors_from_edges(edge_img)

                for uid, score in self._idx.search_sketch_fit(fit_vec, coarse_k):
                    if self._has_structure_slot(uid, "sketch_fit"):
                        sketch_scores[uid] = max(sketch_scores.get(uid, 0.0), score)
                for uid, score in self._idx.search_sketch_crop(crop_vec, coarse_k):
                    if self._has_structure_slot(uid, "sketch_crop"):
                        sketch_scores[uid] = max(sketch_scores.get(uid, 0.0), score)

                desired = min(coarse_k, max(1, self._lib.image_count()))
                structure_complete = self._structure_index_complete()
                if not structure_complete or len(sketch_scores) < desired:
                    sk_vec = _legacy_sketch_vector(edge_img)
                    for uid, score in self._idx.search_sketch(sk_vec, coarse_k):
                        if uid not in sketch_scores and self._has_indexed_slot(uid, "sketch"):
                            sketch_scores[uid] = score
            except Exception:
                pass

        faiss_ids: Set[str] = set(semantic_scores) | set(sketch_scores)

        if faiss_ids:
            candidate_ids: List[str] = list(faiss_ids)
        elif recall_attempted and not (weights.get("color", 0) > 0 and color is not None):
            return []
        else:
            candidate_ids = self._lib.list_all_ids()

        if not candidate_ids:
            return []

        color_scores: Dict[str, float] = {}
        if weights.get("color", 0) > 0 and color is not None:
            for uid in candidate_ids:
                meta = self._lib.load_image_metadata(uid)
                if meta and meta.get("dominant_color"):
                    dom = tuple(meta["dominant_color"])
                    dist = image_proc.color_distance_lab(color, dom)  # type: ignore
                    color_scores[uid] = image_proc.color_score_from_distance(dist)
                else:
                    color_scores[uid] = 0.0

        fused = fuse_scores(
            semantic_scores=normalize_scores(semantic_scores) if semantic_scores else {},
            color_scores=normalize_scores(color_scores) if color_scores else {},
            structure_scores=normalize_scores(sketch_scores) if sketch_scores else {},
            weights=weights,
            candidate_ids=candidate_ids,
        )

        results = []
        for uid, score, per_modal in fused[:top_k]:
            meta = self._lib.load_image_metadata(uid)
            if meta is None:
                continue
            thumb = self._lib.get_thumbnail_path(uid)
            results.append({
                "uid": uid,
                "score": score,
                "per_modal": per_modal,
                "meta": meta,
                "thumbnail_path": thumb,
            })
        return results

    def _weights_for_request(self, enabled: Optional[Set[str]]) -> Dict[str, float]:
        if enabled is None:
            return self._settings.effective_weights()
        raw = {
            "semantic": self._settings.semantic_weight if "semantic" in enabled else 0.0,
            "color": self._settings.color_weight if "color" in enabled else 0.0,
            "structure": (
                self._settings.structure_weight
                if ("sketch" in enabled or "structure" in enabled)
                else 0.0
            ),
        }
        total = sum(raw.values())
        if total <= 0:
            return {k: 0.0 for k in raw}
        return {k: v / total for k, v in raw.items()}

    def _has_indexed_slot(self, uid: str, slot: str) -> bool:
        meta = self._lib.load_image_metadata(uid)
        if not meta:
            return False
        indexed = meta.get("indexed_slots", {}) or {}
        return slot in indexed

    def _has_structure_slot(self, uid: str, slot: str) -> bool:
        meta = self._lib.load_image_metadata(uid)
        if not meta:
            return False
        indexed = meta.get("indexed_slots", {}) or {}
        return slot in indexed or "sketch_dual_128" in indexed

    def _structure_index_complete(self) -> bool:
        try:
            total = self._lib.image_count()
            if total <= 0:
                return False
            fit = len(set(self._idx.slot_ids("sketch_fit")))
            crop = len(set(self._idx.slot_ids("sketch_crop")))
            return fit >= total and crop >= total
        except Exception:
            return False


def _load_query_image(path: Optional[str], data_url: str = "") -> Image.Image:
    if data_url:
        _, b64 = data_url.split(",", 1)
        img = Image.open(BytesIO(base64.b64decode(b64)))
        return _to_rgb_on_white(img)
    if not path:
        raise ValueError("No query image provided")
    return _to_rgb_on_white(Image.open(path))


def _to_rgb_on_white(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        return bg.convert("RGB")
    return image.convert("RGB")


def _legacy_sketch_vector(edge_img: Image.Image) -> np.ndarray:
    edge_gray = edge_img.convert("L").resize((64, 64))
    edge_arr = np.array(edge_gray, dtype=np.float32) / 255.0
    return normalize(edge_arr.flatten())
