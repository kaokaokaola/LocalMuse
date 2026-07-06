"""
Auto tagger — CLIP zero-shot 打标，词表人工限定。

设计原则（v3）：
  • 标签范围被刻意限制在一份人工筛选的建筑常用词表内（约 40 个），
    分为 类型 / 材料 / 风格 / 场景 四类，避免开放词表带来的标签泛滥。
  • 只使用已有的 512-D CLIP 语义向量做余弦比对，不重新编码图像 ——
    对整库打标时通过 FAISS reconstruct 取回向量，速度极快。
  • 每张图最多 MAX_TAGS 个标签，且相似度需超过 MIN_SIM 阈值；
    达不到阈值时只保留最优的 1 个，保证标签"少而准"。

No UI, No File IO.
"""

from __future__ import annotations
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core import clip_model as clip_module

MAX_TAGS = 4
MIN_SIM = 0.21          # CLIP 图文余弦经验阈值（0.2~0.3 为较强相关）

# ------------------------------------------------------------------ #
#  人工限定词表：label（写入 tags 的短标签）→ CLIP 文本提示词
# ------------------------------------------------------------------ #

TAG_PROMPTS: Dict[str, str] = {
    # ---- 建筑类型 ----
    "house":        "a photo of a small residential house",
    "housing":      "a photo of an apartment or collective housing block",
    "tower":        "a photo of a high-rise tower or skyscraper",
    "museum":       "a photo of a museum or cultural building",
    "pavilion":     "a photo of a small architectural pavilion",
    "bridge":       "a photo of a bridge structure",
    "stadium":      "a photo of a stadium or large-span arena",
    "religious":    "a photo of a church, chapel, temple or mosque",
    "interior":     "a photo of an architectural interior space",
    "stair":        "a photo of an architectural staircase",
    "facade":       "a close-up photo of a building facade",
    "courtyard":    "a photo of a building courtyard or atrium",
    # ---- 场景 / 环境 ----
    "urban":        "a photo of a dense urban street or city block",
    "skyline":      "a photo of a city skyline",
    "landscape":    "a photo of landscape architecture, a park or garden",
    "waterfront":   "a photo of a building beside water, a lake or the sea",
    "mountain":     "a photo of architecture in mountains or forest",
    "aerial":       "an aerial drone photo of buildings from above",
    "night":        "a photo of architecture at night with artificial lighting",
    "snow":         "a photo of architecture in snow or winter",
    "sunset":       "a photo of architecture at sunset or dusk",
    # ---- 材料 ----
    "concrete":     "a photo of exposed concrete architecture",
    "timber":       "a photo of timber or wooden architecture",
    "brick":        "a photo of brick architecture",
    "glass":        "a photo of a glass curtain-wall building",
    "steel":        "a photo of exposed steel structure architecture",
    "stone":        "a photo of stone masonry architecture",
    # ---- 风格 / 表现 ----
    "minimalist":   "a photo of minimalist white architecture",
    "brutalist":    "a photo of brutalist architecture",
    "futuristic":   "a photo of futuristic parametric architecture",
    "traditional":  "a photo of traditional or vernacular architecture",
    "classical":    "a photo of classical European architecture with columns",
    "organic":      "a photo of organic curved architecture",
    "industrial":   "a photo of an industrial building or warehouse",
    "render":       "a computer rendering or visualization of a building",
    "model":        "a photo of a physical architectural scale model",
    "drawing":      "an architectural hand drawing or sketch",
    "plan":         "an architectural plan, section or technical drawing",
}

# ------------------------------------------------------------------ #
#  Lazy text-embedding cache
# ------------------------------------------------------------------ #

_lock = threading.Lock()
_labels: List[str] = []
_matrix: Optional[np.ndarray] = None   # shape (n_tags, 512), L2-normalized


def _ensure_embeddings() -> bool:
    """Encode all tag prompts once. Returns False if CLIP is unavailable."""
    global _labels, _matrix
    if _matrix is not None:
        return True
    with _lock:
        if _matrix is not None:
            return True
        try:
            model = clip_module.get_clip_model()
        except Exception:
            return False
        labels, rows = [], []
        for label, prompt in TAG_PROMPTS.items():
            try:
                vec = model.encode_text(prompt)
            except Exception:
                continue
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                continue
            labels.append(label)
            rows.append(vec / norm)
        if not rows:
            return False
        _labels = labels
        _matrix = np.vstack(rows).astype(np.float32)
        return True


def tags_for_vector(
    semantic_vec: Optional[np.ndarray],
    max_tags: int = MAX_TAGS,
    min_sim: float = MIN_SIM,
) -> List[str]:
    """
    Return the best matching tags for a normalized 512-D CLIP image vector.
    Always returns at least the single best tag (if CLIP loaded), at most
    ``max_tags`` tags whose cosine similarity exceeds ``min_sim``.
    """
    if semantic_vec is None:
        return []
    if not _ensure_embeddings():
        return []
    vec = np.asarray(semantic_vec, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return []
    vec = vec / norm
    sims = _matrix @ vec                      # (n_tags,)
    order = np.argsort(-sims)
    picked: List[Tuple[str, float]] = []
    for i in order[: max_tags * 2]:
        s = float(sims[i])
        if len(picked) >= max_tags:
            break
        if s >= min_sim or not picked:        # 保底保留最优 1 个
            picked.append((_labels[i], s))
    return [label for label, _ in picked]
