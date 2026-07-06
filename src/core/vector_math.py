"""
FAISS index wrapper and score fusion logic.
Supports both Inner-Product (cosine on normalized vecs) and L2 metrics.
No UI, No File IO.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional

import numpy as np


# ------------------------------------------------------------------ #
#  VectorIndex — metric-configurable FAISS wrapper
# ------------------------------------------------------------------ #

class VectorIndex:
    """
    Thin wrapper around a FAISS flat index. Maintains a parallel
    string ID list for UID lookup.

    metric:  "ip"  → IndexFlatIP  (cosine on L2-normalized vecs)
             "l2"  → IndexFlatL2  (Euclidean distance)
    """

    def __init__(self, dim: int = 512, metric: str = "ip"):
        import faiss
        self.DIM = dim
        self._metric = metric
        self._faiss = faiss
        self._index = self._new_index()
        self._id_map: List[str] = []

    def _new_index(self):
        if self._metric == "l2":
            return self._faiss.IndexFlatL2(self.DIM)
        return self._faiss.IndexFlatIP(self.DIM)

    # ---------------------------------------------------------------- #
    #  Mutation
    # ---------------------------------------------------------------- #

    def add(self, uid: str, vector: np.ndarray) -> int:
        vec = np.asarray(vector, dtype=np.float32).reshape(1, self.DIM)
        self._index.add(vec)
        row = len(self._id_map)
        self._id_map.append(uid)
        return row

    def add_batch(self, uids: List[str], vectors: np.ndarray) -> None:
        vecs = np.asarray(vectors, dtype=np.float32)
        self._index.add(vecs)
        self._id_map.extend(uids)

    def remove_uid(self, uid: str) -> int:
        """Remove all rows for a UID by rebuilding the flat FAISS index."""
        return self.remove_uids({uid})

    def remove_uids(self, uids: set) -> int:
        """Remove all rows whose UID is in *uids* and return removed count."""
        if not uids or not self._id_map:
            return 0

        kept_uids: List[str] = []
        kept_vecs: List[np.ndarray] = []
        removed = 0

        for row, existing_uid in enumerate(self._id_map):
            if existing_uid in uids:
                removed += 1
                continue
            vec = np.empty(self.DIM, dtype=np.float32)
            self._index.reconstruct(row, vec)
            kept_uids.append(existing_uid)
            kept_vecs.append(vec)

        if removed:
            self._index = self._new_index()
            self._id_map = []
            if kept_vecs:
                self.add_batch(kept_uids, np.vstack(kept_vecs))

        return removed

    def size(self) -> int:
        return self._index.ntotal

    def get_vector(self, uid: str) -> Optional[np.ndarray]:
        """Reconstruct the stored vector for a UID (flat indexes only)."""
        try:
            row = self._id_map.index(uid)
        except ValueError:
            return None
        vec = np.empty(self.DIM, dtype=np.float32)
        self._index.reconstruct(row, vec)
        return vec

    # ---------------------------------------------------------------- #
    #  Search
    # ---------------------------------------------------------------- #

    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """
        Returns [(uid, score), …] sorted by relevance descending.
        • IP  → score ∈ [-1, 1] (higher = more similar)
        • L2  → score = 1/(1+√dist) ∈ (0, 1] (higher = more similar)
        """
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        q = np.asarray(query, dtype=np.float32).reshape(1, self.DIM)
        distances, indices = self._index.search(q, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            if self._metric == "l2":
                score = 1.0 / (1.0 + float(np.sqrt(max(0.0, dist))))
            else:
                score = float(dist)
            results.append((self._id_map[idx], score))
        # IP: already sorted desc by FAISS; L2: sorted asc by FAISS (smaller dist=more similar)
        if self._metric == "l2":
            results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ---------------------------------------------------------------- #
    #  Serialization
    # ---------------------------------------------------------------- #

    def serialize(self) -> Tuple[bytes, List[str], int, str]:
        """Returns (faiss_bytes, id_map, dim, metric)."""
        return (
            self._faiss.serialize_index(self._index).tobytes(),
            list(self._id_map),
            self.DIM,
            self._metric,
        )

    @classmethod
    def deserialize(
        cls,
        faiss_bytes: bytes,
        id_map: List[str],
        dim: int = 512,
        metric: str = "ip",
    ) -> "VectorIndex":
        import faiss
        inst = cls.__new__(cls)
        inst.DIM = dim
        inst._metric = metric
        inst._faiss = faiss
        arr = np.frombuffer(faiss_bytes, dtype=np.uint8)
        inst._index = faiss.deserialize_index(arr)
        inst._id_map = list(id_map)
        return inst

    def all_ids(self) -> List[str]:
        return list(self._id_map)


# ------------------------------------------------------------------ #
#  Normalization utilities
# ------------------------------------------------------------------ #

def normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D float32 vector."""
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def normalize_scores(
    scores: Dict[str, float],
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> Dict[str, float]:
    """Min-max normalize a {uid: score} dict to [min_val, max_val]."""
    if not scores:
        return scores
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: max_val for k in scores}
    return {
        k: min_val + (v - lo) / (hi - lo) * (max_val - min_val)
        for k, v in scores.items()
    }


# ------------------------------------------------------------------ #
#  Score fusion
# ------------------------------------------------------------------ #

def fuse_scores(
    semantic_scores:  Dict[str, float],
    color_scores:     Dict[str, float],
    structure_scores: Dict[str, float],
    weights:          Dict[str, float],
    candidate_ids:    Optional[List[str]] = None,
    depth_scores:     Optional[Dict[str, float]] = None,
    pose_scores:      Optional[Dict[str, float]] = None,
) -> List[Tuple[str, float, Dict[str, float]]]:
    """
    Fuse five score dicts into a ranked list.
    Returns [(uid, fused_score, per_modality_dict), …] sorted DESC.
    """
    _depth = depth_scores or {}
    _pose  = pose_scores  or {}

    all_ids = (
        set(candidate_ids)
        if candidate_ids is not None
        else (
            set(semantic_scores)
            | set(color_scores)
            | set(structure_scores)
            | set(_depth)
            | set(_pose)
        )
    )

    w_s = weights.get("semantic",  0.0)
    w_c = weights.get("color",     0.0)
    w_e = weights.get("structure", 0.0)
    w_d = weights.get("depth",     0.0)
    w_p = weights.get("pose",      0.0)

    results = []
    for uid in all_ids:
        s = semantic_scores.get(uid,  0.0)
        c = color_scores.get(uid,     0.0)
        e = structure_scores.get(uid, 0.0)
        d = _depth.get(uid,           0.0)
        p = _pose.get(uid,            0.0)
        score = w_s * s + w_c * c + w_e * e + w_d * d + w_p * p
        results.append((
            uid, score,
            {
                "semantic": s,
                "color": c,
                "structure": e,
                "sketch": e,
                "depth": d,
                "pose": p,
            },
        ))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
