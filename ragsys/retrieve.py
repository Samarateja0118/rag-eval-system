"""Retrievers: BM25, dense, and hybrid fusion (RRF or min-max weighted), all filter-aware.

Filters are applied as a boolean mask *before* top-k (pre-filtering), so a filtered query always
gets k results from the allowed subset instead of k-minus-whatever survived post-filtering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ragsys.store import Index
from ragsys.text import tokenize


@dataclass
class Hit:
    chunk_id: int
    score: float


def _topk(scores: np.ndarray, k: int, mask: np.ndarray | None) -> list[Hit]:
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [Hit(int(i), float(scores[i])) for i in idx]


def bm25_scores(index: Index, query: str) -> np.ndarray:
    toks = [t for t in tokenize(query) if t in index.bm25.vocab_dict]
    if not toks:
        return np.zeros(len(index), dtype=np.float32)
    return index.bm25.get_scores(toks)


def dense_scores(index: Index, qvec: np.ndarray) -> np.ndarray:
    return index.emb @ qvec.astype(np.float32)


def bm25_search(index: Index, query: str, k: int, mask=None) -> list[Hit]:
    return _topk(bm25_scores(index, query), k, mask)


def dense_search(index: Index, qvec: np.ndarray, k: int, mask=None) -> list[Hit]:
    return _topk(dense_scores(index, qvec), k, mask)


def rrf(rankings: list[list[Hit]], k: int, c: int = 60, weights: list[float] | None = None) -> list[Hit]:
    """Reciprocal rank fusion (Cormack et al. 2009): score = sum_i w_i / (c + rank_i)."""
    weights = weights or [1.0] * len(rankings)
    agg: dict[int, float] = {}
    for w, ranking in zip(weights, rankings, strict=True):
        for r, h in enumerate(ranking):
            agg[h.chunk_id] = agg.get(h.chunk_id, 0.0) + w / (c + r + 1)
    best = sorted(agg.items(), key=lambda kv: -kv[1])[:k]
    return [Hit(i, s) for i, s in best]


def weighted(index: Index, query: str, qvec: np.ndarray, k: int, alpha: float, mask=None) -> list[Hit]:
    """Convex combination of min-max normalised BM25 and cosine scores over the allowed set."""
    b = bm25_scores(index, query).astype(np.float64)
    d = dense_scores(index, qvec).astype(np.float64)
    sel = mask if mask is not None else np.ones(len(index), dtype=bool)

    def norm(x: np.ndarray) -> np.ndarray:
        lo, hi = x[sel].min(), x[sel].max()
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    return _topk(alpha * norm(d) + (1 - alpha) * norm(b), k, mask)


class HybridRetriever:
    """mode: 'bm25' | 'dense' | 'rrf' | 'weighted'."""

    def __init__(self, index: Index, embedder, mode: str = "rrf", depth: int = 100, alpha: float = 0.5):
        self.index, self.embedder, self.mode, self.depth, self.alpha = index, embedder, mode, depth, alpha

    def search(
        self, query: str, k: int = 10, filters: dict | None = None, qvec: np.ndarray | None = None
    ) -> list[Hit]:
        mask = self.index.mask(filters)
        if self.mode == "bm25":
            return bm25_search(self.index, query, k, mask)
        if qvec is None:
            qvec = self.embedder.encode([query])[0]
        if self.mode == "dense":
            return dense_search(self.index, qvec, k, mask)
        if self.mode == "rrf":
            return rrf(
                [
                    bm25_search(self.index, query, self.depth, mask),
                    dense_search(self.index, qvec, self.depth, mask),
                ],
                k,
            )
        if self.mode == "weighted":
            return weighted(self.index, query, qvec, k, self.alpha, mask)
        raise ValueError(self.mode)
