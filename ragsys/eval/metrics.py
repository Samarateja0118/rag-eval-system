"""Retrieval metrics against FinanceBench page-level evidence.

A retrieved chunk is relevant to an evidence item if it comes from the evidence document and its
page span contains the evidence page. A question may have 1-3 evidence items (pages).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ragsys.corpus import Question
from ragsys.store import Index


def relevant_items(index: Index, chunk_id: int, q: Question) -> set[int]:
    c = index.chunks[chunk_id]
    return {
        i
        for i, e in enumerate(q.evidence)
        if e.doc_name == c.doc_name and c.page_start <= e.page <= c.page_end
    }


def judge(index: Index, ranked: Sequence[int], q: Question) -> list[set[int]]:
    return [relevant_items(index, cid, q) for cid in ranked]


def recall_at_k(rel: list[set[int]], n_items: int, k: int) -> float:
    covered = set().union(*rel[:k]) if rel[:k] else set()
    return len(covered) / n_items


def hit_at_k(rel: list[set[int]], k: int) -> float:
    return float(any(rel[:k]))


def mrr(rel: list[set[int]], k: int = 10) -> float:
    for r, s in enumerate(rel[:k], 1):
        if s:
            return 1.0 / r
    return 0.0


def doc_hit_at_k(index: Index, ranked: Sequence[int], q: Question, k: int) -> float:
    gold = {e.doc_name for e in q.evidence}
    return float(any(index.chunks[c].doc_name in gold for c in ranked[:k]))


KS = (1, 5, 10, 20)


def score_run(index: Index, ranked_by_q: dict[str, list[int]], questions: list[Question]) -> dict:
    rows = []
    for q in questions:
        ranked = ranked_by_q[q.qid]
        rel = judge(index, ranked, q)
        row = {"qid": q.qid, "mrr@10": mrr(rel, 10), "doc_hit@5": doc_hit_at_k(index, ranked, q, 5)}
        for k in KS:
            row[f"recall@{k}"] = recall_at_k(rel, len(q.evidence), k)
            row[f"hit@{k}"] = hit_at_k(rel, k)
        row["ctx_tokens@5"] = sum(index.chunks[c].n_tokens for c in ranked[:5])
        rows.append(row)
    keys = [k for k in rows[0] if k != "qid"]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    return {"summary": summary, "per_question": rows}


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n: int = 10_000, seed: int = 0) -> dict:
    """95% CI for mean(b - a) by resampling questions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(1)
    return {
        "diff": float(d.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
    }
