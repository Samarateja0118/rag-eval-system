import numpy as np
import pytest

from ragsys.corpus import Evidence, Question
from ragsys.eval.metrics import hit_at_k, judge, mrr, paired_bootstrap, recall_at_k
from ragsys.retrieve import Hit, HybridRetriever, rrf
from ragsys.store import ChunkRecord, Index
from ragsys.text import tokenize


class FakeEmbedder:
    """Bag-of-words hashing embedder so tests need no model files."""

    dim = 64

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, hash(tok) % self.dim] += 1
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out


def make_index():
    import bm25s

    texts = [
        ("3M", "3M_2018_10K", 2018, "capital expenditures purchases of property plant and equipment 1577"),
        ("3M", "3M_2018_10K", 2018, "headquarters in st paul minnesota employees"),
        (
            "Apple",
            "APPLE_2018_10K",
            2018,
            "capital expenditures purchases of property plant and equipment 13313",
        ),
        ("Apple", "APPLE_2018_10K", 2018, "iphone net sales by region"),
    ]
    chunks = [
        ChunkRecord(i, d, c, "IT", "10k", y, i, i, len(t.split()), t) for i, (c, d, y, t) in enumerate(texts)
    ]
    emb = FakeEmbedder().encode([c.text for c in chunks])
    bm = bm25s.BM25()
    bm.index([tokenize(c.text) for c in chunks], show_progress=False)
    return Index(None, chunks, emb, bm, {"embed_model": "fake", "fingerprint": "x", "config": {"name": "t"}})


@pytest.mark.parametrize("mode", ["bm25", "dense", "rrf", "weighted"])
def test_filters_are_applied_before_topk(mode):
    idx = make_index()
    r = HybridRetriever(idx, FakeEmbedder(), mode=mode)
    hits = r.search("capital expenditures", k=4, filters={"company": "Apple"})
    assert hits and all(idx.chunks[h.chunk_id].company == "Apple" for h in hits)
    assert len(hits) == 2  # k results from the allowed subset, not fewer


def test_unknown_filter_field_rejected():
    with pytest.raises(KeyError):
        make_index().mask({"ticker": "MMM"})


def test_rrf_rewards_agreement():
    a = [Hit(1, 9), Hit(2, 8), Hit(3, 7)]
    b = [Hit(2, 0.9), Hit(4, 0.8), Hit(1, 0.7)]
    fused = [h.chunk_id for h in rrf([a, b], k=4)]
    assert fused[0] in (1, 2) and set(fused[:2]) == {1, 2}


def test_metrics_page_level_relevance():
    idx = make_index()
    q = Question(
        "q",
        "capex 3M 2018",
        "$1577",
        "3M",
        "3M_2018_10K",
        "x",
        (Evidence("3M_2018_10K", 0, ""), Evidence("3M_2018_10K", 1, "")),
    )
    rel = judge(idx, [2, 0, 3, 1], q)
    assert recall_at_k(rel, 2, 1) == 0 and recall_at_k(rel, 2, 2) == 0.5 and recall_at_k(rel, 2, 4) == 1.0
    assert hit_at_k(rel, 2) == 1.0 and mrr(rel) == 0.5


def test_paired_bootstrap_detects_shift():
    a = np.zeros(100)
    b = np.ones(100) * 0.1
    ci = paired_bootstrap(a, b, n=500)
    assert ci["lo"] > 0 and abs(ci["diff"] - 0.1) < 1e-9
