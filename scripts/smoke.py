"""CI smoke test on real data: one FinanceBench filing end to end (no LLM weights needed).

Downloads 3M's FY2018 10-K, builds a tok256 index with the real ONNX embedder, and checks that the
retrievers find the cash-flow-statement page for the capex question under a company filter.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragsys.config import Settings  # noqa: E402
from ragsys.corpus import load_questions  # noqa: E402
from ragsys.embed import OnnxEmbedder  # noqa: E402
from ragsys.eval.metrics import relevant_items  # noqa: E402
from ragsys.ingest.pipeline import build_index, fetch  # noqa: E402
from ragsys.retrieve import HybridRetriever  # noqa: E402

DOC = "3M_2018_10K"


def main() -> None:
    data = Path(os.environ.get("RAG_DATA_DIR", tempfile.mkdtemp()))
    s = Settings(data_dir=data, model_dir=Path("models"), index_name="tok256")
    fetch(s, docs=[DOC])
    emb = OnnxEmbedder(s.model_dir / s.embed_model, threads=2)
    idx = build_index(s, "tok256", emb, docs=[DOC])
    print("chunks:", len(idx), "manifest seconds:", idx.manifest["seconds"])
    qs = [q for q in load_questions(s.raw_dir) if q.doc_name == DOC]
    assert qs, "no questions for smoke doc"
    q = next(q for q in qs if "capital expenditure" in q.question.lower())
    # The question says "capital expenditure"; the cash-flow table says "Purchases of property, plant
    # and equipment". BM25 misses that vocabulary gap, dense retrieval bridges it - the reason for hybrid.
    found_any = False
    for mode in ("bm25", "dense", "rrf"):
        hits = HybridRetriever(idx, emb, mode=mode).search(q.question, k=10, filters={"company": "3M"})
        ranks = [r for r, h in enumerate(hits, 1) if relevant_items(idx, h.chunk_id, q)]
        print(
            f"{mode:6s} top10 pages={[idx.chunks[h.chunk_id].page_start + 1 for h in hits]} evidence ranks={ranks}"
        )
        found_any |= bool(ranks)
    assert found_any, "no retriever found the evidence page in the top 10"
    print("smoke ok")


if __name__ == "__main__":
    main()
