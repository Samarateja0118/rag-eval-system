"""Retrieval experiments over all 150 FinanceBench open-source questions.

Writes results/retrieval/<run>.json (summary + per-question rows) and results/retrieval_summary.md.
There is no train/test split: 150 questions is too few to hold some out, so configuration choices
made from these tables are optimistic and the README says so.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from ragsys.config import Settings
from ragsys.corpus import load_questions
from ragsys.embed import OnnxEmbedder
from ragsys.eval.metrics import paired_bootstrap, score_run
from ragsys.filters import FilterExtractor
from ragsys.ingest.chunking import CONFIGS
from ragsys.retrieve import HybridRetriever
from ragsys.store import Index

log = logging.getLogger(__name__)
OUT = Path("results/retrieval")
DEPTH = 20


def filters_for(kind: str, q, fx: FilterExtractor) -> dict | None:
    if kind == "none":
        return None
    if kind == "auto_company":
        return fx.extract(q.question) or None
    if kind == "auto_company_year":
        return fx.extract(q.question, use_year=True) or None
    if kind == "oracle_doc":
        return {"doc_name": sorted({e.doc_name for e in q.evidence})}
    raise ValueError(kind)


def run(
    index: Index,
    emb: OnnxEmbedder,
    questions,
    mode: str,
    filt: str,
    qvecs: dict,
    alpha: float = 0.5,
    reranker=None,
    rerank_cache: dict | None = None,
) -> dict:
    fx = FilterExtractor(sorted(set(index.meta["company"].tolist())))
    r = HybridRetriever(index, emb, mode=mode, depth=100, alpha=alpha)
    ranked, lat = {}, []
    for q in questions:
        t0 = time.perf_counter()
        hits = r.search(q.question, k=DEPTH, filters=filters_for(filt, q, fx), qvec=qvecs[q.qid])
        ids = [h.chunk_id for h in hits]
        lat.append((time.perf_counter() - t0) * 1000)
        if reranker is not None:
            key = f"{q.qid}:{','.join(map(str, ids))}"
            if rerank_cache is not None and key in rerank_cache:
                scores = rerank_cache[key]
            else:
                scores = reranker.score(q.question, [index.chunks[i].text for i in ids])
                if rerank_cache is not None:
                    rerank_cache[key] = scores
            ids = [i for _, i in sorted(zip(scores, ids, strict=True), key=lambda x: -x[0])]
        ranked[q.qid] = ids
    res = score_run(index, ranked, questions)
    res["latency_ms_p50"] = float(np.percentile(lat, 50))
    res["ranked"] = ranked
    return res


def main(s: Settings, quick: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    questions = load_questions(s.raw_dir)
    emb = OnnxEmbedder(s.model_dir / s.embed_model, threads=s.threads)
    t0 = time.perf_counter()
    vecs = emb.encode([q.question for q in questions])
    log.info("embedded %d questions in %.2fs", len(questions), time.perf_counter() - t0)
    qvecs = {q.qid: v for q, v in zip(questions, vecs, strict=True)}
    results: dict[str, dict] = {}
    built = [c for c in CONFIGS if (s.data_dir / "index" / c / "manifest.json").exists()]

    def record(name: str, res: dict, **meta) -> None:
        res.update(meta)
        results[name] = res
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
        sm = res["summary"]
        log.info("%-40s R@5=%.3f R@10=%.3f MRR=%.3f", name, sm["recall@5"], sm["recall@10"], sm["mrr@10"])

    indexes = {c: Index.load(s.data_dir / "index" / c) for c in built}
    # 1. chunking x retriever (auto company filter)
    for c, idx in indexes.items():
        for mode in ("bm25", "dense", "rrf"):
            record(
                f"chunk-{c}__{mode}__auto_company",
                run(idx, emb, questions, mode, "auto_company", qvecs),
                chunking=c,
                mode=mode,
                filter="auto_company",
                rerank=False,
            )
    best_c = max(built, key=lambda c: results[f"chunk-{c}__rrf__auto_company"]["summary"]["mrr@10"])
    idx = indexes[best_c]
    log.info("best chunking by hybrid MRR@10: %s", best_c)
    # 2. filters
    for filt in ("none", "auto_company_year", "oracle_doc"):
        for mode in ("bm25", "dense", "rrf"):
            record(
                f"chunk-{best_c}__{mode}__{filt}",
                run(idx, emb, questions, mode, filt, qvecs),
                chunking=best_c,
                mode=mode,
                filter=filt,
                rerank=False,
            )
    # 3. fusion
    for a in (0.3, 0.5, 0.7):
        record(
            f"chunk-{best_c}__weighted{a}__auto_company",
            run(idx, emb, questions, "weighted", "auto_company", qvecs, alpha=a),
            chunking=best_c,
            mode=f"weighted a={a}",
            filter="auto_company",
            rerank=False,
        )
    # 4. reranker
    if not quick:
        from ragsys.rerank import Qwen3Reranker

        cache_p = OUT / f"rerank_cache_{best_c}.json"
        cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
        rr = Qwen3Reranker(s.model_dir / s.reranker_gguf, threads=s.threads)
        for filt in ("auto_company", "none"):
            t1 = time.perf_counter()
            res = run(idx, emb, questions, "rrf", filt, qvecs, reranker=rr, rerank_cache=cache)
            cache_p.write_text(json.dumps(cache))
            record(
                f"chunk-{best_c}__rrf+rerank__{filt}",
                res,
                chunking=best_c,
                mode="rrf+rerank",
                filter=filt,
                rerank=True,
                wall_s=time.perf_counter() - t1,
            )
    write_summary(results, best_c)


def write_summary(results: dict, best_c: str) -> None:
    lines = [
        "| run | chunking | retriever | filter | R@1 | R@5 | R@10 | R@20 | MRR@10 | doc hit@5 | ctx tokens@5 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        m = r["summary"]
        lines.append(
            f"| {name} | {r['chunking']} | {r['mode']} | {r['filter']} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | "
            f"{m['recall@10']:.3f} | {m['recall@20']:.3f} | {m['mrr@10']:.3f} | {m['doc_hit@5']:.3f} | {m['ctx_tokens@5']:.0f} |"
        )
    # paired bootstrap for the headline comparisons
    comps = [
        (f"chunk-{best_c}__bm25__auto_company", f"chunk-{best_c}__rrf__auto_company"),
        (f"chunk-{best_c}__dense__auto_company", f"chunk-{best_c}__rrf__auto_company"),
        (f"chunk-{best_c}__rrf__none", f"chunk-{best_c}__rrf__auto_company"),
        (f"chunk-{best_c}__rrf__auto_company", f"chunk-{best_c}__rrf+rerank__auto_company"),
    ]
    lines += [
        "",
        "Paired bootstrap (10k resamples over questions), difference in MRR@10 and Recall@5 (B - A):",
        "",
        "| A | B | dMRR@10 [95% CI] | dR@5 [95% CI] |",
        "|---|---|---|---|",
    ]
    for a, b in comps:
        if a in results and b in results:
            pa, pb = results[a]["per_question"], results[b]["per_question"]
            m = paired_bootstrap([x["mrr@10"] for x in pa], [x["mrr@10"] for x in pb])
            r5 = paired_bootstrap([x["recall@5"] for x in pa], [x["recall@5"] for x in pb])
            lines.append(
                f"| {a} | {b} | {m['diff']:+.3f} [{m['lo']:+.3f}, {m['hi']:+.3f}] | "
                f"{r5['diff']:+.3f} [{r5['lo']:+.3f}, {r5['hi']:+.3f}] |"
            )
    Path("results/retrieval_summary.md").write_text("\n".join(lines) + "\n")
