"""Latency + cost report.

Measures, on this machine, the online path stage by stage:
  * search only (filter extraction + query embedding + BM25 + dense + RRF), cold query-embedding cache
  * search + cross-encoder rerank of the top 20
  * full /ask (search + rerank + generation), taken from the generation eval rows
  * answer-cache hits, against Redis when RAG_REDIS_URL is set (else in-process LRU)
and converts CPU time into $ using an explicit, overridable price assumption.

usage: python scripts/latency_report.py [--n 50] [--usd-per-hour 0.17]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragsys.config import get_settings  # noqa: E402
from ragsys.corpus import load_questions  # noqa: E402


def pct(xs, p):
    return float(np.percentile(xs, p)) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="questions for the search/rerank benchmarks")
    ap.add_argument("--n-ask", type=int, default=5, help="questions for the true end-to-end /ask benchmark")
    ap.add_argument(
        "--usd-per-hour",
        type=float,
        default=0.17,
        help="ASSUMED price of the 4-vCPU machine the pipeline runs on (default ~ a 4-vCPU on-demand cloud VM)",
    )
    args = ap.parse_args()
    s = get_settings()
    from ragsys.cache import make_cache
    from ragsys.service import RagService, Timer

    svc = RagService.from_settings(s, with_generator=False)
    svc.cache = make_cache(s.redis_url)
    qs = load_questions(s.raw_dir)[: args.n]

    # warm-up (model load, first-call allocations) excluded from the numbers
    svc.search(qs[0].question, rerank=True)

    search, rerank, stages = [], [], {}
    for q in qs:
        t = Timer()
        t0 = time.perf_counter()
        svc.search(q.question + " ", rerank=False, timer=t)  # trailing space: bypass query-vector cache
        search.append((time.perf_counter() - t0) * 1000)
        t = Timer()
        t0 = time.perf_counter()
        svc.search(q.question + "  ", rerank=True, timer=t)
        rerank.append((time.perf_counter() - t0) * 1000)
        for k, v in t.stages.items():
            stages.setdefault(k, []).append(v)

    # answer cache: store then read back n answers
    from ragsys.cache import make_key

    hit = []
    for i in range(len(qs)):
        key = make_key("bench", i=i)
        svc.cache.set(key, {"answer": "x" * 600, "citations": [], "passages": [{"text": "y" * 1500}] * 5})
        t0 = time.perf_counter()
        svc.cache.get(key)
        hit.append((time.perf_counter() - t0) * 1000)

    gen_p = Path("results/generation/answers.jsonl")
    gen_rows = [json.loads(line) for line in open(gen_p)] if gen_p.exists() else []
    ask_gen = [r["timings_ms"].get("generate", 0) for r in gen_rows]

    # true end-to-end /ask on a few questions: first call misses the answer cache, second call hits it
    from ragsys.generate import LocalGenerator

    svc.generator = LocalGenerator(s.model_dir / s.llm_gguf, threads=s.threads)
    ask_total, ask_hit = [], []
    for q in qs[: args.n_ask]:
        r1 = svc.ask(q.question)
        r2 = svc.ask(q.question)
        assert r2["cached"], "second identical ask should be a cache hit"
        ask_total.append(r1["timings_ms"]["total"])
        ask_hit.append(r2["timings_ms"]["total"])
    p_tok = [r["usage"]["prompt_tokens"] for r in gen_rows]
    c_tok = [r["usage"]["completion_tokens"] for r in gen_rows]

    usd_per_ms = args.usd_per_hour / 3_600_000
    report = {
        "machine": {
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "n_search_queries": len(qs),
        "search_ms": {"p50": pct(search, 50), "p95": pct(search, 95)},
        "search_plus_rerank20_ms": {"p50": pct(rerank, 50), "p95": pct(rerank, 95)},
        "rerank_stage_ms": {
            "p50": pct(stages.get("rerank", []), 50),
            "p95": pct(stages.get("rerank", []), 95),
        },
        "embed_query_ms_p50": pct(stages.get("embed_query", []), 50),
        "retrieve_ms_p50": pct(stages.get("retrieve", []), 50),
        "answer_cache_hit_ms": {
            "backend": type(svc.cache).__name__,
            "p50": pct(hit, 50),
            "p95": pct(hit, 95),
        },
        "ask_full_ms": {
            "n": len(ask_total),
            "p50": pct(ask_total, 50),
            "max": max(ask_total) if ask_total else None,
        },
        "ask_cache_hit_ms": {
            "n": len(ask_hit),
            "p50": pct(ask_hit, 50),
            "max": max(ask_hit) if ask_hit else None,
        },
        "generate_ms_p50": pct(ask_gen, 50),
        "tokens_per_ask": {
            "prompt_mean": float(np.mean(p_tok)) if p_tok else None,
            "completion_mean": float(np.mean(c_tok)) if c_tok else None,
        },
        "cost_assumption_usd_per_hour": args.usd_per_hour,
        "cost_usd_per_1k": {
            "search": pct(search, 50) * usd_per_ms * 1000,
            "search_plus_rerank": pct(rerank, 50) * usd_per_ms * 1000,
            "ask_full": pct(ask_total, 50) * usd_per_ms * 1000 if ask_total else None,
            "ask_cache_hit": pct(ask_hit, 50) * usd_per_ms * 1000 if ask_hit else None,
        },
        "index": svc.index.manifest,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/latency_cost.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "index"}, indent=2))


if __name__ == "__main__":
    main()
