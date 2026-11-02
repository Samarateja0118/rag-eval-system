"""End-to-end generation eval: answer every question with the full pipeline, then LLM-judge
(1) faithfulness: are the answer's claims supported by the passages it was given, and
(2) correctness: does the answer match FinanceBench's gold answer.

The judge is a local model (Qwen3-4B-Instruct-2507 Q4_K_M) with JSON-schema-constrained output. It is
the same model family as the generator, so self-preference bias is possible; judge labels were
spot-checked by hand on a sample (see results/generation_spotcheck.md).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from ragsys.config import Settings
from ragsys.corpus import load_questions
from ragsys.eval.metrics import relevant_items
from ragsys.retrieve import Hit

log = logging.getLogger(__name__)
OUT = Path("results/generation")

FAITH_INSTRUCTIONS = (
    "Task: check whether the ANSWER below is grounded in the numbered passages above. Split the ANSWER into "
    "its atomic factual claims (numbers, facts, conclusions; ignore citation markers). For each claim decide "
    "if it is directly stated in, or follows by simple arithmetic from, the passages. Do not use outside "
    "knowledge."
)
FAITH_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"claim": {"type": "string"}, "supported": {"type": "boolean"}},
                "required": ["claim", "supported"],
            },
        }
    },
    "required": ["claims"],
}
CORRECT_SYS = (
    "You grade answers to financial questions against a reference answer. An answer is correct if its key "
    "figure(s) or conclusion match the reference (allow rounding and unit differences such as $1,577M vs "
    "$1.58 billion). If the answer says it cannot answer, the verdict is abstained."
)
CORRECT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "incorrect", "abstained"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}


def judge_faithfulness(judge, passages: list[dict], answer: str) -> dict:
    from ragsys.generate import SYSTEM, passages_block

    user = f"{passages_block(passages)}{FAITH_INSTRUCTIONS}\n\nANSWER:\n{answer}"
    txt, usage, secs = judge.chat(SYSTEM, user, max_tokens=600, json_schema=FAITH_SCHEMA)
    try:
        claims = json.loads(txt)["claims"]
    except (json.JSONDecodeError, KeyError):
        claims = []
    n = len(claims)
    sup = sum(bool(c.get("supported")) for c in claims)
    return {
        "claims": claims,
        "n_claims": n,
        "n_supported": sup,
        "faithfulness": sup / n if n else None,
        "judge_seconds": secs,
    }


def judge_correctness(judge, question: str, gold: str, answer: str) -> dict:
    user = f"QUESTION: {question}\nREFERENCE ANSWER: {gold}\nCANDIDATE ANSWER: {answer}"
    txt, usage, secs = judge.chat(CORRECT_SYS, user, max_tokens=200, json_schema=CORRECT_SCHEMA)
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        d = {"verdict": "incorrect", "reason": f"unparseable judge output: {txt[:100]}"}
    d["judge_seconds"] = secs
    return d


def main(s: Settings, limit: int | None = None) -> None:
    """Generate from the exact top-5 lists of the best retrieval run (hybrid + rerank, auto company
    filter), so generation metrics sit on top of the retrieval numbers reported next to them and the
    0.6B reranker does not have to run a second time on this CPU-only box."""
    from ragsys.generate import LocalGenerator
    from ragsys.service import RagService

    OUT.mkdir(parents=True, exist_ok=True)
    svc = RagService.from_settings(s, with_reranker=False)
    judge = (
        svc.generator
        if s.judge_gguf == s.llm_gguf
        else LocalGenerator(s.model_dir / s.judge_gguf, threads=s.threads)
    )
    run_p = Path(f"results/retrieval/chunk-{s.index_name}__rrf+rerank__auto_company.json")
    ranked = json.loads(run_p.read_text())["ranked"]
    questions = load_questions(s.raw_dir)[:limit]
    out_p = OUT / "answers.jsonl"
    done = {json.loads(line)["qid"] for line in open(out_p)} if out_p.exists() else set()
    with open(out_p, "a") as f:
        for i, q in enumerate(questions):
            if q.qid in done:
                continue
            t0 = time.perf_counter()
            passages = svc.passages([Hit(c, 0.0) for c in ranked[q.qid][: svc.cfg.k]])
            g = svc.generator.answer(q.question, passages)
            r = {
                "answer": g.text,
                "passages": passages,
                "usage": {"prompt_tokens": g.prompt_tokens, "completion_tokens": g.completion_tokens},
                "timings_ms": {"generate": g.seconds * 1000},
            }
            abstained = "INSUFFICIENT CONTEXT" in r["answer"].upper()
            ctx_has_evidence = any(relevant_items(svc.index, p["chunk_id"], q) for p in r["passages"])
            faith = None if abstained else judge_faithfulness(judge, r["passages"], r["answer"])
            corr = judge_correctness(judge, q.question, q.answer, r["answer"])
            row = {
                "qid": q.qid,
                "question": q.question,
                "gold": q.answer,
                "question_type": q.question_type,
                "answer": r["answer"],
                "abstained": abstained,
                "ctx_has_evidence": ctx_has_evidence,
                "passages": [
                    {k: p[k] for k in ("doc_name", "page_start", "page_end", "chunk_id")}
                    for p in r["passages"]
                ],
                "usage": r["usage"],
                "timings_ms": r["timings_ms"],
                "faithfulness": faith,
                "correctness": corr,
                "wall_s": time.perf_counter() - t0,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            log.info(
                "%d/%d %s verdict=%s faith=%s %.0fs",
                i + 1,
                len(questions),
                q.qid,
                corr["verdict"],
                None if faith is None else faith["faithfulness"],
                row["wall_s"],
            )
    summarize(out_p)


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in open(path)]
    n = len(rows)
    fs = [
        r["faithfulness"]["faithfulness"]
        for r in rows
        if r["faithfulness"] and r["faithfulness"]["faithfulness"] is not None
    ]
    claims = sum(r["faithfulness"]["n_claims"] for r in rows if r["faithfulness"])
    sup = sum(r["faithfulness"]["n_supported"] for r in rows if r["faithfulness"])
    verdict = [r["correctness"]["verdict"] for r in rows]
    ev = [r for r in rows if r["ctx_has_evidence"]]
    nev = [r for r in rows if not r["ctx_has_evidence"]]

    def acc(rs):
        return sum(r["correctness"]["verdict"] == "correct" for r in rs) / len(rs) if rs else float("nan")

    by_type = {}
    for t in sorted({r["question_type"] for r in rows}):
        rs = [r for r in rows if r["question_type"] == t]
        by_type[t] = {"n": len(rs), "correct": acc(rs)}
    summ = {
        "n": n,
        "correct": verdict.count("correct") / n,
        "incorrect": verdict.count("incorrect") / n,
        "abstained": verdict.count("abstained") / n,
        "abstain_flag_rate": sum(r["abstained"] for r in rows) / n,
        "faithfulness_mean_per_answer": float(np.mean(fs)) if fs else None,
        "faithfulness_claim_level": sup / claims if claims else None,
        "answers_fully_faithful": float(np.mean([f == 1.0 for f in fs])) if fs else None,
        "n_judged_for_faithfulness": len(fs),
        "evidence_in_context_rate": len(ev) / n,
        "correct_when_evidence_in_context": acc(ev),
        "correct_when_evidence_missing": acc(nev),
        "by_question_type": by_type,
        "generate_s_p50": float(np.percentile([r["timings_ms"]["generate"] / 1000 for r in rows], 50)),
        "generate_s_p95": float(np.percentile([r["timings_ms"]["generate"] / 1000 for r in rows], 95)),
        "prompt_tokens_mean": float(np.mean([r["usage"]["prompt_tokens"] for r in rows])),
        "completion_tokens_mean": float(np.mean([r["usage"]["completion_tokens"] for r in rows])),
    }
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    return summ
