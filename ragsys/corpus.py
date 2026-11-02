"""FinanceBench corpus + question loading (https://github.com/patronus-ai/financebench).

FinanceBench open-source subset: 150 analyst-written questions over public SEC filings (10-K,
10-Q, 8-K, earnings releases), each with the gold answer and page-level evidence. We index the 84
filings those questions reference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main"


@dataclass(frozen=True)
class DocMeta:
    doc_name: str
    company: str
    gics_sector: str
    doc_type: str
    doc_period: int


@dataclass(frozen=True)
class Evidence:
    doc_name: str
    page: int  # 0-indexed physical PDF page
    text: str


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    answer: str
    company: str
    doc_name: str
    question_type: str
    evidence: tuple[Evidence, ...]


def load_doc_meta(raw_dir: Path) -> dict[str, DocMeta]:
    out = {}
    for line in (raw_dir / "financebench_document_information.jsonl").read_text().splitlines():
        d = json.loads(line)
        out[d["doc_name"]] = DocMeta(
            d["doc_name"], d["company"], d["gics_sector"], d["doc_type"], int(d["doc_period"])
        )
    return out


def load_questions(raw_dir: Path) -> list[Question]:
    qs = []
    for line in (raw_dir / "financebench_open_source.jsonl").read_text().splitlines():
        d = json.loads(line)
        ev = tuple(
            Evidence(e["doc_name"], int(e["evidence_page_num"]), e["evidence_text"]) for e in d["evidence"]
        )
        qs.append(
            Question(
                d["financebench_id"],
                d["question"],
                d["answer"],
                d["company"],
                d["doc_name"],
                d["question_type"],
                ev,
            )
        )
    return qs


def referenced_docs(raw_dir: Path) -> list[str]:
    return sorted(
        {q.doc_name for q in load_questions(raw_dir)}
        | {e.doc_name for q in load_questions(raw_dir) for e in q.evidence}
    )
