"""Grounded answer generation with a local instruct model (Qwen3-4B-Instruct-2507, Q4_K_M GGUF).

The prompt numbers each retrieved passage and asks for bracketed citations; the model is told to say
it cannot answer when the passages do not contain the answer. Decoding is greedy (temperature 0) so
eval runs are repeatable on the same hardware/build.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

SYSTEM = (
    "You are a financial analyst answering questions about SEC filings. Use ONLY the numbered passages "
    "provided. Cite the passages you use like [1] or [2][3]. Show any arithmetic briefly. If the passages "
    "do not contain enough information, reply exactly: INSUFFICIENT CONTEXT. Keep answers under 120 words."
)

_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Generation:
    text: str
    cited: list[int]
    prompt_tokens: int
    completion_tokens: int
    seconds: float


def format_context(passages: list[dict]) -> str:
    parts = []
    for i, p in enumerate(passages, 1):
        parts.append(
            f"[{i}] ({p['company']} {p['doc_type'].upper()} {p['doc_period']}, page {p['page_start'] + 1})\n{p['text']}"
        )
    return "\n\n".join(parts)


class LocalGenerator:
    def __init__(self, gguf: str | Path, threads: int = 4, n_ctx: int = 8192, max_tokens: int = 300):
        import llama_cpp

        self.llm = llama_cpp.Llama(str(gguf), n_ctx=n_ctx, n_threads=threads, n_batch=512, verbose=False)
        self.max_tokens = max_tokens
        self.model_name = Path(gguf).name

    def chat(
        self, system: str, user: str, max_tokens: int | None = None, json_schema: dict | None = None
    ) -> tuple[str, dict, float]:
        t0 = time.perf_counter()
        kw = {}
        if json_schema is not None:
            kw["response_format"] = {"type": "json_object", "schema": json_schema}
        r = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=max_tokens or self.max_tokens,
            **kw,
        )
        return r["choices"][0]["message"]["content"].strip(), r["usage"], time.perf_counter() - t0

    def answer(self, question: str, passages: list[dict]) -> Generation:
        user = f"Passages:\n{format_context(passages)}\n\nQuestion: {question}"
        text, usage, secs = self.chat(SYSTEM, user)
        cited = sorted({int(c) for c in _CITE.findall(text) if 1 <= int(c) <= len(passages)})
        return Generation(text, cited, usage["prompt_tokens"], usage["completion_tokens"], secs)
