"""Cross-encoder reranking with Qwen3-Reranker-0.6B (GGUF, llama.cpp via llama-cpp-python).

llama.cpp converts the reranker's yes/no LM head into a 2-way classifier head; with
pooling=RANK the "embedding" of a sequence is [p(yes), p(no)]. We score each (query, passage) pair
with the prompt format from the model card and sort by p(yes).
"""

from __future__ import annotations

import time
from pathlib import Path

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCT = (
    "Given a question about a company's SEC filing, judge whether the passage contains the information "
    "needed to answer it"
)


class Qwen3Reranker:
    def __init__(self, gguf: str | Path, threads: int = 4, n_ctx: int = 2048, max_doc_chars: int = 2400):
        import llama_cpp

        self.llm = llama_cpp.Llama(
            str(gguf),
            embedding=True,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_RANK,
            n_ctx=n_ctx,
            n_batch=n_ctx,
            n_ubatch=n_ctx,
            n_threads=threads,
            verbose=False,
        )
        self.max_doc_chars = max_doc_chars
        self.last_tokens = 0
        self.last_seconds = 0.0

    def _prompt(self, query: str, doc: str) -> str:
        return f"{PREFIX}<Instruct>: {INSTRUCT}\n<Query>: {query}\n<Document>: {doc[: self.max_doc_chars]}{SUFFIX}"

    def score(self, query: str, docs: list[str]) -> list[float]:
        t0 = time.perf_counter()
        out, ntok = [], 0
        for d in docs:
            p = self._prompt(query, d)
            ntok += len(self.llm.tokenize(p.encode(), special=True))
            out.append(float(self.llm.embed(p)[0]))
        self.last_tokens, self.last_seconds = ntok, time.perf_counter() - t0
        return out
