"""Chunking strategies compared in the experiments.

All strategies emit chunks that remember which PDF pages they span, so a chunk can be judged
relevant against FinanceBench's page-level evidence labels.

Token counts use the embedding model's WordPiece tokenizer so a "256-token chunk" really is 256
tokens for the encoder (all-MiniLM-L6-v2 truncates at 256 word pieces).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'$])|\n{2,}")


@dataclass
class Chunk:
    text: str
    page_start: int  # 0-indexed, inclusive
    page_end: int  # inclusive

    @property
    def pages(self) -> range:
        return range(self.page_start, self.page_end + 1)


@dataclass(frozen=True)
class ChunkConfig:
    name: str
    strategy: str  # "page" | "tokens" | "sentences"
    size: int = 256  # max tokens per chunk
    overlap: int = 0  # tokens (tokens strategy) or sentences (sentences strategy)


CONFIGS: dict[str, ChunkConfig] = {
    "page": ChunkConfig("page", "page"),
    "tok128": ChunkConfig("tok128", "tokens", 128, 16),
    "tok256": ChunkConfig("tok256", "tokens", 256, 32),
    "tok512": ChunkConfig("tok512", "tokens", 512, 64),
    "sent256": ChunkConfig("sent256", "sentences", 256, 1),
}

TokenCounter = Callable[[str], int]


def _words_with_pages(pages: Sequence[str]) -> tuple[list[str], list[int]]:
    words, page_of = [], []
    for p, text in enumerate(pages):
        for w in text.split():
            words.append(w)
            page_of.append(p)
    return words, page_of


def chunk_pages(pages: Sequence[str]) -> list[Chunk]:
    return [Chunk(t, i, i) for i, t in enumerate(pages) if t.strip()]


def chunk_tokens(pages: Sequence[str], size: int, overlap: int, count: TokenCounter) -> list[Chunk]:
    """Sliding window over the whole document (crosses page boundaries).

    Windows are built on whitespace words and grown until the tokenizer count reaches `size`, so
    numbers like "$1,577" are never split mid-token by the chunker itself.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words, page_of = _words_with_pages(pages)
    if not words:
        return []
    # per-word token cost (cheap approximation of the joint tokenization, exact enough for sizing)
    cost = [max(1, count(w)) for w in words]
    chunks: list[Chunk] = []
    i = 0
    n = len(words)
    while i < n:
        j, total = i, 0
        while j < n and total + cost[j] <= size:
            total += cost[j]
            j += 1
        if j == i:  # single giant token-run longer than size
            j = i + 1
        chunks.append(Chunk(" ".join(words[i:j]), page_of[i], page_of[j - 1]))
        if j >= n:
            break
        # step back `overlap` tokens
        back, k = 0, j
        while k > i + 1 and back + cost[k - 1] <= overlap:
            back += cost[k - 1]
            k -= 1
        i = k
    return chunks


def chunk_sentences(pages: Sequence[str], size: int, overlap: int, count: TokenCounter) -> list[Chunk]:
    """Greedy packing of sentences (and paragraph breaks) up to `size` tokens, never splitting a
    sentence unless it alone exceeds `size`. `overlap` = number of trailing sentences repeated."""
    sents: list[tuple[str, int, int]] = []
    for p, text in enumerate(pages):
        for s in _SENT.split(text):
            s = " ".join(s.split())
            if s:
                sents.append((s, p, max(1, count(s))))
    chunks: list[Chunk] = []
    cur: list[tuple[str, int, int]] = []
    tot = 0
    for s in sents:
        if cur and tot + s[2] > size:
            chunks.append(Chunk(" ".join(x[0] for x in cur), cur[0][1], cur[-1][1]))
            cur = cur[-overlap:] if overlap else []
            tot = sum(x[2] for x in cur)
            if tot + s[2] > size:
                cur, tot = [], 0
        if s[2] > size:  # oversized sentence: hard-split by words
            for c in chunk_tokens([s[0]], size, 0, count):
                chunks.append(Chunk(c.text, s[1], s[1]))
            continue
        cur.append(s)
        tot += s[2]
    if cur:
        chunks.append(Chunk(" ".join(x[0] for x in cur), cur[0][1], cur[-1][1]))
    return chunks


def chunk_document(pages: Sequence[str], cfg: ChunkConfig, count: TokenCounter) -> list[Chunk]:
    if cfg.strategy == "page":
        return chunk_pages(pages)
    if cfg.strategy == "tokens":
        return chunk_tokens(pages, cfg.size, cfg.overlap, count)
    if cfg.strategy == "sentences":
        return chunk_sentences(pages, cfg.size, cfg.overlap, count)
    raise ValueError(f"unknown strategy {cfg.strategy}")
