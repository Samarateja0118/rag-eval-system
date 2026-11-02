"""On-disk index: chunk table (JSONL) + dense matrix (.npy) + BM25 (bm25s) + manifest.

Deliberately simple: the corpus is ~10^4-10^5 chunks, so exact dense search over a float32 matrix
is a few ms and avoids ANN recall loss. The `Index` object keeps metadata as numpy arrays so filters
are vectorised masks applied *before* top-k selection (pre-filtering, not post-filtering).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class ChunkRecord:
    chunk_id: int
    doc_name: str
    company: str
    gics_sector: str
    doc_type: str
    doc_period: int
    page_start: int
    page_end: int
    n_tokens: int
    text: str


FILTER_FIELDS = ("doc_name", "company", "gics_sector", "doc_type", "doc_period")


class Index:
    def __init__(self, path: Path, chunks: list[ChunkRecord], emb: np.ndarray, bm25, manifest: dict):
        self.path = path
        self.chunks = chunks
        self.emb = emb
        self.bm25 = bm25
        self.manifest = manifest
        self.meta = {f: np.array([getattr(c, f) for c in chunks]) for f in FILTER_FIELDS}
        self.page_start = np.array([c.page_start for c in chunks])
        self.page_end = np.array([c.page_end for c in chunks])

    def __len__(self) -> int:
        return len(self.chunks)

    def mask(self, filters: dict | None) -> np.ndarray | None:
        """filters: {field: value | [values]}; AND across fields, OR within a field's list."""
        if not filters:
            return None
        m = np.ones(len(self.chunks), dtype=bool)
        for k, v in filters.items():
            if k not in self.meta:
                raise KeyError(f"unknown filter field {k!r}; allowed {FILTER_FIELDS}")
            vals = v if isinstance(v, list | tuple | set) else [v]
            m &= np.isin(self.meta[k], list(vals))
        return m

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / "chunks.jsonl", "w") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c)) + "\n")
        np.save(self.path / "emb.npy", self.emb)
        self.bm25.save(str(self.path / "bm25"))
        (self.path / "manifest.json").write_text(json.dumps(self.manifest, indent=2))

    @classmethod
    def load(cls, path: Path) -> Index:
        import bm25s

        chunks = [ChunkRecord(**json.loads(line)) for line in open(path / "chunks.jsonl")]
        emb = np.load(path / "emb.npy")
        bm25 = bm25s.BM25.load(str(path / "bm25"))
        manifest = json.loads((path / "manifest.json").read_text())
        return cls(path, chunks, emb, bm25, manifest)
