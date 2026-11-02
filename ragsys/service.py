"""The online RAG pipeline: (cache) -> filter extraction -> hybrid retrieval -> rerank -> generate.

Every stage is timed so the API and the latency report read the same numbers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ragsys.cache import make_cache, make_key, normalise_question
from ragsys.config import Settings
from ragsys.embed import OnnxEmbedder
from ragsys.filters import FilterExtractor
from ragsys.retrieve import Hit, HybridRetriever
from ragsys.store import Index


@dataclass
class PipelineConfig:
    mode: str = "rrf"  # bm25 | dense | rrf | weighted
    depth: int = 100  # candidates per retriever before fusion
    rerank_top: int = 20  # candidates sent to the cross-encoder (0 disables reranking)
    k: int = 5  # passages given to the generator
    auto_filter: bool = True  # extract company filter from the question


@dataclass
class Timer:
    stages: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, t0: float) -> float:
        now = time.perf_counter()
        self.stages[name] = self.stages.get(name, 0.0) + (now - t0) * 1000
        return now


class RagService:
    def __init__(
        self,
        index: Index,
        embedder: OnnxEmbedder,
        reranker=None,
        generator=None,
        cfg: PipelineConfig | None = None,
        cache=None,
    ):
        self.index, self.embedder, self.reranker, self.generator = index, embedder, reranker, generator
        self.cfg = cfg or PipelineConfig()
        self.cache = cache if cache is not None else make_cache("")
        self.extractor = FilterExtractor(sorted(set(index.meta["company"].tolist())))
        self.retriever = HybridRetriever(index, embedder, self.cfg.mode, self.cfg.depth)

    @classmethod
    def from_settings(
        cls, s: Settings, with_reranker: bool = True, with_generator: bool = True
    ) -> RagService:
        from ragsys.generate import LocalGenerator
        from ragsys.rerank import Qwen3Reranker

        index = Index.load(s.index_dir)
        emb = OnnxEmbedder(s.model_dir / s.embed_model, threads=s.threads)
        rr = Qwen3Reranker(s.model_dir / s.reranker_gguf, threads=s.threads) if with_reranker else None
        gen = LocalGenerator(s.model_dir / s.llm_gguf, threads=s.threads) if with_generator else None
        return cls(index, emb, rr, gen, cache=make_cache(s.redis_url))

    # -- retrieval -------------------------------------------------------------------------------
    def embed_query(self, q: str, timer: Timer) -> list[float]:
        t0 = time.perf_counter()
        key = make_key("qvec", q=q, model=self.index.manifest["embed_model"])
        v = self.cache.get(key)
        if v is None:
            v = self.embedder.encode([q])[0].tolist()
            self.cache.set(key, v)
        timer.add("embed_query", t0)
        return v

    def search(
        self,
        query: str,
        filters: dict | None = None,
        timer: Timer | None = None,
        rerank: bool | None = None,
        k: int | None = None,
    ) -> list[Hit]:
        import numpy as np

        timer = timer or Timer()
        k = k or self.cfg.k
        if filters is None and self.cfg.auto_filter:
            filters = self.extractor.extract(query) or None
        qvec = None
        if self.cfg.mode != "bm25":
            qvec = np.asarray(self.embed_query(query, timer), dtype=np.float32)
        t0 = time.perf_counter()
        do_rerank = (
            (rerank if rerank is not None else True) and self.reranker is not None and self.cfg.rerank_top > 0
        )
        n_first = self.cfg.rerank_top if do_rerank else k
        hits = self.retriever.search(query, k=n_first, filters=filters, qvec=qvec)
        t0 = timer.add("retrieve", t0)
        if do_rerank and hits:
            scores = self.reranker.score(query, [self.index.chunks[h.chunk_id].text for h in hits])
            hits = [
                Hit(h.chunk_id, s) for h, s in sorted(zip(hits, scores, strict=True), key=lambda x: -x[1])
            ]
            timer.add("rerank", t0)
        return hits[:k]

    def passages(self, hits: list[Hit]) -> list[dict]:
        out = []
        for h in hits:
            c = self.index.chunks[h.chunk_id]
            out.append(
                {
                    "chunk_id": c.chunk_id,
                    "doc_name": c.doc_name,
                    "company": c.company,
                    "doc_type": c.doc_type,
                    "doc_period": int(c.doc_period),
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "score": h.score,
                    "text": c.text,
                }
            )
        return out

    # -- generation ------------------------------------------------------------------------------
    def ask(self, question: str, filters: dict | None = None, use_cache: bool = True) -> dict:
        timer = Timer()
        t_all = time.perf_counter()
        key = make_key(
            "answer",
            q=normalise_question(question),
            filters=filters,
            index=self.index.manifest["fingerprint"],
            cfg=self.cfg.__dict__,
            llm=getattr(self.generator, "model_name", None),
        )
        if use_cache:
            t0 = time.perf_counter()
            hit = self.cache.get(key)
            timer.add("cache_lookup", t0)
            if hit is not None:
                hit = dict(hit, cached=True)
                hit["timings_ms"] = dict(timer.stages, total=(time.perf_counter() - t_all) * 1000)
                return hit
        hits = self.search(question, filters=filters, timer=timer)
        ps = self.passages(hits)
        if self.generator is None:
            raise RuntimeError("generator not loaded")
        t0 = time.perf_counter()
        g = self.generator.answer(question, ps)
        timer.add("generate", t0)
        out = {
            "answer": g.text,
            "citations": [dict(ps[i - 1], n=i) for i in g.cited],
            "passages": ps,
            "usage": {
                "prompt_tokens": g.prompt_tokens,
                "completion_tokens": g.completion_tokens,
                "rerank_tokens": getattr(self.reranker, "last_tokens", 0),
            },
            "cached": False,
        }
        out["timings_ms"] = dict(timer.stages, total=(time.perf_counter() - t_all) * 1000)
        if use_cache:
            self.cache.set(key, {k: v for k, v in out.items() if k != "timings_ms"})
        return out
