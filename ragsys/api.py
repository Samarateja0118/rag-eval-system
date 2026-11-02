"""FastAPI service. `uvicorn ragsys.api:app` (models load once at startup)."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ragsys.config import get_settings
from ragsys.store import FILTER_FIELDS

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from ragsys.service import RagService

    s = get_settings()
    state["svc"] = RagService.from_settings(
        s,
        with_reranker=os.environ.get("RAG_RERANK", "1") == "1",
        with_generator=os.environ.get("RAG_GENERATE", "1") == "1",
    )
    yield
    state.clear()


app = FastAPI(title="rag-eval-system", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    k: int = Field(5, ge=1, le=50)
    filters: dict[str, str | int | list[str | int]] | None = None
    rerank: bool = True


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    filters: dict[str, str | int | list[str | int]] | None = None
    use_cache: bool = True


def _check_filters(f: dict | None) -> None:
    if f and (bad := set(f) - set(FILTER_FIELDS)):
        raise HTTPException(422, f"unknown filter fields {sorted(bad)}; allowed {list(FILTER_FIELDS)}")


@app.get("/healthz")
def healthz() -> dict:
    svc = state.get("svc")
    return {
        "ok": svc is not None,
        "index": svc.index.manifest["config"]["name"] if svc else None,
        "chunks": len(svc.index) if svc else 0,
    }


@app.post("/search")
def search(req: SearchRequest) -> dict:
    from ragsys.service import Timer

    _check_filters(req.filters)
    svc = state["svc"]
    timer = Timer()
    hits = svc.search(req.query, filters=req.filters, timer=timer, rerank=req.rerank, k=req.k)
    return {"results": svc.passages(hits), "timings_ms": timer.stages}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    _check_filters(req.filters)
    svc = state["svc"]
    if svc.generator is None:
        raise HTTPException(503, "generator disabled (RAG_GENERATE=0)")
    return svc.ask(req.question, filters=req.filters, use_cache=req.use_cache)


@app.get("/stats")
def stats() -> dict:
    c = state["svc"].cache
    return {"cache_hits": c.hits, "cache_misses": c.misses, "cache_backend": type(c).__name__}
