"""Ingestion pipeline: fetch -> extract pages (cached by PDF sha256) -> chunk -> embed -> BM25 -> save.

Idempotent: the manifest stores a fingerprint of (PDF hashes, chunk config, embed model); a rerun with
the same fingerprint is a no-op unless --force.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ragsys.config import Settings
from ragsys.corpus import RAW_BASE, load_doc_meta, referenced_docs
from ragsys.ingest.chunking import CONFIGS, chunk_document
from ragsys.ingest.pdf import extract_pages
from ragsys.store import ChunkRecord, Index
from ragsys.text import tokenize

log = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(settings: Settings, docs: list[str] | None = None) -> list[Path]:
    """Download FinanceBench metadata + the referenced PDFs (skips files already present)."""
    raw = settings.raw_dir
    (raw / "pdfs").mkdir(parents=True, exist_ok=True)
    for name in ("financebench_open_source.jsonl", "financebench_document_information.jsonl"):
        if not (raw / name).exists():
            urllib.request.urlretrieve(f"{RAW_BASE}/data/{name}", raw / name)
    docs = docs or referenced_docs(raw)

    def get(doc: str) -> Path:
        p = raw / "pdfs" / f"{doc}.pdf"
        if not p.exists() or p.stat().st_size == 0:
            urllib.request.urlretrieve(f"{RAW_BASE}/pdfs/{doc}.pdf", p)
        return p

    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(get, docs))


def _extract_one(args: tuple[Path, Path, str]) -> tuple[str, list[str]]:
    raw, cache, doc = args
    pdf = raw / "pdfs" / f"{doc}.pdf"
    digest = sha256(pdf)
    cp = cache / f"{doc}.json"
    if cp.exists():
        c = json.loads(cp.read_text())
        if c["sha256"] == digest:
            return doc, c["pages"]
    pages = extract_pages(pdf)
    cp.write_text(json.dumps({"sha256": digest, "pages": pages}))
    return doc, pages


def extract_all(settings: Settings, workers: int = 4, docs: list[str] | None = None) -> dict[str, list[str]]:
    """Page texts per doc, cached in data/pages/<doc>.json keyed by PDF hash."""
    raw = settings.raw_dir
    cache = settings.data_dir / "pages"
    cache.mkdir(parents=True, exist_ok=True)
    docs = docs or referenced_docs(raw)

    # PDFium is not thread-safe; use processes
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(workers) as ex:
        return dict(ex.map(_extract_one, [(raw, cache, d) for d in docs]))


def fingerprint(settings: Settings, cfg_name: str, docs: list[str] | None = None) -> str:
    raw = settings.raw_dir
    h = hashlib.sha256()
    for doc in docs or referenced_docs(raw):
        h.update(doc.encode())
        h.update(sha256(raw / "pdfs" / f"{doc}.pdf").encode())
    h.update(json.dumps(CONFIGS[cfg_name].__dict__, sort_keys=True).encode())
    h.update(settings.embed_model.encode())
    return h.hexdigest()[:16]


def build_index(
    settings: Settings, cfg_name: str, embedder, force: bool = False, docs: list[str] | None = None
) -> Index:
    import bm25s

    cfg = CONFIGS[cfg_name]
    out = settings.data_dir / "index" / cfg_name
    fp = fingerprint(settings, cfg_name, docs)
    if not force and (out / "manifest.json").exists():
        if json.loads((out / "manifest.json").read_text()).get("fingerprint") == fp:
            log.info("index %s up to date (%s)", cfg_name, fp)
            return Index.load(out)

    t0 = time.perf_counter()
    meta = load_doc_meta(settings.raw_dir)
    pages_by_doc = extract_all(settings, docs=docs)
    t_extract = time.perf_counter() - t0

    records: list[ChunkRecord] = []
    for doc in sorted(pages_by_doc):
        m = meta[doc]
        for ch in chunk_document(pages_by_doc[doc], cfg, embedder.count_tokens):
            # prefix the chunk with its document identity so both BM25 and the encoder see it
            records.append(
                ChunkRecord(
                    len(records),
                    doc,
                    m.company,
                    m.gics_sector,
                    m.doc_type,
                    m.doc_period,
                    ch.page_start,
                    ch.page_end,
                    embedder.count_tokens(ch.text),
                    ch.text,
                )
            )
    t_chunk = time.perf_counter() - t0 - t_extract

    headers = [f"{r.company} {r.doc_type.upper()} {r.doc_period}. " for r in records]
    emb = embedder.encode([h + r.text for h, r in zip(headers, records, strict=True)])
    t_embed = time.perf_counter() - t0 - t_extract - t_chunk

    bm25 = bm25s.BM25(k1=1.2, b=0.75)
    bm25.index([tokenize(h + r.text) for h, r in zip(headers, records, strict=True)], show_progress=False)
    t_bm25 = time.perf_counter() - t0 - t_extract - t_chunk - t_embed

    n_tok = np.array([r.n_tokens for r in records])
    manifest = {
        "fingerprint": fp,
        "config": cfg.__dict__,
        "embed_model": settings.embed_model,
        "n_docs": len(pages_by_doc),
        "n_pages": sum(len(p) for p in pages_by_doc.values()),
        "n_chunks": len(records),
        "tokens_per_chunk_mean": float(n_tok.mean()),
        "tokens_per_chunk_p95": float(np.percentile(n_tok, 95)),
        "frac_truncated_by_encoder": float((n_tok > 254).mean()),
        "seconds": {"extract": t_extract, "chunk": t_chunk, "embed": t_embed, "bm25": t_bm25},
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    idx = Index(out, records, emb, bm25, manifest)
    idx.save()
    log.info("built %s: %d chunks in %.1fs", cfg_name, len(records), time.perf_counter() - t0)
    return idx
