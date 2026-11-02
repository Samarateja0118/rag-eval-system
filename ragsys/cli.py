"""Command line entry point: `python -m ragsys.cli <command>`."""

from __future__ import annotations

import argparse
import logging
import sys

from ragsys.config import get_settings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ragsys")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download FinanceBench metadata + referenced PDFs")
    b = sub.add_parser("build", help="extract, chunk, embed and index")
    b.add_argument("--config", action="append", help="chunk config(s); default: all")
    b.add_argument("--force", action="store_true")
    e = sub.add_parser("eval-retrieval")
    e.add_argument("--quick", action="store_true", help="skip reranker runs")
    sub.add_parser("eval-generation")
    q = sub.add_parser("ask")
    q.add_argument("question")
    q.add_argument("--company")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = get_settings()

    if args.cmd == "fetch":
        from ragsys.ingest.pipeline import fetch

        print(f"{len(fetch(s))} PDFs present")
    elif args.cmd == "build":
        from ragsys.embed import OnnxEmbedder
        from ragsys.ingest.chunking import CONFIGS
        from ragsys.ingest.pipeline import build_index

        emb = OnnxEmbedder(s.model_dir / s.embed_model, threads=s.threads)
        for name in args.config or list(CONFIGS):
            idx = build_index(s, name, emb, force=args.force)
            print(name, idx.manifest["n_chunks"], "chunks", idx.manifest["seconds"])
    elif args.cmd == "eval-retrieval":
        from ragsys.eval.run_retrieval import main as run

        run(s, quick=args.quick)
    elif args.cmd == "eval-generation":
        from ragsys.eval.run_generation import main as run

        run(s)
    elif args.cmd == "ask":
        from ragsys.service import RagService

        svc = RagService.from_settings(s)
        filters = {"company": args.company} if args.company else None
        out = svc.ask(args.question, filters=filters)
        print(out["answer"])
        for c in out["citations"]:
            print(f"  [{c['n']}] {c['doc_name']} p.{c['page_start'] + 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
