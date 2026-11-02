"""Runtime settings. Everything is overridable by environment variable so the same code runs
locally, in Docker Compose and in CI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("RAG_DATA_DIR", "data")))
    model_dir: Path = field(default_factory=lambda: Path(_env("RAG_MODEL_DIR", "models")))
    index_name: str = field(default_factory=lambda: _env("RAG_INDEX", "tok256"))
    redis_url: str = field(default_factory=lambda: _env("RAG_REDIS_URL", ""))
    embed_model: str = field(default_factory=lambda: _env("RAG_EMBED_MODEL", "all-MiniLM-L6-v2"))
    reranker_gguf: str = field(
        default_factory=lambda: _env("RAG_RERANKER_GGUF", "qwen3-reranker-0.6b-q8_0.gguf")
    )
    llm_gguf: str = field(default_factory=lambda: _env("RAG_LLM_GGUF", "qwen3-4b-instruct-2507-q4_k_m.gguf"))
    judge_gguf: str = field(
        default_factory=lambda: _env("RAG_JUDGE_GGUF", "qwen3-4b-instruct-2507-q4_k_m.gguf")
    )
    threads: int = field(default_factory=lambda: int(_env("RAG_THREADS", str(os.cpu_count() or 4))))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index" / self.index_name


def get_settings() -> Settings:
    return Settings()
