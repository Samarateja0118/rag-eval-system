"""Two-level cache: query embeddings and final answers.

Backend is Redis when RAG_REDIS_URL is set (shared across API replicas, survives restarts) and an
in-process LRU otherwise. Values must be JSON-serialisable. Keys include the index fingerprint and pipeline config, so rebuilding the
index or changing retrieval settings can never serve a stale answer.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any


def make_key(namespace: str, **parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return f"rag:{namespace}:{hashlib.sha256(blob.encode()).hexdigest()[:32]}"


def normalise_question(q: str) -> str:
    return " ".join(q.lower().split())


class LRUCache:
    def __init__(self, max_items: int = 10_000, ttl: float = 86_400):
        self.max_items, self.ttl = max_items, ttl
        self._d: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key: str) -> Any | None:
        item = self._d.get(key)
        if item is None or item[0] < time.time():
            self._d.pop(key, None)
            self.misses += 1
            return None
        self._d.move_to_end(key)
        self.hits += 1
        return item[1]

    def set(self, key: str, value: Any) -> None:
        self._d[key] = (time.time() + self.ttl, value)
        self._d.move_to_end(key)
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)


class RedisCache:
    def __init__(self, url: str, ttl: int = 86_400):
        import redis

        self.r = redis.Redis.from_url(url)
        self.r.ping()
        self.ttl = ttl
        self.hits = self.misses = 0

    def get(self, key: str) -> Any | None:
        raw = self.r.get(key)
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    def set(self, key: str, value: Any) -> None:
        self.r.set(
            key, json.dumps(value), ex=self.ttl
        )  # JSON, not pickle: never unpickle from a network store


def make_cache(redis_url: str = ""):
    if redis_url:
        try:
            return RedisCache(redis_url)
        except Exception:  # noqa: BLE001 - degrade to local cache rather than fail requests
            pass
    return LRUCache()
