from __future__ import annotations

from typing import Any

import numpy as np

from src.cache.redis_cache import redis_cache
from src.ingestion.embedder import CohereEmbedder
from src.utils.logger import get_logger

logger = get_logger(__name__)

SIMILARITY_THRESHOLD = 0.92


class SemanticCache:
    def __init__(self) -> None:
        self._embedder = CohereEmbedder()

    async def lookup(self, query: str) -> dict[str, Any] | None:
        query_embedding = await self._embedder.embed([query])
        if not query_embedding:
            return None

        query_vec = np.array(query_embedding[0])

        cached_keys_raw = await redis_cache.get("cache:index:keys")
        cached_keys: list[str] = cached_keys_raw if isinstance(cached_keys_raw, list) else []

        best_match: tuple[float, str | None] = (0.0, None)

        for key in cached_keys[-200:]:
            entry = await redis_cache.get(f"cache:embed:{key}")
            if not entry:
                continue
            cached_vec = np.array(entry.get("embedding", []))
            if cached_vec.size == 0:
                continue

            similarity = float(np.dot(query_vec, cached_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(cached_vec) + 1e-10
            ))

            if similarity > best_match[0]:
                best_match = (similarity, key)

        if best_match[0] >= SIMILARITY_THRESHOLD and best_match[1]:
            cached_result = await redis_cache.get(f"cache:result:{best_match[1]}")
            if cached_result:
                logger.info(
                    "semantic_cache_hit",
                    query=query[:50],
                    matched_key=best_match[1],
                    similarity=round(best_match[0], 3),
                )
                return cached_result

        logger.debug("semantic_cache_miss", query=query[:50])
        return None

    async def store(
        self,
        query: str,
        result: dict[str, Any],
        ttl: int = 3600,
    ) -> None:
        query_embedding = await self._embedder.embed([query])
        if not query_embedding:
            return

        import hashlib
        key = hashlib.sha256(query.encode()).hexdigest()[:16]

        await redis_cache.set(
            f"cache:embed:{key}",
            {"embedding": query_embedding[0], "query": query},
            ttl=ttl * 2,
        )
        await redis_cache.set(
            f"cache:result:{key}",
            result,
            ttl=ttl,
        )

        existing_keys = await redis_cache.get("cache:index:keys") or []
        if isinstance(existing_keys, list):
            if key not in existing_keys:
                existing_keys.append(key)
                await redis_cache.set(
                    "cache:index:keys",
                    existing_keys,
                    ttl=ttl * 2,
                )

        logger.info("semantic_cache_stored", key=key, query=query[:50])


semantic_cache = SemanticCache()
