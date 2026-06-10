from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RedisCache:
    DEFAULT_TTL = 3600

    def __init__(self) -> None:
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        if self._client:
            return
        self._client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
        await self._client.ping()
        logger.info("redis_cache_connected", url=settings.redis_url)

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("redis_cache_disconnected")

    async def get(self, key: str) -> dict[str, Any] | None:
        if not self._client:
            return None
        try:
            data = await self._client.get(key)
            if data:
                logger.debug("cache_hit", key=key[:40])
                return json.loads(data)
            logger.debug("cache_miss", key=key[:40])
            return None
        except Exception as exc:
            logger.warning("cache_get_failed", error=str(exc))
            return None

    async def set(
        self,
        key: str,
        value: dict[str, Any],
        ttl: int = DEFAULT_TTL,
    ) -> None:
        if not self._client:
            return
        try:
            await self._client.setex(key, ttl, json.dumps(value))
            logger.debug("cache_set", key=key[:40], ttl=ttl)
        except Exception as exc:
            logger.warning("cache_set_failed", error=str(exc))

    async def delete(self, key: str) -> None:
        if not self._client:
            return
        try:
            await self._client.delete(key)
        except Exception as exc:
            logger.warning("cache_delete_failed", error=str(exc))

    async def flush(self) -> None:
        if not self._client:
            return
        try:
            await self._client.flushdb()
            logger.info("cache_flushed")
        except Exception as exc:
            logger.warning("cache_flush_failed", error=str(exc))

    def _make_key(self, components: list[str]) -> str:
        return ":".join(components)


redis_cache = RedisCache()
