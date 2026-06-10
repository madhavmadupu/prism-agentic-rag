from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cache.redis_cache import RedisCache
from src.cache.semantic import SemanticCache


@pytest.fixture(autouse=True)
def _patch_settings() -> None:
    with (
        patch("src.cache.redis_cache.settings") as mock_rc,
        patch("src.cache.semantic.settings") as mock_sc,
    ):
        mock_rc.redis_url = "redis://localhost:6379"
        mock_sc.cohere_api_key = "test-key"
        yield


class TestRedisCache:
    @pytest.mark.asyncio
    async def test_get_miss_returns_none(self) -> None:
        cache = RedisCache()
        cache._client = AsyncMock()
        cache._client.get.return_value = None

        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get(self) -> None:
        cache = RedisCache()
        cache._client = AsyncMock()
        cache._client.get.return_value = '{"answer": "test"}'

        result = await cache.get("test_key")
        assert result == {"answer": "test"}

    @pytest.mark.asyncio
    async def test_set_calls_setex(self) -> None:
        cache = RedisCache()
        cache._client = AsyncMock()

        await cache.set("key", {"data": "value"})
        cache._client.setex.assert_called_once()


class TestSemanticCache:
    @pytest.mark.asyncio
    async def test_miss_when_no_cached_keys(self) -> None:
        cache = SemanticCache()
        cache._embedder = AsyncMock()
        cache._embedder.embed.return_value = [[0.1, 0.2, 0.3]]

        with patch("src.cache.semantic.redis_cache") as mock_redis:
            mock_redis.get.return_value = None

            result = await cache.lookup("test query")
            assert result is None

    @pytest.mark.asyncio
    async def test_store_calls_redis_set(self) -> None:
        cache = SemanticCache()
        cache._embedder = AsyncMock()
        cache._embedder.embed.return_value = [[0.1, 0.2, 0.3]]

        with patch("src.cache.semantic.redis_cache") as mock_redis:
            mock_redis.get.return_value = None

            await cache.store("test query", {"answer": "test"})
            assert mock_redis.set.called
