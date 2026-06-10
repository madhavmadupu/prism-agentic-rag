from __future__ import annotations

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable

from src.config import settings
from src.utils.exceptions import GraphQueryError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jClient:
    def __init__(self) -> None:
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        if self._driver:
            return
        try:
            self._driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            await self._driver.verify_connectivity()
            logger.info("neo4j_connected", uri=settings.neo4j_uri)
        except ServiceUnavailable as exc:
            raise GraphQueryError(
                f"Cannot connect to Neo4j at {settings.neo4j_uri}: {exc}"
            ) from exc

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("neo4j_disconnected")

    async def session(self) -> AsyncSession:
        if not self._driver:
            await self.connect()
        return self._driver.session()

    async def run_query(
        self,
        cypher: str,
        parameters: dict | None = None,
    ) -> list[dict]:
        async with await self.session() as session:
            try:
                result = await session.run(cypher, parameters or {})
                records = await result.data()
                logger.debug(
                    "cypher_executed",
                    query=cypher[:80],
                    record_count=len(records),
                )
                return records
            except Exception as exc:
                logger.error(
                    "cypher_failed",
                    error=str(exc),
                    query=cypher[:120],
                )
                raise GraphQueryError(f"Cypher execution failed: {exc}") from exc

    async def create_constraints(self) -> None:
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Executive) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Sector) REQUIRE s.name IS UNIQUE",
        ]
        for cypher in constraints:
            await self.run_query(cypher)
        logger.info("neo4j_constraints_created")
