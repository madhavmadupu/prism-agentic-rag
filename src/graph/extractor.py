from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI

from src.config import settings
from src.graph.schema import (
    NodeLabel,
    RelationType,
)
from src.utils.exceptions import GraphQueryError
from src.utils.logger import get_logger

logger = get_logger(__name__)


EXTRACTION_PROMPT = """You are a financial entity extractor. Extract structured entities and relationships from the given financial document text.

Return a JSON object with two keys: "entities" and "relationships".

### Entity Types:
1. Company: {{"type": "Company", "properties": {{"ticker": str, "name": str, "sector": str}}}}
2. Executive: {{"type": "Executive", "properties": {{"name": str, "title": str, "alma_mater": str}}}}
3. FinancialMetric: {{"type": "FinancialMetric", "properties": {{"quarter": str, "fiscal_year": int, "revenue": float, "rnd_spend": float, "net_income": float}}}}
4. Sector: {{"type": "Sector", "properties": {{"name": str}}}}

### Relationship Types:
- "IS_CEO_OF" (Executive -> Company)
- "IS_CFO_OF" (Executive -> Company)
- "OPERATES_IN" (Company -> Sector)
- "REPORTED" (Company -> FinancialMetric)
- "SUCCEEDED" (Executive -> Executive)

### Relationship Format:
{{"source_type": "Executive", "source_key": {{"name": "..."}}, "target_type": "Company", "target_key": {{"ticker": "..."}}, "type": "IS_CEO_OF"}}

Text to analyze:
{document_text}

Output valid JSON only, no markdown formatting."""


@dataclass
class ExtractedEntity:
    label: NodeLabel
    properties: dict[str, Any]


@dataclass
class ExtractedRelation:
    source_type: NodeLabel
    source_key: dict[str, str]
    target_type: NodeLabel
    target_key: dict[str, str]
    relation_type: RelationType


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelation] = field(default_factory=list)


TYPE_MAP: dict[str, NodeLabel] = {
    "Company": NodeLabel.COMPANY,
    "Executive": NodeLabel.EXECUTIVE,
    "FinancialMetric": NodeLabel.FINANCIAL_METRIC,
    "Sector": NodeLabel.SECTOR,
}

RELATION_MAP: dict[str, RelationType] = {
    "IS_CEO_OF": RelationType.IS_CEO_OF,
    "IS_CFO_OF": RelationType.IS_CFO_OF,
    "IS_CTO_OF": RelationType.IS_CTO_OF,
    "OPERATES_IN": RelationType.OPERATES_IN,
    "REPORTED": RelationType.REPORTED,
    "SUCCEEDED": RelationType.SUCCEEDED,
    "BOARD_MEMBER_OF": RelationType.BOARD_MEMBER_OF,
}


class EntityExtractor:
    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=settings.openai_api_key,
        )

    async def extract(self, document_text: str) -> ExtractionResult:
        if not document_text.strip():
            return ExtractionResult()

        try:
            response = await self._llm.ainvoke(
                EXTRACTION_PROMPT.format(document_text=document_text)
            )
            parsed = json.loads(response.content)
            return self._parse_response(parsed)
        except json.JSONDecodeError as exc:
            logger.error("extraction_parse_failed", error=str(exc))
            raise GraphQueryError(
                f"Failed to parse LLM extraction output: {exc}"
            ) from exc
        except Exception as exc:
            logger.error("extraction_failed", error=str(exc))
            raise GraphQueryError(f"Entity extraction failed: {exc}") from exc

    def _parse_response(self, parsed: dict) -> ExtractionResult:
        result = ExtractionResult()

        for raw in parsed.get("entities", []):
            label = TYPE_MAP.get(raw.get("type", ""))
            if not label:
                continue
            result.entities.append(
                ExtractedEntity(
                    label=label,
                    properties=raw.get("properties", {}),
                )
            )

        for raw in parsed.get("relationships", []):
            rel_type = RELATION_MAP.get(raw.get("type", ""))
            src_type = TYPE_MAP.get(raw.get("source_type", ""))
            tgt_type = TYPE_MAP.get(raw.get("target_type", ""))
            if not all([rel_type, src_type, tgt_type]):
                continue
            result.relationships.append(
                ExtractedRelation(
                    source_type=src_type,
                    source_key=raw.get("source_key", {}),
                    target_type=tgt_type,
                    target_key=raw.get("target_key", {}),
                    relation_type=rel_type,
                )
            )

        logger.info(
            "extraction_complete",
            entities=len(result.entities),
            relationships=len(result.relationships),
        )
        return result
