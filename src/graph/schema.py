from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeLabel(str, Enum):
    COMPANY = "Company"
    EXECUTIVE = "Executive"
    FINANCIAL_METRIC = "FinancialMetric"
    SECTOR = "Sector"


class RelationType(str, Enum):
    IS_CEO_OF = "IS_CEO_OF"
    IS_CFO_OF = "IS_CFO_OF"
    IS_CTO_OF = "IS_CTO_OF"
    OPERATES_IN = "OPERATES_IN"
    REPORTED = "REPORTED"
    SUCCEEDED = "SUCCEEDED"
    BOARD_MEMBER_OF = "BOARD_MEMBER_OF"


@dataclass
class NodeSchema:
    label: NodeLabel
    properties: dict[str, type]
    required_properties: list[str] = field(default_factory=list)
    unique_property: str | None = None


@dataclass
class RelationSchema:
    type: RelationType
    from_label: NodeLabel
    to_label: NodeLabel
    properties: dict[str, type] = field(default_factory=dict)


NODE_SCHEMAS: list[NodeSchema] = [
    NodeSchema(
        label=NodeLabel.COMPANY,
        properties={
            "ticker": str,
            "name": str,
            "sector": str,
            "description": str,
            "founded_year": int,
        },
        required_properties=["ticker", "name"],
        unique_property="ticker",
    ),
    NodeSchema(
        label=NodeLabel.EXECUTIVE,
        properties={
            "name": str,
            "title": str,
            "alma_mater": str,
            "education": str,
            "age": int,
        },
        required_properties=["name", "title"],
        unique_property="name",
    ),
    NodeSchema(
        label=NodeLabel.FINANCIAL_METRIC,
        properties={
            "quarter": str,
            "fiscal_year": int,
            "revenue": float,
            "rnd_spend": float,
            "net_income": float,
            "eps": float,
            "pe_ratio": float,
            "market_cap": float,
        },
        required_properties=["quarter", "fiscal_year"],
    ),
    NodeSchema(
        label=NodeLabel.SECTOR,
        properties={
            "name": str,
            "description": str,
        },
        required_properties=["name"],
        unique_property="name",
    ),
]


RELATION_SCHEMAS: list[RelationSchema] = [
    RelationSchema(
        type=RelationType.IS_CEO_OF,
        from_label=NodeLabel.EXECUTIVE,
        to_label=NodeLabel.COMPANY,
    ),
    RelationSchema(
        type=RelationType.IS_CFO_OF,
        from_label=NodeLabel.EXECUTIVE,
        to_label=NodeLabel.COMPANY,
    ),
    RelationSchema(
        type=RelationType.IS_CTO_OF,
        from_label=NodeLabel.EXECUTIVE,
        to_label=NodeLabel.COMPANY,
    ),
    RelationSchema(
        type=RelationType.OPERATES_IN,
        from_label=NodeLabel.COMPANY,
        to_label=NodeLabel.SECTOR,
    ),
    RelationSchema(
        type=RelationType.REPORTED,
        from_label=NodeLabel.COMPANY,
        to_label=NodeLabel.FINANCIAL_METRIC,
    ),
    RelationSchema(
        type=RelationType.SUCCEEDED,
        from_label=NodeLabel.EXECUTIVE,
        to_label=NodeLabel.EXECUTIVE,
    ),
    RelationSchema(
        type=RelationType.BOARD_MEMBER_OF,
        from_label=NodeLabel.EXECUTIVE,
        to_label=NodeLabel.COMPANY,
    ),
]


def cypher_create_node(node: NodeSchema, properties: dict[str, Any]) -> str:
    props_str = ", ".join(f"{k}: ${k}" for k in properties)
    return f"MERGE ({node.label.value.lower()}:{node.label.value} {{{props_str}}})"


def cypher_create_relation(
    rel: RelationSchema,
    from_props: dict[str, Any],
    to_props: dict[str, Any],
    rel_props: dict[str, Any] | None = None,
) -> str:
    from_key = next(iter(from_props))
    to_key = next(iter(to_props))
    rel_props_str = ""
    if rel_props:
        rel_props_str = ", ".join(f"{k}: ${k}" for k in rel_props)
        rel_props_str = f" {{{rel_props_str}}}"

    return (
        f"MATCH (a:{rel.from_label.value} {{{from_key}: ${from_key}}}) "
        f"MATCH (b:{rel.to_label.value} {{{to_key}: ${to_key}}}) "
        f"MERGE (a)-[r:{rel.type.value}{rel_props_str}]->(b)"
    )
