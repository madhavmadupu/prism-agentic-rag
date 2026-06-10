from __future__ import annotations

from typing import Any

from src.config import settings
from src.evaluation.metrics import RAGASEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineEvaluator:
    def __init__(self) -> None:
        self._ragas = RAGASEvaluator()
        self._feedback_recorder = None

    async def record_query(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> dict[str, float]:
        scores = await self._ragas.evaluate_single(
            question=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

        logger.info(
            "query_recorded",
            faithfulness=scores.get("faithfulness"),
            context_precision=scores.get("context_precision"),
        )
        return scores

    async def evaluate_golden_dataset(
        self,
        dataset: list[dict[str, Any]],
    ) -> dict[str, float]:
        if not dataset:
            return {}

        result = await self._ragas.evaluate_batch(dataset)

        summary = {
            "faithfulness": result.faithfulness,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
            "answer_relevancy": result.answer_relevancy,
            "hallucination_rate": result.hallucination_rate,
            "sample_size": result.sample_size,
        }

        logger.info("golden_dataset_evaluation", **summary)
        return summary


pipeline_evaluator = PipelineEvaluator()
