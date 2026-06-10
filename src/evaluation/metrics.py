from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    faithfulness: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_relevancy: float = 0.0
    hallucination_rate: float = 0.0
    latency_p95: float = 0.0
    cost_per_query: float = 0.0
    sample_size: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)


class RAGASEvaluator:
    METRICS = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    async def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> dict[str, float]:
        dataset = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            dataset["ground_truth"] = [ground_truth]

        try:
            scores = evaluate(dataset, metrics=self.METRICS)
            result = {
                "faithfulness": scores.get("faithfulness", 0.0),
                "answer_relevancy": scores.get("answer_relevancy", 0.0),
                "context_precision": scores.get("context_precision", 0.0),
                "context_recall": scores.get("context_recall", 0.0),
            }
            logger.debug("ragas_evaluation_complete", scores=result)
            return result

        except Exception as exc:
            logger.error("ragas_evaluation_failed", error=str(exc))
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            }

    async def evaluate_batch(
        self,
        dataset: list[dict[str, Any]],
    ) -> EvaluationResult:
        questions = [d["question"] for d in dataset]
        answers = [d["answer"] for d in dataset]
        contexts = [d.get("contexts", []) for d in dataset]
        ground_truths = [d.get("ground_truth") for d in dataset]

        combined: dict[str, list] = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if any(ground_truths):
            combined["ground_truth"] = ground_truths

        try:
            scores = evaluate(combined, metrics=self.METRICS)

            result = EvaluationResult(
                faithfulness=scores.get("faithfulness", 0.0),
                answer_relevancy=scores.get("answer_relevancy", 0.0),
                context_precision=scores.get("context_precision", 0.0),
                context_recall=scores.get("context_recall", 0.0),
                hallucination_rate=1.0 - scores.get("faithfulness", 0.0),
                sample_size=len(dataset),
            )

            logger.info(
                "batch_evaluation_complete",
                faithfulness=result.faithfulness,
                context_precision=result.context_precision,
                sample_size=result.sample_size,
            )
            return result

        except Exception as exc:
            logger.error("batch_evaluation_failed", error=str(exc))
            return EvaluationResult(sample_size=len(dataset))
