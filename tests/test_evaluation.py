from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evaluation.golden_dataset import get_golden_dataset
from src.evaluation.metrics import RAGASEvaluator


class TestGoldenDataset:
    def test_dataset_has_entries(self) -> None:
        dataset = get_golden_dataset()
        assert len(dataset) >= 8

    def test_all_entries_have_required_fields(self) -> None:
        for entry in get_golden_dataset():
            assert "question" in entry
            assert "ground_truth" in entry
            assert "contexts" in entry
            assert isinstance(entry["contexts"], list)


class TestRAGASEvaluator:
    @pytest.mark.asyncio
    async def test_evaluate_single_returns_scores(self) -> None:
        evaluator = RAGASEvaluator()

        with patch("src.evaluation.metrics.evaluate") as mock_evaluate:
            mock_evaluate.return_value = {
                "faithfulness": 0.95,
                "answer_relevancy": 0.90,
                "context_precision": 0.88,
                "context_recall": 0.85,
            }

            scores = await evaluator.evaluate_single(
                question="Test?",
                answer="Test answer.",
                contexts=["Test context."],
                ground_truth="Expected.",
            )

            assert scores["faithfulness"] == 0.95
            assert scores["context_precision"] == 0.88

    @pytest.mark.asyncio
    async def test_evaluate_single_handles_error(self) -> None:
        evaluator = RAGASEvaluator()

        with patch("src.evaluation.metrics.evaluate") as mock_evaluate:
            mock_evaluate.side_effect = Exception("API error")

            scores = await evaluator.evaluate_single(
                question="Test?",
                answer="Test.",
                contexts=["Context."],
            )

            assert scores["faithfulness"] == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_batch_returns_result(self) -> None:
        evaluator = RAGASEvaluator()

        with patch("src.evaluation.metrics.evaluate") as mock_evaluate:
            mock_evaluate.return_value = {
                "faithfulness": 0.92,
                "answer_relevancy": 0.89,
                "context_precision": 0.86,
                "context_recall": 0.82,
            }

            result = await evaluator.evaluate_batch([
                {"question": "Q1", "answer": "A1", "contexts": ["C1"]},
                {"question": "Q2", "answer": "A2", "contexts": ["C2"]},
            ])

            assert result.faithfulness == 0.92
            assert result.sample_size == 2
