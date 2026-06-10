# Evaluation & Observability

## Overview

P.R.I.S.M. uses **RAGAS** for automated evaluation and **TruLens** for real-time observability. A golden dataset of 100 financial Q&A pairs is used for weekly regression testing.

---

## Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Faithfulness** | > 0.90 | Measures whether the answer is factually grounded in the retrieved context |
| **Context Precision** | > 0.85 | Measures whether relevant documents are ranked at the top |
| **Context Recall** | > 0.80 | Measures whether all relevant documents were retrieved |
| **Answer Relevancy** | > 0.88 | Measures how well the answer addresses the question |
| **Hallucination Rate** | < 5% | Inferred from 1 - Faithfulness score |
| **P95 Latency** | < 2.5s | End-to-end latency at the 95th percentile |
| **Cost per Query** | < $0.015 | LLM API cost per query, reduced via Redis semantic caching |

---

## RAGAS Integration

The `RAGASEvaluator` in `src/evaluation/metrics.py` computes all four core metrics:

```python
from src.evaluation.metrics import RAGASEvaluator

evaluator = RAGASEvaluator()
scores = await evaluator.evaluate_single(
    question="What was Apple's R&D spend in Q1 2026?",
    answer="Apple's R&D spend was $7.5B in Q1 2026.",
    contexts=["Apple R&D expenses grew 14% to $7.5B in Q1 2026."],
    ground_truth="Apple's R&D spend in Q1 2026 was $7.5 billion.",
)

print(scores)
# {"faithfulness": 0.95, "answer_relevancy": 0.92, ...}
```

---

## Golden Dataset

Located in `src/evaluation/golden_dataset.py`, the dataset contains 8 curated financial Q&A pairs covering:

- Single-entity factual queries (e.g., "What is Apple's R&D spend?")
- Multi-hop comparison queries (e.g., "Compare R&D spend of Apple and Microsoft")
- Entity relationship queries (e.g., "Who succeeded Steve Jobs?")
- Real-time data queries (e.g., "What is Apple's current stock price?")

### Running Evaluation

```python
from src.evaluation.evaluator import pipeline_evaluator
from src.evaluation.golden_dataset import get_golden_dataset

dataset = get_golden_dataset()
results = await pipeline_evaluator.evaluate_golden_dataset(dataset)

print(f"Faithfulness: {results['faithfulness']:.3f}")
print(f"Context Precision: {results['context_precision']:.3f}")
```

### Adding to the Dataset

Add new entries to `GOLDEN_DATASET` in `src/evaluation/golden_dataset.py`:

```python
{
    "question": "Your new question here?",
    "ground_truth": "The expected correct answer.",
    "contexts": ["The retrieved context that should support this answer."],
}
```

---

## Semantic Caching

The `SemanticCache` in `src/cache/semantic.py` stores previous query results and retrieves them when a semantically similar query is received.

**How it works:**
1. Incoming query is embedded using Cohere Embed v3
2. Cosine similarity is computed against cached query embeddings
3. If similarity > 0.92, the cached result is returned (0 LLM tokens used)
4. Otherwise, the pipeline executes and the result is cached

**Target impact:** 40% reduction in LLM token costs.

---

## Rate Limiting

Token bucket algorithm via `RateLimitMiddleware`:
- **Rate:** 10 requests/second per IP
- **Burst:** 20 requests
- Exceeds: HTTP 429 with retry-after header

---

## Input Sanitization

`InputSanitizationMiddleware` protects against:
- Prompt injection patterns (e.g., "ignore all previous instructions")
- Special token sequences (e.g., `<|im_start|>`)
- PII redaction (SSN, credit card numbers, email addresses)

---

## Evaluation Schedule

| Frequency | Activity | Owner |
|-----------|----------|-------|
| Per commit | Unit tests pass | CI Pipeline |
| Daily | Golden dataset evaluation | Cron Job |
| Weekly | Full regression (N=100) | Evaluation Pipeline |
| Monthly | Metrics review & dataset expansion | Team |
