from __future__ import annotations

from typing import Any

GOLDEN_DATASET: list[dict[str, Any]] = [
    {
        "question": "What was Apple's R&D spend in Q1 2026?",
        "ground_truth": "Apple's R&D spend in Q1 2026 was $7.5 billion.",
        "contexts": [
            "Apple Inc. reported Q1 2026 financial results with revenue of $94.8 billion. "
            "Research and Development expenses increased to $7.5 billion, up 14% year-over-year."
        ],
    },
    {
        "question": "Who is the CEO of Apple?",
        "ground_truth": "Tim Cook is the CEO of Apple.",
        "contexts": [
            "Tim Cook, CEO of Apple Inc., graduated from Duke University. "
            "He succeeded Steve Jobs as CEO in 2011."
        ],
    },
    {
        "question": "Compare R&D spend of Apple and Microsoft",
        "ground_truth": "Apple's R&D spend was $7.5B while Microsoft's R&D spend was $8.2B in Q1 2026.",
        "contexts": [
            "Apple's R&D expenses grew 14% to $7.5B in Q1 2026.",
            "Microsoft's R&D spend increased 18% to $8.2B in the same period.",
        ],
    },
    {
        "question": "What is the current stock price of Apple?",
        "ground_truth": "The current stock price of Apple can be found via real-time market data.",
        "contexts": [
            "Real-time stock price for AAPL can be retrieved via Yahoo Finance API. "
            "The current price reflects the latest trading session data."
        ],
    },
    {
        "question": "What companies operate in the Technology sector?",
        "ground_truth": "Apple, Microsoft, Google, and NVIDIA operate in the Technology sector.",
        "contexts": [
            "Apple Inc. operates in the Technology sector.",
            "Microsoft Corporation is classified under the Technology sector.",
            "Alphabet (Google) and NVIDIA also operate in the Technology sector.",
        ],
    },
    {
        "question": "What is Apple's market capitalization?",
        "ground_truth": "Apple's market capitalization is approximately $3.1 trillion.",
        "contexts": [
            "Apple Inc. (AAPL) has a market capitalization of approximately $3.1 trillion "
            "as of the latest trading session."
        ],
    },
    {
        "question": "Show me the revenue trend from Apple's latest earnings",
        "ground_truth": "Apple's Q1 2026 revenue was $94.8 billion.",
        "contexts": [
            "Apple Q1 2026 Earnings: Revenue of $94.8B, up 8% YoY. "
            "Net Income of $24.5B. EPS of $1.58."
        ],
    },
    {
        "question": "Which executive succeeded Steve Jobs at Apple?",
        "ground_truth": "Tim Cook succeeded Steve Jobs as CEO of Apple.",
        "contexts": [
            "Tim Cook succeeded Steve Jobs as CEO of Apple in August 2011. "
            "Cook had previously served as Apple's COO."
        ],
    },
]


def get_golden_dataset() -> list[dict[str, Any]]:
    return GOLDEN_DATASET


def get_golden_questions() -> list[str]:
    return [d["question"] for d in GOLDEN_DATASET]


def get_golden_ground_truths() -> list[str]:
    return [d.get("ground_truth", "") for d in GOLDEN_DATASET]
