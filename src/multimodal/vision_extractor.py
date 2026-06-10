from __future__ import annotations

import base64
import io
from typing import Any

from langchain_openai import ChatOpenAI

from src.config import settings
from src.utils.exceptions import MultimodalError
from src.utils.logger import get_logger

logger = get_logger(__name__)

CHART_ANALYSIS_PROMPT = """You are a financial chart analyst. Analyze the provided chart or financial document image and extract:

1. **Chart Type**: (e.g., line chart, bar chart, candlestick, table)
2. **Trend Summary**: Describe the overall trend (upward, downward, volatile)
3. **Key Data Points**: Extract specific values, dates, and figures visible in the chart
4. **Annotations**: Note any labels, callouts, or highlighted regions
5. **Correlation with Text**: If context is provided, explain how the chart relates

Be precise with numbers. If exact values are not legible, provide approximate ranges.

Context (if any): {context}

Return a structured analysis."""


PDF_PAGE_ANALYSIS_PROMPT = """You are analyzing a page from a financial document (10-K, earnings report, etc.). Extract all meaningful information:

1. **Document Type & Date**: Identify the document and fiscal period
2. **Key Financial Figures**: Revenue, expenses, profit, margins, etc.
3. **Tables**: Transcribe any tables into structured markdown format
4. **Charts/Graphs**: Describe any visual data representations
5. **Narrative Highlights**: Key management discussion points

Return a comprehensive, structured extraction."""


class VisionExtractor:
    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=settings.openai_api_key,
            max_tokens=2048,
        )

    async def extract_from_image(
        self,
        image_data: bytes,
        mime_type: str = "image/png",
        context: str = "",
    ) -> str:
        try:
            base64_image = base64.b64encode(image_data).decode("utf-8")
            data_url = f"data:{mime_type};base64,{base64_image}"

            prompt = CHART_ANALYSIS_PROMPT.format(context=context)

            response = await self._llm.ainvoke(
                [
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]}
                ]
            )

            result = response.content.strip()
            logger.info(
                "vision_extraction_complete",
                content_length=len(result),
            )
            return result

        except Exception as exc:
            logger.error("vision_extraction_failed", error=str(exc))
            raise MultimodalError(f"Vision extraction failed: {exc}") from exc

    async def extract_from_pdf_page(
        self,
        page_image: bytes,
        mime_type: str = "image/png",
    ) -> str:
        try:
            base64_image = base64.b64encode(page_image).decode("utf-8")
            data_url = f"data:{mime_type};base64,{base64_image}"

            response = await self._llm.ainvoke(
                [
                    {"role": "user", "content": [
                        {"type": "text", "text": PDF_PAGE_ANALYSIS_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]}
                ]
            )

            result = response.content.strip()
            logger.info(
                "pdf_page_extraction_complete",
                content_length=len(result),
            )
            return result

        except Exception as exc:
            logger.error("pdf_page_extraction_failed", error=str(exc))
            raise MultimodalError(f"PDF page extraction failed: {exc}") from exc
