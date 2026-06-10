from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.multimodal.vision_extractor import VisionExtractor
from src.utils.exceptions import MultimodalError


@pytest.fixture(autouse=True)
def _patch_settings() -> None:
    with patch("src.multimodal.vision_extractor.settings") as mock_s:
        mock_s.openai_api_key = "test-key"
        yield


class TestVisionExtractor:
    @pytest.mark.asyncio
    async def test_extract_from_image_returns_analysis(self) -> None:
        extractor = VisionExtractor()
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Chart type: line chart. Trend: upward."
        mock_llm.ainvoke.return_value = mock_response
        extractor._llm = mock_llm

        result = await extractor.extract_from_image(
            image_data=b"fake-image-bytes",
            mime_type="image/png",
            context="Q1 2026 revenue chart",
        )
        assert "upward" in result

    @pytest.mark.asyncio
    async def test_extract_raises_on_error(self) -> None:
        extractor = VisionExtractor()
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("API error")
        extractor._llm = mock_llm

        with pytest.raises(MultimodalError, match="Vision extraction failed"):
            await extractor.extract_from_image(
                image_data=b"fake",
                context="test",
            )

    @pytest.mark.asyncio
    async def test_extract_from_pdf_page(self) -> None:
        extractor = VisionExtractor()
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Document: 10-K. Revenue: $94.8B."
        mock_llm.ainvoke.return_value = mock_response
        extractor._llm = mock_llm

        result = await extractor.extract_from_pdf_page(
            page_image=b"fake-pdf-page",
        )
        assert "10-K" in result
