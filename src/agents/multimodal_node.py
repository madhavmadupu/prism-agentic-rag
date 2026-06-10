from __future__ import annotations

from src.agents.state import AgentState, RetrievalResult
from src.multimodal.vision_extractor import VisionExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MultimodalNode:
    def __init__(self) -> None:
        self._extractor = VisionExtractor()

    async def run(self, state: AgentState) -> dict:
        query = state.get("query", "")

        multimodal_data = state.get("multimodal_data", None)
        if not multimodal_data:
            logger.info("multimodal_skipped_no_data", query=query[:60])
            return {"retrieval_results": []}

        image_bytes = multimodal_data.get("image_bytes")
        mime_type = multimodal_data.get("mime_type", "image/png")
        context = multimodal_data.get("context", "")

        if image_bytes:
            result = await self._extractor.extract_from_image(
                image_data=image_bytes,
                mime_type=mime_type,
                context=context,
            )
        else:
            logger.info("multimodal_skipped_no_image")
            return {"retrieval_results": []}

        retrieval = RetrievalResult(
            source_type="multimodal",
            content=result,
            score=0.95,
            metadata={
                "mime_type": mime_type,
                "context": context,
            },
        )

        logger.info("multimodal_extraction_complete", content_length=len(result))
        return {"retrieval_results": [retrieval.__dict__]}
