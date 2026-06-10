from __future__ import annotations

import re

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.utils.logger import get_logger

logger = get_logger(__name__)

INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+all\s+(previous|prior|above)", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(previous|prior|above)", re.IGNORECASE),
    re.compile(r"system\s*(prompt|instruction|message)", re.IGNORECASE),
    re.compile(r"you\s+are\s+(now|not\s+required)", re.IGNORECASE),
    re.compile(r"<\|im_start\|>|<\|im_end\|>", re.IGNORECASE),
    re.compile(r"disregard|ignore|override", re.IGNORECASE),
]

SENSITIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b\d{16}\b"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
]


class InputSanitizationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method not in ("POST", "PUT", "PATCH"):
            return await call_next(request)

        body = await request.body()
        if not body:
            return await call_next(request)

        text = body.decode("utf-8", errors="ignore")

        for pattern in INJECTION_PATTERNS:
            if pattern.search(text):
                logger.warning("prompt_injection_detected", pattern=pattern.pattern)
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Input contains prohibited patterns.",
                        "detail": "Your query was rejected by the security filter.",
                    },
                )

        sanitized = text
        for pattern in SENSITIVE_PATTERNS:
            sanitized = pattern.sub("[REDACTED]", sanitized)

        if sanitized != text:
            logger.info("pii_redacted_in_request")

        request._body = sanitized.encode("utf-8")
        return await call_next(request)
