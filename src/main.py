from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.cache.redis_cache import redis_cache
from src.config import settings
from src.middleware.rate_limit import RateLimitMiddleware
from src.middleware.sanitization import InputSanitizationMiddleware
from src.utils.logger import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger = get_logger("prism")
    logger.info("startup", app=settings.app_name, debug=settings.debug)
    await redis_cache.connect()
    yield
    await redis_cache.close()
    logger.info("shutdown", app=settings.app_name)


app = FastAPI(
    title=settings.app_name,
    description="Pipeline for Retrieval, Inference, & Structured Memory",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(InputSanitizationMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    rate=10.0,
    burst=20,
)

app.include_router(router)
