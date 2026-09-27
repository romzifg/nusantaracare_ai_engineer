import asyncio
import hmac
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.schemas import AskRequest, Answer

logger = logging.getLogger("nusantaracare")
key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def error_response(status, reason, answer):
    return JSONResponse(status_code=status, content=Answer(
        answer=answer, confidence_label="low", reason_code=reason
    ).model_dump())


def create_app(agent=None, settings=None):
    settings = settings or Settings.from_env()
    settings.validate()

    @asynccontextmanager
    async def lifespan(application):
        from app.services.runtime_stats import log_stage
        started = time.perf_counter()
        log_stage("begin", started)
        if application.state.agent is None:
            from app.services.rag import build_agent
            log_stage("retrieval_imported", started)
            application.state.agent = await run_in_threadpool(build_agent, settings)
        log_stage("ready", started)
        yield

    application = FastAPI(title="NusantaraCare RAG + Bounded Agent", version="1.0.0", lifespan=lifespan)
    application.state.agent = agent
    slots = asyncio.Semaphore(settings.max_concurrent)

    @application.middleware("http")
    async def body_limit(request, call_next):
        if request.url.path == "/ask":
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 16384:
                    return error_response(413, "request_too_large", "Body maksimal 16 KiB.")
            request._body = bytes(body)
        return await call_next(request)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Nilai pertanyaan bisa ikut dalam error validasi, jadi jangan kirim balik.
        return error_response(422, "invalid_request", "Kirim question berupa teks 3–1500 karakter; tanpa field tambahan.")

    @application.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error_response(exc.status_code, "http_error", "Permintaan HTTP tidak dapat diproses.")

    @application.exception_handler(Exception)
    async def internal_error(request, exc):
        logger.error("request_failed type=%s", type(exc).__name__)
        return error_response(500, "internal_error", "Terjadi kesalahan internal.")

    @application.get("/health", response_model=Answer)
    def health():
        return Answer(answer="Proses API berjalan.", confidence_label="high", reason_code="alive")

    @application.get("/ready", response_model=Answer)
    def ready():
        if application.state.agent is None:
            return error_response(503, "not_ready", "Indeks belum siap.")
        return Answer(answer="Indeks siap. Status koneksi LLM diuji saat /ask, bukan oleh readiness.",
                      confidence_label="high", reason_code="index_ready", mode=settings.provider)

    @application.post("/ask", response_model=Answer)
    async def ask(payload: AskRequest, api_key: str | None = Security(key_header)):
        if settings.api_token and not hmac.compare_digest(api_key or "", settings.api_token):
            return error_response(401, "unauthorized", "API key tidak valid.")
        if application.state.agent is None:
            return error_response(503, "not_ready", "Indeks belum siap.")
        try:
            await asyncio.wait_for(slots.acquire(), timeout=0.05)
        except asyncio.TimeoutError:
            return error_response(429, "busy", "Kapasitas permintaan sedang penuh; coba kembali.")
        try:
            result = await run_in_threadpool(application.state.agent.ask, payload.question)
        finally:
            slots.release()
        status = 503 if result.reason_code == "model_unavailable" else 502 if result.reason_code == "invalid_model_output" else 200
        return JSONResponse(status_code=status, content=result.model_dump())

    return application


app = create_app()
