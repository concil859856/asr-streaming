"""Qwen3-ASR-1.7B HTTP transcription server (vLLM backend).

Endpoints:
  GET  /health                 - liveness probe
  GET  /                       - small JSON describing the API
  POST /transcribe             - JSON body: audio_base64 | audio_url, optional language
  POST /transcribe/file        - multipart upload: file=<audio>, language=<optional>
  POST /v1/audio/transcriptions - OpenAI-compatible multipart endpoint

Response shape: {"text": "...", "language": "<detected or forced>"}
"""

from __future__ import annotations

import asyncio
import base64
import collections
import logging
import os
import tempfile
import threading
import time as _time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("qwen3-asr-server")

SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_PATH = os.environ.get("QWEN3_ASR_MODEL", str(SCRIPT_DIR / "Qwen3-ASR-1.7B"))
HOST = os.environ.get("HOST", "0.0.0.0")
# Port 8114 matches the Vocence /studio/ops dashboard's STT service convention.
# Existing pre-defined pods on 53284 keep working — admin registers them
# with the explicit host port.
PORT = int(os.environ.get("PORT", "8114"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.7"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "16"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))

# Vocence /studio/ops integration knobs ------------------------------------
# Bearer token required on /healthz and /metrics. Empty / unset = open
# (back-compat with existing pre-defined pods on 47.186.29.91:53284).
STT_API_KEY = (os.environ.get("STT_API_KEY") or "").strip()
# Max concurrent transcription requests. vLLM batches internally so cap > 1
# is meaningful here (unlike single-thread services); default matches
# MAX_BATCH_SIZE so we don't queue beyond what the engine can absorb.
STT_CAP = int(os.environ.get("STT_CAP") or str(MAX_BATCH_SIZE))

state: dict = {"model": None}


class TranscribeRequest(BaseModel):
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None
    language: Optional[str] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    from qwen_asr import Qwen3ASRModel

    logger.info("Loading Qwen3-ASR model from %s ...", MODEL_PATH)
    state["model"] = Qwen3ASRModel.LLM(
        model=MODEL_PATH,
        tokenizer_mode="hf",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_inference_batch_size=MAX_BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    logger.info("Model loaded.")
    try:
        yield
    finally:
        state["model"] = None


app = FastAPI(title="Qwen3-ASR-1.7B", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Vocence /studio/ops integration: /healthz + /metrics + bearer auth +
# inflight middleware. Bolted on top of the existing endpoints — zero
# shape changes to /transcribe, /transcribe/file, /v1/audio/transcriptions.
# ---------------------------------------------------------------------------

class _Metrics:
    def __init__(self, recent_window: int = 1000) -> None:
        self._lock = threading.Lock()
        self.start_ts = _time.time()
        self.requests_total = 0
        self.requests_ok = 0
        self.requests_err: dict[str, int] = {}
        self.duration_ms_sum = 0.0
        self.duration_ms_count = 0
        self.recent_durations_ms: "collections.deque[float]" = collections.deque(maxlen=recent_window)
        self.bytes_sent_total = 0
        self.audio_ms_total = 0

    def record_success(self, duration_ms: float, bytes_sent: int = 0) -> None:
        with self._lock:
            self.requests_total += 1
            self.requests_ok += 1
            self.duration_ms_sum += duration_ms
            self.duration_ms_count += 1
            self.recent_durations_ms.append(duration_ms)
            self.bytes_sent_total += bytes_sent

    def record_error(self, code: str, duration_ms: float = 0.0) -> None:
        with self._lock:
            self.requests_total += 1
            self.requests_err[code] = self.requests_err.get(code, 0) + 1
            if duration_ms > 0:
                self.duration_ms_sum += duration_ms
                self.duration_ms_count += 1
                self.recent_durations_ms.append(duration_ms)

    def snapshot(self) -> dict:
        with self._lock:
            durations = sorted(self.recent_durations_ms)
            n = len(durations)
            def pct(p: float) -> float:
                if n == 0:
                    return 0.0
                return durations[min(n - 1, int(p * n))]
            return {
                "uptime_seconds": int(_time.time() - self.start_ts),
                "requests_total": self.requests_total,
                "requests_ok": self.requests_ok,
                "requests_err": dict(self.requests_err),
                "duration_ms_sum": self.duration_ms_sum,
                "duration_ms_count": self.duration_ms_count,
                "duration_ms_avg": (self.duration_ms_sum / self.duration_ms_count) if self.duration_ms_count else 0.0,
                "duration_ms_p50": pct(0.50),
                "duration_ms_p95": pct(0.95),
                "duration_ms_p99": pct(0.99),
                "bytes_sent_total": self.bytes_sent_total,
                "audio_ms_total": self.audio_ms_total,
            }


_metrics = _Metrics()


class _InflightTracker:
    def __init__(self, cap: int) -> None:
        self._cap = max(1, int(cap))
        self._count = 0
        self._lock = asyncio.Lock()

    @property
    def cap(self) -> int: return self._cap

    @property
    def inflight(self) -> int: return self._count

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._count >= self._cap:
                return False
            self._count += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self._count = max(0, self._count - 1)


_inflight = _InflightTracker(cap=STT_CAP)

# Endpoints that bypass inflight / metrics — the observability layer itself
# plus FastAPI's docs. Health stays open so monitoring works under load.
_OPS_PATHS = {"/healthz", "/metrics", "/health", "/", "/docs", "/redoc", "/openapi.json"}


def _check_bearer(request: Request) -> JSONResponse | None:
    if not STT_API_KEY:
        return None
    header = request.headers.get("authorization", "")
    if header == f"Bearer {STT_API_KEY}":
        return None
    return JSONResponse(
        {"type": "error", "code": "auth", "message": "missing or invalid bearer token"},
        status_code=401,
    )


@app.middleware("http")
async def _ops_middleware(request: Request, call_next):
    path = request.url.path
    if path in _OPS_PATHS:
        return await call_next(request)

    if not await _inflight.try_acquire():
        _metrics.record_error("server_busy", 0)
        return JSONResponse(
            {"type": "error", "code": "server_busy", "message": f"inflight cap ({_inflight.cap}) exhausted"},
            status_code=503,
        )

    t0 = _time.perf_counter()
    try:
        response = await call_next(request)
        elapsed = (_time.perf_counter() - t0) * 1000.0
        if response.status_code < 400:
            _metrics.record_success(elapsed)
        else:
            _metrics.record_error(f"http_{response.status_code}", elapsed)
        return response
    except Exception as e:
        _metrics.record_error(f"exception_{type(e).__name__}", (_time.perf_counter() - t0) * 1000.0)
        raise
    finally:
        await _inflight.release()


@app.get("/healthz")
def healthz(request: Request):
    """Vocence ops dashboard health probe. Bearer-auth'd when STT_API_KEY set."""
    err = _check_bearer(request)
    if err is not None:
        return err
    return JSONResponse({
        "status": "ok",
        "service": "stt",
        "model_id": MODEL_PATH,
        "sample_rate": 16000,
        "inflight": _inflight.inflight,
        "cap": _inflight.cap,
        "loaded": state["model"] is not None,
        "dev_stub": False,
    })


@app.get("/metrics")
def metrics_endpoint(request: Request):
    """Vocence ops dashboard scrape endpoint."""
    err = _check_bearer(request)
    if err is not None:
        return err
    snap = _metrics.snapshot()
    snap["service"] = "stt"
    snap["inflight"] = _inflight.inflight
    snap["cap"] = _inflight.cap
    return JSONResponse(snap)


@app.get("/")
def root():
    return {
        "model": MODEL_PATH,
        "endpoints": ["/transcribe", "/transcribe/file", "/v1/audio/transcriptions", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "loaded": state["model"] is not None}


def _transcribe(audio, language: Optional[str]) -> dict:
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        results = model.transcribe(audio=audio, language=language)
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    if not results:
        raise HTTPException(status_code=500, detail="Empty result from model")
    r = results[0]
    return {"text": getattr(r, "text", ""), "language": getattr(r, "language", None)}


def _bytes_to_tempfile(audio_bytes: bytes, suffix: str = ".audio") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(audio_bytes)
    return path


@app.post("/transcribe")
def transcribe_json(req: TranscribeRequest):
    if not req.audio_base64 and not req.audio_url:
        raise HTTPException(status_code=400, detail="Provide audio_base64 or audio_url")

    if req.audio_base64:
        try:
            audio_bytes = base64.b64decode(req.audio_base64, validate=False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")
        path = _bytes_to_tempfile(audio_bytes)
        try:
            return _transcribe(path, req.language)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    return _transcribe(req.audio_url, req.language)


@app.post("/transcribe/file")
async def transcribe_file(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    audio_bytes = await file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".audio"
    path = _bytes_to_tempfile(audio_bytes, suffix=suffix)
    try:
        return _transcribe(path, language)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),  # ignored, kept for OpenAI client compat
    language: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
):
    audio_bytes = await file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".audio"
    path = _bytes_to_tempfile(audio_bytes, suffix=suffix)
    try:
        result = _transcribe(path, language)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if response_format == "text":
        return JSONResponse(content=result["text"], media_type="text/plain")
    return result


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
