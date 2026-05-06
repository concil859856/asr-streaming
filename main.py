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

import base64
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
PORT = int(os.environ.get("PORT", "53284"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.7"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "16"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))

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
