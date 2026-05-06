# Qwen3-ASR-1.7B Server

A small FastAPI HTTP server that serves Alibaba's [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) speech-to-text model with the **vLLM** backend. Send audio (URL, base64, or file upload), get back transcription + detected language.

Supports 30 languages (English, Chinese, Cantonese, Arabic, German, French, Spanish, Portuguese, Japanese, Korean, Russian, Hindi, etc.) plus 22 Chinese dialects.

---

## Requirements

- **NVIDIA GPU** with ≥ 8 GB free VRAM (model itself is ~4 GB; vLLM overhead + KV cache pushes it higher)
- **CUDA 12.x** drivers
- **Python 3.10+** (3.12 recommended)
- ~10 GB free disk for weights + venv

Tested on: RTX 4090 (24 GB), Python 3.12, CUDA 12.6, vLLM 0.14, Linux.

---

## Quick start

```bash
git clone https://github.com/<you>/qwen3-asr-server.git
cd qwen3-asr-server
./setup.sh                          # creates venv, installs deps, downloads weights
.venv/bin/python main.py            # starts server on 0.0.0.0:53284
```

That's it. Server takes ~1 min to come up (model load + CUDA graph capture).

To run it detached:

```bash
nohup .venv/bin/python main.py > server.log 2>&1 &
```

---

## Manual setup (if you don't trust the script)

```bash
# 1. Venv + deps
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# 2. Download model weights into ./Qwen3-ASR-1.7B/
.venv/bin/pip install -U "huggingface_hub[cli]"
.venv/bin/huggingface-cli download Qwen/Qwen3-ASR-1.7B \
    --local-dir Qwen3-ASR-1.7B

# 3. Generate the fast tokenizer.json (HF repo ships only the slow tokenizer files)
.venv/bin/python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen3-ASR-1.7B', use_fast=True)
tok.save_pretrained('Qwen3-ASR-1.7B')
"

# 4. Run
.venv/bin/python main.py
```

---

## Configuration

All via environment variables. Defaults shown.

| Variable                  | Default                          | Notes                                                            |
|---------------------------|----------------------------------|------------------------------------------------------------------|
| `PORT`                    | `53284`                          |                                                                  |
| `HOST`                    | `0.0.0.0`                        | Bind to all interfaces. Use `127.0.0.1` for localhost-only.      |
| `QWEN3_ASR_MODEL`         | `<repo>/Qwen3-ASR-1.7B`          | Path to model weights directory.                                 |
| `GPU_MEMORY_UTILIZATION`  | `0.7`                            | vLLM target. Lower if you share the GPU; raise for more KV cache.|
| `MAX_BATCH_SIZE`          | `16`                             |                                                                  |
| `MAX_NEW_TOKENS`          | `2048`                           | Raise for very long audio.                                       |

Example:

```bash
PORT=8000 GPU_MEMORY_UTILIZATION=0.5 .venv/bin/python main.py
```

---

## API

All transcribe endpoints return `{"text": "...", "language": "..."}`.

### `POST /transcribe` — JSON

| field          | type    | description                                                |
|----------------|---------|------------------------------------------------------------|
| `audio_base64` | string  | base64-encoded audio file bytes (one of these is required) |
| `audio_url`    | string  | http(s) URL to an audio file                               |
| `language`     | string  | optional, e.g. `"English"`, `"Chinese"` — omit for auto    |

```bash
# URL input
curl -X POST http://<host>:53284/transcribe \
  -H 'Content-Type: application/json' \
  -d '{"audio_url":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}'

# Base64 input
B64=$(base64 -w0 my_audio.wav)
curl -X POST http://<host>:53284/transcribe \
  -H 'Content-Type: application/json' \
  -d "{\"audio_base64\":\"$B64\"}"
```

### `POST /transcribe/file` — multipart upload

```bash
curl -X POST http://<host>:53284/transcribe/file \
  -F "file=@my_audio.wav" \
  -F "language=English"   # optional
```

### `POST /v1/audio/transcriptions` — OpenAI-compatible

Drop-in replacement for OpenAI's transcription endpoint. Works with the OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://<host>:53284/v1", api_key="EMPTY")
with open("my_audio.wav", "rb") as f:
    out = client.audio.transcriptions.create(
        model="qwen3-asr-1.7b", file=f
    )
print(out.text)
```

### `GET /health`

```bash
curl http://<host>:53284/health
# {"status":"ok","loaded":true}
```

---

## Python client examples

```python
import base64, requests

# From a URL
r = requests.post("http://<host>:53284/transcribe",
                  json={"audio_url": "https://example.com/clip.wav"})
print(r.json())  # {"text": "...", "language": "..."}

# From a local file via base64
with open("clip.wav", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
r = requests.post("http://<host>:53284/transcribe",
                  json={"audio_base64": b64, "language": "English"})
print(r.json())

# From a local file via multipart upload
with open("clip.wav", "rb") as f:
    r = requests.post("http://<host>:53284/transcribe/file",
                      files={"file": f},
                      data={"language": "English"})
print(r.json())
```

---

## Audio formats

Anything `soundfile` / `librosa` can decode: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, etc. Audio is internally resampled to 16 kHz mono.

---

## Troubleshooting

**`Free memory on device cuda:0 ... is less than desired GPU memory utilization`**
Another process is using the GPU. Either kill it, or lower `GPU_MEMORY_UTILIZATION` (e.g. `0.5`). Also: don't run two instances of `main.py` at once.

**`No tokenizer file found in directory`**
You skipped step 3 of the manual setup — the HF repo ships only the slow tokenizer files (`vocab.json` + `merges.txt`) and vLLM needs `tokenizer.json`. Run the snippet in step 3, or just use `./setup.sh`.

**Model dir is detected as a Mistral tokenizer**
Already handled in `main.py` — we pass `tokenizer_mode="hf"` to vLLM. (vLLM's auto-detect recursively scans the model dir; if your `.venv/` lives next to the model dir, it picks up `mistral_common`'s bundled `tokenizer.model.v1` file and mis-classifies.)

**Port already in use**
```bash
lsof -i :53284                # find the process
PORT=8000 .venv/bin/python main.py   # or just use a different port
```

**Stop a running server**
```bash
pkill -f "python main.py"
```
Note: when killing the parent, the vLLM EngineCore subprocess can sometimes be orphaned. If GPU memory isn't freed, run `pkill -f VLLM::EngineCore` too.

---

## How it works

`main.py` wraps `qwen-asr`'s `Qwen3ASRModel.LLM(...)` (which itself wraps vLLM) inside a FastAPI app. On startup it loads the model into GPU memory via vLLM, captures CUDA graphs, and serves requests with vLLM's batched scheduler. Each request resolves the audio (URL fetch, base64 decode, or temp file from upload), passes it to `model.transcribe(...)`, and returns the parsed `text` + `language` from the model's output.

---

## Files in this repo

```
main.py            # FastAPI server
requirements.txt   # Python deps (qwen-asr[vllm], fastapi, uvicorn, ...)
setup.sh           # One-shot installer
README.md          # This file
.gitignore         # Excludes weights, venv, logs
```

After `setup.sh`, you'll also have:

```
.venv/                # Python virtualenv          (gitignored)
Qwen3-ASR-1.7B/       # Model weights (~4.4 GB)    (gitignored)
server.log            # If you redirect logs       (gitignored)
```

---

## License

The model weights are released under Apache 2.0 by Alibaba. See the [Hugging Face model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) for details.
