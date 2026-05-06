# Qwen3-ASR-1.7B — HTTP API

## Run the server

```bash
cd /workspace/Qwen3-ASR-1.7B
.venv/bin/python main.py
```

That's it — model loads on startup (~1 minute), then serves on `0.0.0.0:53284`.

To run detached:

```bash
nohup .venv/bin/python main.py > server.log 2>&1 &
```

## Public endpoint

```
http://47.186.29.91:53284
```

(Make sure the host/cloud firewall allows inbound TCP on 53284.)

Health check:

```bash
curl http://47.186.29.91:53284/health
```

## Endpoints

All responses: `{"text": "...", "language": "<detected>"}`.

### `POST /transcribe` — JSON

Body fields (one of `audio_base64` / `audio_url` is required):

| field          | type    | description                           |
|----------------|---------|---------------------------------------|
| `audio_base64` | string  | base64-encoded audio file bytes       |
| `audio_url`    | string  | http(s) URL to an audio file          |
| `language`     | string  | optional, e.g. `"English"`, `"Chinese"`. Omit for auto-detect. |

Example — URL:

```bash
curl -X POST http://47.186.29.91:53284/transcribe \
  -H 'Content-Type: application/json' \
  -d '{"audio_url":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}'
```

Example — base64:

```bash
B64=$(base64 -w0 my_audio.wav)
curl -X POST http://47.186.29.91:53284/transcribe \
  -H 'Content-Type: application/json' \
  -d "{\"audio_base64\":\"$B64\"}"
```

Python:

```python
import base64, requests
b64 = base64.b64encode(open("my_audio.wav","rb").read()).decode()
r = requests.post("http://47.186.29.91:53284/transcribe",
                  json={"audio_base64": b64})
print(r.json())  # {"text": "...", "language": "..."}
```

### `POST /transcribe/file` — multipart upload

```bash
curl -X POST http://47.186.29.91:53284/transcribe/file \
  -F "file=@my_audio.wav" \
  -F "language=English"   # optional
```

### `POST /v1/audio/transcriptions` — OpenAI-compatible

Works with the OpenAI Python SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://47.186.29.91:53284/v1", api_key="EMPTY")
with open("my_audio.wav","rb") as f:
    out = client.audio.transcriptions.create(model="qwen3-asr-1.7b", file=f)
print(out.text)
```

## Supported audio

Anything `librosa` / `soundfile` can decode: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, etc. Audio is internally resampled to 16 kHz.

## Supported languages (auto-detect or force via `language`)

Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian.

## Tunables (environment variables)

| var                       | default                      |
|---------------------------|------------------------------|
| `PORT`                    | `53284`                      |
| `HOST`                    | `0.0.0.0`                    |
| `QWEN3_ASR_MODEL`         | `/workspace/Qwen3-ASR-1.7B`  |
| `GPU_MEMORY_UTILIZATION`  | `0.7`                        |
| `MAX_BATCH_SIZE`          | `16`                         |
| `MAX_NEW_TOKENS`          | `2048`                       |
