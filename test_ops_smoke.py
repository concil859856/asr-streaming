"""Non-GPU smoke test for the Vocence /studio/ops integration on asr-streaming.

Same shape as voice_clone/test_ops_smoke.py. Verifies that the FastAPI
app instantiates, /healthz + /metrics respond with the standardized
schema, the inflight cap is respected, and bearer auth gates these
endpoints when configured.

GPU-dependent paths (actual vLLM-backed transcription) are NOT exercised
here — run those on the rented 4090 box once the image is up.

Run:  python3 test_ops_smoke.py
"""
from __future__ import annotations

import os
import sys
import types


def _stub_module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# The lifespan does `from qwen_asr import Qwen3ASRModel` and only runs
# when the test client triggers startup. We never trigger startup in this
# test so we don't even need a stub. Stub anyway for defensive completeness.
if "qwen_asr" not in sys.modules:
    qa = _stub_module("qwen_asr")
    qa.Qwen3ASRModel = type("Qwen3ASRModel", (), {})

# Configure env BEFORE importing main, to test env-driven config.
os.environ["STT_API_KEY"] = "test-stt-key"
os.environ["STT_CAP"] = "2"
os.environ["PORT"] = "9114"

import main  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402


def main_test() -> int:
    print(f"PORT (module) = {main.PORT}    (expected: 9114)")
    print(f"STT_API_KEY set = {bool(main.STT_API_KEY)}    (expected: True)")
    print(f"STT_CAP = {main.STT_CAP}    (expected: 2)")
    print(f"_inflight.cap = {main._inflight.cap}    (expected: 2)")

    # IMPORTANT: don't use `with TestClient(app)` — that triggers the
    # lifespan which loads the model. Use the bare TestClient and skip
    # startup; FastAPI routes still work.
    client = TestClient(main.app, raise_server_exceptions=False)

    failures = 0

    # /healthz without bearer -> 401
    r = client.get("/healthz")
    if r.status_code != 401:
        print(f"FAIL: /healthz without bearer should be 401, got {r.status_code}")
        failures += 1
    else:
        print("PASS: /healthz without bearer -> 401")

    # /healthz with correct bearer -> 200 + expected fields
    r = client.get("/healthz", headers={"Authorization": "Bearer test-stt-key"})
    if r.status_code != 200:
        print(f"FAIL: /healthz with bearer should be 200, got {r.status_code} body={r.text[:200]}")
        failures += 1
    else:
        body = r.json()
        required = {"status", "service", "model_id", "sample_rate", "inflight", "cap", "dev_stub"}
        missing = required - set(body)
        if missing:
            print(f"FAIL: /healthz missing fields: {missing}")
            failures += 1
        elif body["service"] != "stt":
            print(f"FAIL: /healthz service={body['service']!r}, expected 'stt'")
            failures += 1
        elif body["cap"] != 2:
            print(f"FAIL: /healthz cap={body['cap']!r}, expected 2")
            failures += 1
        else:
            print(f"PASS: /healthz with bearer -> 200 service={body['service']} cap={body['cap']} inflight={body['inflight']}")

    # /metrics with bearer -> 200 + shape
    r = client.get("/metrics", headers={"Authorization": "Bearer test-stt-key"})
    if r.status_code != 200:
        print(f"FAIL: /metrics should be 200, got {r.status_code}")
        failures += 1
    else:
        body = r.json()
        required = {"uptime_seconds", "requests_total", "requests_ok", "requests_err",
                    "duration_ms_sum", "duration_ms_count", "duration_ms_p50",
                    "duration_ms_p95", "duration_ms_p99", "inflight", "cap", "service"}
        missing = required - set(body)
        if missing:
            print(f"FAIL: /metrics missing fields: {missing}")
            failures += 1
        else:
            print(f"PASS: /metrics with bearer -> 200 requests_total={body['requests_total']}")

    # /metrics with wrong bearer -> 401
    r = client.get("/metrics", headers={"Authorization": "Bearer wrong-key"})
    if r.status_code != 401:
        print(f"FAIL: /metrics with wrong bearer should be 401, got {r.status_code}")
        failures += 1
    else:
        print("PASS: /metrics with wrong bearer -> 401")

    # Legacy /health still open + 200 (kept for back-compat)
    r = client.get("/health")
    if r.status_code != 200:
        print(f"FAIL: legacy /health should be 200, got {r.status_code}")
        failures += 1
    else:
        body = r.json()
        if "status" not in body or "loaded" not in body:
            print(f"FAIL: legacy /health body shape changed: {body}")
            failures += 1
        else:
            print(f"PASS: legacy /health -> 200 loaded={body['loaded']}")

    # /healthz repeated -> all 200 (middleware skips it; doesn't consume inflight)
    for _ in range(5):
        r = client.get("/healthz", headers={"Authorization": "Bearer test-stt-key"})
        if r.status_code != 200:
            print(f"FAIL: /healthz loop iteration returned {r.status_code}")
            failures += 1
            break
    else:
        print("PASS: /healthz 5x consecutive -> all 200 (middleware skips it)")

    # Routes registered (existing endpoints + ops endpoints)
    expected_routes = {"/transcribe", "/transcribe/file", "/v1/audio/transcriptions",
                       "/healthz", "/metrics", "/health", "/"}
    actual = {r.path for r in main.app.routes if hasattr(r, "path")}
    missing = expected_routes - actual
    if missing:
        print(f"FAIL: missing routes: {missing}")
        failures += 1
    else:
        print(f"PASS: all expected routes registered ({len(expected_routes)} checked)")

    print()
    if failures == 0:
        print("=" * 50)
        print("ALL TESTS PASSED")
        print("=" * 50)
        return 0
    else:
        print("=" * 50)
        print(f"{failures} TEST(S) FAILED")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main_test())
