"""
scientific-llm - Step 8a verification: the FastAPI serving layer,
tested the way the project spec asks for it - over real HTTP, the same
protocol curl speaks - not FastAPI's in-process TestClient.

Starts `uvicorn src.api.main:app` as a real subprocess (a separate
process, listening on a real port - the same thing you get running the
uvicorn command yourself), polls /health until the model finishes
loading, then exercises /generate with a real prompt and a real
response, and confirms request validation (empty prompt) still gets
enforced by the live server, not just in unit tests. Tears the server
process down cleanly afterward either way.

This subprocess+HTTP harness itself (start server, poll health, hit
endpoints, clean teardown) was proven out against a throwaway FastAPI
app with a fake generate function before being pointed at the real
model - see this project's build notes. What could not be tested ahead
of time is the real model actually being live behind the API - that is
exactly what this script checks for real, on your machine.

Run directly:
    python scripts\\verify_step8a.py
(this is slower than most verify scripts - it waits for the full model
load, the same one Step 2/4/5/7 already do, so give it several minutes
before assuming something is stuck.)
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"
HEALTH_TIMEOUT_S = 300  # model load can be slow, especially a cold cache
LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "verify_step8a_server.log"

results: list[tuple[str, bool, str]] = []


def record(label: str, passed: bool, detail: str = "") -> None:
    results.append((label, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail else ""))


def start_server(log_file) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", HOST, "--port", str(PORT)],
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def wait_for_health(timeout_s: int) -> dict | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=3) as resp:
                return json.loads(resp.read())
        except Exception:  # noqa: BLE001 - server not up yet, or still loading
            time.sleep(2)
    return None


def post_json(path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main() -> int:
    print("=" * 70)
    print("scientific-llm - Step 8a verification (FastAPI serving layer)")
    print("=" * 70)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nStarting uvicorn as a real subprocess (server log: {LOG_PATH})...")
    log_file = LOG_PATH.open("w", encoding="utf-8")
    proc = start_server(log_file)

    try:
        print(f"Waiting up to {HEALTH_TIMEOUT_S}s for /health (this includes the full model load)...")
        health = wait_for_health(HEALTH_TIMEOUT_S)

        if health is None:
            record("Server became healthy in time", False, f"see {LOG_PATH} for server output")
            return 1
        record("Server responded on /health", True, str(health))

        if not health.get("model_loaded"):
            record("Model loaded", False, f"server is in degraded mode - see {LOG_PATH}")
            return 1
        record("Model loaded", True, f"adapter_loaded={health.get('adapter_loaded')}, gpu_available={health.get('gpu_available')}")

        print("\nSending a real prompt to /generate...")
        status, body = post_json(
            "/generate",
            {"prompt": "In one sentence, what does the heat equation describe?", "max_new_tokens": 60},
        )
        record(
            "POST /generate returns 200 with non-empty text",
            status == 200 and bool(body.get("generated_text", "").strip()),
            f"status={status}, generated_text={body.get('generated_text', '')[:100]!r}",
        )

        print("\nConfirming request validation is enforced by the live server (empty prompt)...")
        status, body = post_json("/generate", {"prompt": "", "max_new_tokens": 10})
        record("Empty prompt rejected with 422", status == 422, f"status={status}")

    finally:
        print("\nShutting down the server subprocess...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()

    print("=" * 70)
    n_total = len(results)
    n_passed = sum(1 for _, ok, _ in results if ok)
    print(f"{n_passed}/{n_total} checks passed")
    print("=" * 70)

    failed = [label for label, ok, _ in results if not ok]
    if failed:
        print("Failed checks:")
        for label in failed:
            print(f"  - {label}")
        print(f"\nFull server output is in {LOG_PATH}")
        return 1

    print("All Step 8a checks passed. Ready for Step 8b (Docker).")
    print("\nTry it yourself with curl while iterating later (PowerShell - note curl.exe,")
    print("not the curl alias, and single-quoted JSON so PowerShell passes it through as-is):")
    print(f"  curl.exe {BASE_URL}/health")
    print(
        f"  curl.exe -X POST {BASE_URL}/generate -H \"Content-Type: application/json\" "
        f"-d '{{\"prompt\": \"What is E=mc^2?\", \"max_new_tokens\": 100}}'"
    )
    print("(the server above has already been shut down - start it again with:")
    print("  uvicorn src.api.main:app --host 0.0.0.0 --port 8000)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
