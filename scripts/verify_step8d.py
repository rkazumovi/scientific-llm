"""
scientific-llm - Step 8d verification: Prometheus metrics on top of the
Step 8a serving layer, over real HTTP against a real subprocess server -
same harness shape as verify_step8a.py (start uvicorn for real, poll
/health, hit endpoints, clean teardown), extended to also drive
/metrics and confirm the numbers in it actually reflect the requests
this script itself just made - not just that the endpoint responds.

Run directly:
    python scripts\\verify_step8d.py
(this waits for the full model load, same as verify_step8a.py - give it
several minutes before assuming something is stuck.)
"""

import json
import os
import re
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
LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "verify_step8d_server.log"

results: list[tuple[str, bool, str]] = []


def record(label: str, passed: bool, detail: str = "") -> None:
    results.append((label, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail else ""))


def start_server(log_file) -> subprocess.Popen:
    # PYTHONUNBUFFERED=1 (equivalent to python -u): with stdout
    # redirected to a real file instead of a terminal, Python normally
    # switches to full block buffering, so print() output - including
    # main.py's own diagnostic messages during model load - can sit in
    # an internal buffer and never reach the log file if this process
    # gets terminated before that buffer naturally flushes. That is a
    # real gap in exactly the situation this log file exists for:
    # diagnosing a failure. Forcing unbuffered output here means the
    # log always reflects what actually happened, not just what
    # happened to be flushed before shutdown.
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", HOST, "--port", str(PORT)],
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
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


def get_metrics_text() -> tuple[int, str]:
    with urllib.request.urlopen(f"{BASE_URL}/metrics", timeout=10) as resp:
        return resp.status, resp.read().decode("utf-8")


def metric_value(text: str, name: str, labels: str = "") -> float | None:
    """Same extraction logic used while building/testing this step - a
    bare (unlabeled) metric renders as `name value`, a labeled one as
    `name{labels} value`. Returns None if not found, rather than
    raising, so callers can produce a clear PASS/FAIL record."""
    if labels:
        pattern = re.escape(name) + r"\{" + re.escape(labels) + r"\}\s+([0-9.eE+-]+)"
    else:
        pattern = r"(?m)^" + re.escape(name) + r"\s+([0-9.eE+-]+)"
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def main() -> int:
    print("=" * 70)
    print("scientific-llm - Step 8d verification (Prometheus metrics)")
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

        # A second /health hit and one real /generate call, plus one
        # validation failure - deliberately chosen so the /metrics
        # assertions below have exact expected counts to check against
        # (2 GET /health 200s: the one wait_for_health ended on, plus
        # this one; 1 POST /generate 200; 1 POST /generate 422).
        print("\nMaking a second /health call and a real /generate call, plus one validation failure...")
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=10) as resp:
            resp.read()  # drain and close cleanly; the status/body are not needed again here
        status, body = post_json(
            "/generate",
            {"prompt": "In one sentence, what does the heat equation describe?", "max_new_tokens": 60},
        )
        record(
            "POST /generate returns 200 with non-empty text",
            status == 200 and bool(body.get("generated_text", "").strip()),
            f"status={status}, generated_text={body.get('generated_text', '')[:100]!r}",
        )
        status, body = post_json("/generate", {"prompt": "", "max_new_tokens": 10})
        record("Empty prompt rejected with 422", status == 422, f"status={status}")

        print("\nFetching /metrics and confirming it reflects the requests made above...")
        metrics_status, metrics_text = get_metrics_text()
        record(
            "GET /metrics returns 200 with text/plain content",
            metrics_status == 200 and "scientific_llm_requests_total" in metrics_text,
            f"status={metrics_status}",
        )

        health_count = metric_value(metrics_text, "scientific_llm_requests_total", 'endpoint="/health",method="GET",status_code="200"')
        record(
            "GET /health request count is at least 2 in /metrics",
            health_count is not None and health_count >= 2,
            f"observed={health_count}",
        )

        generate_200 = metric_value(metrics_text, "scientific_llm_requests_total", 'endpoint="/generate",method="POST",status_code="200"')
        record(
            "POST /generate 200 is counted exactly once in /metrics",
            generate_200 == 1.0,
            f"observed={generate_200}",
        )

        generate_422 = metric_value(metrics_text, "scientific_llm_requests_total", 'endpoint="/generate",method="POST",status_code="422"')
        record(
            "POST /generate 422 is counted exactly once in /metrics",
            generate_422 == 1.0,
            f"observed={generate_422}",
        )

        model_loaded_gauge = metric_value(metrics_text, "scientific_llm_model_loaded")
        record(
            "scientific_llm_model_loaded gauge reads 1 (matches /health)",
            model_loaded_gauge == 1.0,
            f"observed={model_loaded_gauge}",
        )

        latency_count = metric_value(metrics_text, "scientific_llm_request_duration_seconds_count", 'endpoint="/generate"')
        record(
            "Request latency histogram recorded the /generate call",
            latency_count is not None and latency_count >= 1,
            f"observed={latency_count}",
        )

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

    print("All Step 8d checks passed. Ready for Step 8e (Gradio UI).")
    print("\nTry it yourself with curl while iterating later (PowerShell - note curl.exe,")
    print("not the curl alias):")
    print(f"  curl.exe {BASE_URL}/metrics")
    print("(the server above has already been shut down - start it again with:")
    print("  uvicorn src.api.main:app --host 0.0.0.0 --port 8000)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
