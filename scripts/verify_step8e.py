"""
scientific-llm - Step 8e verification: the Gradio UI in front of the
Step 8a FastAPI serving layer, tested the same way every other 8-series
step has been - over real HTTP, against real subprocesses, not
in-process fakes.

This one drives TWO real servers at once:
  1. `uvicorn src.api.main:app` (the same Step 8a/8d server) on
     http://127.0.0.1:8000 - waits for the full model load, same as
     verify_step8a.py / verify_step8d.py.
  2. `python -m src.ui.gradio_app` (the new Step 8e UI) on
     http://127.0.0.1:7860, pointed at server #1 via the API_BASE_URL
     environment variable (the same variable src/ui/gradio_app.py's
     create_production_interface() reads).

Then it uses `gradio_client.Client` - the Gradio project's own
equivalent of curl, driving the UI's HTTP API exactly the way a real
browser session would - to call the UI's two exposed endpoints
(api_name="/check_health", api_name="/generate") and confirm the full
chain works end to end: browser-facing HTTP -> Gradio server -> HTTP
call to the FastAPI backend -> real model -> back through Gradio -> the
text a user would actually see.

The DI callback logic itself (empty-prompt rejection, backend-error
surfacing) was already proven with 6 unit tests with no server at all,
and the full healthy-path AND degraded-path chains were each proven
end-to-end against throwaway servers before this script was written -
see this project's build notes. What this script checks for real, on
your machine, is the one thing that could not be tested ahead of time:
your real Gradio UI talking to your real API talking to your real
fine-tuned model.

Run directly (needs both `uvicorn` and `gradio` installed - see
requirements-step8a.txt and requirements-step8e.txt):
    python scripts\\verify_step8e.py
(this is slower than most verify scripts - it waits for the full model
load, the same one Step 8a/8d already do, so give it several minutes
before assuming something is stuck.)
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_HOST = "127.0.0.1"
API_PORT = 8000
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"
UI_HOST = "127.0.0.1"
UI_PORT = 7860
UI_BASE_URL = f"http://{UI_HOST}:{UI_PORT}"
HEALTH_TIMEOUT_S = 300  # model load can be slow, especially a cold cache
UI_READY_TIMEOUT_S = 60  # the UI process itself does not load a model - should be fast
API_LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "verify_step8e_api_server.log"
UI_LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "verify_step8e_ui_server.log"

results: list[tuple[str, bool, str]] = []


def record(label: str, passed: bool, detail: str = "") -> None:
    results.append((label, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail else ""))


def start_api_server(log_file) -> subprocess.Popen:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", API_HOST, "--port", str(API_PORT)],
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )


def start_ui_server(log_file) -> subprocess.Popen:
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        API_BASE_URL=API_BASE_URL,
        UI_PORT=str(UI_PORT),
    )
    return subprocess.Popen(
        [sys.executable, "-m", "src.ui.gradio_app"],
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )


def wait_for_api_health(timeout_s: int) -> dict | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{API_BASE_URL}/health", timeout=3) as resp:
                return json.loads(resp.read())
        except Exception:  # noqa: BLE001 - server not up yet, or still loading
            time.sleep(2)
    return None


def wait_for_ui_client(timeout_s: int):
    """Repeatedly try to construct a gradio_client.Client against the UI
    server - this itself does an HTTP handshake against the running
    server's /config endpoint, so it is a real readiness check, not a
    guess. Returns the connected Client, or None on timeout."""
    from gradio_client import Client

    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            return Client(UI_BASE_URL)
        except Exception as e:  # noqa: BLE001 - UI server not up yet
            last_error = e
            time.sleep(1)
    print(f"(UI never became reachable - last error: {last_error})")
    return None


def main() -> int:
    try:
        import gradio_client  # noqa: F401
    except ImportError:
        print("gradio_client is not installed - run: pip install -r requirements-step8e.txt")
        return 1

    print("=" * 70)
    print("scientific-llm - Step 8e verification (Gradio UI)")
    print("=" * 70)

    API_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    UI_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting the Step 8a API server as a real subprocess (log: {API_LOG_PATH})...")
    api_log_file = API_LOG_PATH.open("w", encoding="utf-8")
    api_proc = start_api_server(api_log_file)
    ui_proc = None
    ui_log_file = None

    try:
        print(f"Waiting up to {HEALTH_TIMEOUT_S}s for the API's /health (this includes the full model load)...")
        health = wait_for_api_health(HEALTH_TIMEOUT_S)

        if health is None:
            record("API server became healthy in time", False, f"see {API_LOG_PATH} for server output")
            return 1
        record("API server responded on /health", True, str(health))

        if not health.get("model_loaded"):
            record("Model loaded", False, f"API is in degraded mode - see {API_LOG_PATH}")
            return 1
        record("Model loaded", True, f"adapter_loaded={health.get('adapter_loaded')}, gpu_available={health.get('gpu_available')}")

        print(f"\nStarting the Step 8e Gradio UI as a real subprocess, pointed at {API_BASE_URL} (log: {UI_LOG_PATH})...")
        ui_log_file = UI_LOG_PATH.open("w", encoding="utf-8")
        ui_proc = start_ui_server(ui_log_file)

        print(f"Waiting up to {UI_READY_TIMEOUT_S}s for the UI server to become reachable...")
        client = wait_for_ui_client(UI_READY_TIMEOUT_S)
        if client is None:
            record("UI server became reachable in time", False, f"see {UI_LOG_PATH} for server output")
            return 1
        record("UI server reachable (gradio_client connected over real HTTP)", True, UI_BASE_URL)

        print("\nCalling the UI's /check_health endpoint (UI -> HTTP -> API -> back)...")
        health_text = client.predict(api_name="/check_health")
        record(
            "UI /check_health reports a healthy backend",
            "status: ok" in health_text and "model_loaded: True" in health_text,
            health_text,
        )

        print("\nCalling the UI's /generate endpoint with a real prompt (UI -> HTTP -> API -> real model -> back)...")
        generated_text = client.predict(
            "In one sentence, what does the heat equation describe?", 60, api_name="/generate"
        )
        record(
            "UI /generate returns real non-empty generated text",
            bool(generated_text.strip()) and not generated_text.startswith("Error:"),
            generated_text[:150],
        )

        print("\nConfirming the UI rejects an empty prompt client-side (no backend round trip needed)...")
        empty_result = client.predict("   ", 10, api_name="/generate")
        record(
            "UI /generate rejects an empty/whitespace prompt with a clear message",
            "prompt" in empty_result.lower(),
            empty_result,
        )

    finally:
        print("\nShutting down the UI and API server subprocesses...")
        for proc in (ui_proc, api_proc):
            if proc is None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        api_log_file.close()
        if ui_log_file is not None:
            ui_log_file.close()

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
        print(f"\nFull server output is in {API_LOG_PATH} and {UI_LOG_PATH}")
        return 1

    print("All Step 8e checks passed. Ready for Step 8f (GitHub Actions CI/CD).")
    print("\nTry it yourself afterward - two terminals, both from this project folder:")
    print(f"  uvicorn src.api.main:app --host 0.0.0.0 --port {API_PORT}")
    print(f"  python -m src.ui.gradio_app")
    print(f"Then open http://localhost:{UI_PORT} in a browser.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
