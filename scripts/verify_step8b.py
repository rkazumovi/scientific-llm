"""
scientific-llm - Step 8b verification: the Dockerized API, over real
HTTP against an already-running container.

Unlike verify_step8a.py, this script does NOT start the server itself -
building and running a GPU container is a slower, more manual process
(docker build, docker run --gpus all) that setup_step8b.ps1 drives
directly so you can see the build/run output as it happens. This script
just does the same HTTP checks verify_step8a.py did (poll /health until
the model is loaded, send a real prompt to /generate, confirm
validation), pointed at whatever --base-url you give it (default:
http://localhost:8000, matching the port setup_step8b.ps1 publishes).

Also reused, unmodified in spirit, by Step 8c's setup_step8c.ps1 against
a minikube Service instead of a plain container - that is what
--allow-degraded is for (see its help text below). Step 8b itself never
passes that flag, so its own pass/fail contract is unchanged.

Run directly (with the container already started):
    python scripts\\verify_step8b.py
    python scripts\\verify_step8b.py --base-url http://localhost:8000
    python scripts\\verify_step8b.py --base-url http://localhost:8000 --allow-degraded
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

HEALTH_TIMEOUT_S = 300  # first-run model load inside the container can be slow

results: list[tuple[str, bool, str]] = []


def record(label: str, passed: bool, detail: str = "") -> None:
    results.append((label, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail else ""))


def wait_for_health(base_url: str, timeout_s: int) -> dict | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=3) as resp:
                return json.loads(resp.read())
        except Exception:  # noqa: BLE001 - container/model not ready yet
            time.sleep(2)
    return None


def post_json(base_url: str, path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{base_url}{path}",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help=(
            "Do not fail if model_loaded is false - instead confirm the server is up "
            "and correctly reports a degraded state (health.status='degraded', 503 on "
            "/generate) rather than crashing. Meant for Step 8c: minikube's docker "
            "driver does not support NVIDIA GPU passthrough on a Windows host, so the "
            "model cannot load there even though this exact image already proved it "
            "loads fine with a real GPU in Step 8b. Without this flag (the default, "
            "used by Step 8b itself), a missing model is still a hard failure."
        ),
    )
    args = parser.parse_args()

    print("=" * 70)
    print("scientific-llm - Step 8b verification (Dockerized API)")
    print(f"Target: {args.base_url}")
    print("=" * 70)

    print(f"\nWaiting up to {HEALTH_TIMEOUT_S}s for the container's /health...")
    health = wait_for_health(args.base_url, HEALTH_TIMEOUT_S)

    if health is None:
        record(
            "Server responded on /health",
            False,
            "no response - is it running? try: docker ps / kubectl get pods, and check its logs",
        )
        return 1
    record("Server responded on /health", True, str(health))

    model_loaded = bool(health.get("model_loaded"))
    if not model_loaded and not args.allow_degraded:
        record("Model loaded", False, "check the server logs (docker logs / kubectl logs)")
        return 1
    if not model_loaded:
        record(
            "Model not loaded, but server reports degraded state correctly (--allow-degraded)",
            health.get("status") == "degraded",
            f"status={health.get('status')!r} - expected on a GPU-less target such as minikube on Windows",
        )
    else:
        record(
            "Model loaded",
            True,
            f"adapter_loaded={health.get('adapter_loaded')}, gpu_available={health.get('gpu_available')}",
        )

    if model_loaded and not health.get("gpu_available"):
        print(
            "[WARN] gpu_available is false - the server may be running on CPU. "
            "Check `docker run` used --gpus all and that Docker Desktop's GPU "
            "passthrough is enabled (see README Step 8b)."
        )

    if model_loaded:
        print("\nSending a real prompt to /generate...")
        status, body = post_json(
            args.base_url,
            "/generate",
            {"prompt": "In one sentence, what does the heat equation describe?", "max_new_tokens": 60},
        )
        record(
            "POST /generate returns 200 with non-empty text",
            status == 200 and bool(body.get("generated_text", "").strip()),
            f"status={status}, generated_text={body.get('generated_text', '')[:100]!r}",
        )
    else:
        print("\nModel is not loaded (degraded mode) - confirming /generate fails cleanly with 503, not a crash...")
        status, body = post_json(
            args.base_url,
            "/generate",
            {"prompt": "In one sentence, what does the heat equation describe?", "max_new_tokens": 60},
        )
        record("POST /generate returns 503 while degraded (not a crash)", status == 503, f"status={status}")

    print("\nConfirming request validation is enforced by the server...")
    status, body = post_json(args.base_url, "/generate", {"prompt": "", "max_new_tokens": 10})
    record("Empty prompt rejected with 422", status == 422, f"status={status}")

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
        return 1

    if args.allow_degraded:
        print("All checks passed (degraded-mode run - see setup_step8c.ps1's own summary for what is next).")
    else:
        print("All Step 8b checks passed. Ready for Step 8c (Kubernetes / minikube).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
