"""
scientific-llm - Step 8e: Gradio UI in front of the Step 8a FastAPI
serving layer.

This does NOT load the model itself - it is a thin client that talks to
an already-running API (Step 8a/8b/8c, whichever you have up) over the
exact same HTTP contract curl already uses (/health, /generate). That
keeps the model loaded in exactly one place: running the UI never means
a second multi-gigabyte load competing for the same GPU, and the UI
inherits every guarantee already proven for the API itself (graceful
degraded mode, request validation, /metrics).

Same two-entry-point split as src/api/main.py, for the same reason:
  - build_interface(generate_fn, health_fn) builds the actual Gradio
    Blocks UI around already-resolved callables. It makes no HTTP calls
    itself, which is what makes it directly testable with fake
    functions - no running API, no GPU, no network needed - the same
    idea as create_app()'s TestClient-testable route logic.
  - create_production_interface(api_base_url) is the real entry point:
    wires real HTTP calls (urllib, matching every verify_stepN.py
    script's own choice - no new HTTP client dependency for this) to
    build_interface().

Run directly (with a Step 8a/8b/8c API already running):
    python src\\ui\\gradio_app.py
(defaults to http://localhost:8000 for the API - set API_BASE_URL to
point at a different one, e.g. Step 8b's Docker container or Step 8c's
port-forwarded minikube Service, both of which already publish to that
same port.)
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_UI_PORT = 7860


def make_generate_callback(generate_fn: Callable[[str, int], str]) -> Callable[[str, int], str]:
    """Returns the actual Gradio click-handler, closed over generate_fn.
    Kept as its own top-level factory (rather than inline inside
    build_interface, invisible to anything outside it) specifically so
    it can be unit-tested directly with a fake generate_fn - no Gradio
    server, no event system, just a plain function call."""

    def on_generate(prompt: str, max_new_tokens: int) -> str:
        if not prompt or not prompt.strip():
            return "Please enter a prompt."
        try:
            return generate_fn(prompt, int(max_new_tokens))
        except Exception as e:  # noqa: BLE001 - shown to the user, not raised into Gradio's own error UI
            return f"Error: {type(e).__name__}: {e}"

    return on_generate


def make_health_callback(health_fn: Callable[[], dict]) -> Callable[[], str]:
    def on_check_health() -> str:
        try:
            health = health_fn()
        except Exception as e:  # noqa: BLE001 - API unreachable, wrong port, etc. - shown, not raised
            return f"Could not reach the API: {type(e).__name__}: {e}"
        return (
            f"status: {health.get('status')} | model_loaded: {health.get('model_loaded')} | "
            f"adapter_loaded: {health.get('adapter_loaded')} | gpu_available: {health.get('gpu_available')}"
        )

    return on_check_health


def build_interface(generate_fn: Callable[[str, int], str], health_fn: Callable[[], dict]) -> gr.Blocks:
    on_generate = make_generate_callback(generate_fn)
    on_check_health = make_health_callback(health_fn)

    with gr.Blocks(title="scientific-llm") as demo:
        gr.Markdown(
            "# scientific-llm\n"
            "QLoRA-fine-tuned Mistral-7B-Instruct for scientific (physics/math) Q&A. "
            "This UI talks to the Step 8a API over HTTP - it does not load the model itself."
        )

        with gr.Row():
            health_button = gr.Button("Check API health")
            health_output = gr.Textbox(label="API health", interactive=False)
        health_button.click(fn=on_check_health, outputs=health_output, api_name="check_health")

        prompt_input = gr.Textbox(
            label="Prompt", lines=3, placeholder="What does the heat equation describe?"
        )
        max_tokens_slider = gr.Slider(minimum=1, maximum=1024, value=200, step=1, label="Max new tokens")
        generate_button = gr.Button("Generate")
        generate_output = gr.Textbox(label="Generated text", lines=8, interactive=False)
        generate_button.click(
            fn=on_generate,
            inputs=[prompt_input, max_tokens_slider],
            outputs=generate_output,
            api_name="generate",
        )

    return demo


def _call_health_over_http(api_base_url: str) -> dict:
    with urllib.request.urlopen(f"{api_base_url}/health", timeout=10) as resp:
        return json.loads(resp.read())


def _call_generate_over_http(api_base_url: str, prompt: str, max_new_tokens: int) -> str:
    req = urllib.request.Request(
        f"{api_base_url}/generate",
        data=json.dumps({"prompt": prompt, "max_new_tokens": max_new_tokens}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
            return body["generated_text"]
    except urllib.error.HTTPError as e:
        detail_body = e.read()
        try:
            detail = json.loads(detail_body).get("detail", detail_body.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            detail = detail_body.decode("utf-8", "replace")
        raise RuntimeError(f"API returned {e.code}: {detail}") from e


def create_production_interface(api_base_url: str | None = None) -> gr.Blocks:
    resolved_api_base_url = api_base_url or os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL)
    return build_interface(
        generate_fn=lambda prompt, max_new_tokens: _call_generate_over_http(
            resolved_api_base_url, prompt, max_new_tokens
        ),
        health_fn=lambda: _call_health_over_http(resolved_api_base_url),
    )


if __name__ == "__main__":
    demo = create_production_interface()
    port = int(os.environ.get("UI_PORT", DEFAULT_UI_PORT))
    demo.launch(server_name="0.0.0.0", server_port=port)
