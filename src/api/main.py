"""
scientific-llm - Step 8a: FastAPI serving layer. Step 8d adds
Prometheus metrics on top (GET /metrics) - see create_app() below.

Two entry points, deliberately separated:
  - create_app(generate_fn, ...) builds the FastAPI app (routes, request
    validation, status codes) around an ALREADY-RESOLVED generate
    function and health flags. It never loads a model itself. That is
    what makes the app's route logic testable with FastAPI's TestClient
    and a fake generate_fn - no GPU, no downloaded model, no adapter
    checkpoint needed - and it was tested that way (see
    scripts/verify_step8a.py's route-level checks and this project's
    build notes).
  - create_production_app() is the real entry point: loads Step 2's
    base model, auto-detects and attaches a trained Step 4 LoRA adapter
    if one is found, and wires the real generate_fn into create_app().
    This is what `app` (below, module level) actually is, and what
    uvicorn imports when you run `uvicorn src.api.main:app`.

A real but easy-to-get-wrong detail this file gets right: loading a
TRAINED adapter for inference is NOT the same call as Step 2's
attach_lora(). attach_lora() (via peft's get_peft_model) creates FRESH,
randomly-initialized adapter matrices for TRAINING - using it here would
silently discard everything Step 4 trained (lora_B starts at zero, so a
freshly attached adapter has zero effect, indistinguishable from the
untrained base model until you looked closely). Loading a checkpoint's
actual trained weights is peft's PeftModel.from_pretrained(base_model,
adapter_dir) instead - the same call merge.py already uses for exactly
this reason. See _load_model_and_tokenizer() below.

If no adapter checkpoint is found (ADAPTER_DIR, default
outputs/checkpoints/step4_verify - the directory verify_step4.py itself
produces), the API serves the base model alone rather than failing - a
missing adapter is a normal, expected state before Step 4 has been run,
not an error condition.

Run directly:
    python src\\api\\main.py
(starts a real uvicorn server on 0.0.0.0:8000, loading the real model -
see scripts/verify_step8a.py for the automated version of this, and the
project README for manual curl commands to try against it.)
"""

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.schemas import GenerateRequest, GenerateResponse, HealthResponse

DEFAULT_ADAPTER_DIR = "outputs/checkpoints/step4_verify"


def create_app(
    generate_fn: Callable[[str, int], str],
    model_loaded: bool,
    adapter_loaded: bool,
    gpu_available: bool,
) -> FastAPI:
    app = FastAPI(
        title="scientific-llm API",
        description="QLoRA-fine-tuned Mistral-7B-Instruct for scientific (physics/math) Q&A.",
        version="1.0.0",
    )

    # A dedicated CollectorRegistry per app instance, not
    # prometheus_client's implicit global default registry. This
    # matters because create_app() gets called more than once in the
    # same process - every test in this project's suite does that, and
    # so does create_production_app() being re-invoked across a module
    # reload. Registering the same metric name twice against the
    # global default registry raises a hard ValueError; a fresh
    # registry per app instance means each create_app() call is fully
    # independent, with no cross-instance state to collide.
    registry = CollectorRegistry()
    request_count = Counter(
        "scientific_llm_requests_total",
        "Total HTTP requests handled, by endpoint/method/status code.",
        ["endpoint", "method", "status_code"],
        registry=registry,
    )
    request_latency = Histogram(
        "scientific_llm_request_duration_seconds",
        "Request latency in seconds, by endpoint.",
        ["endpoint"],
        registry=registry,
    )
    model_loaded_gauge = Gauge(
        "scientific_llm_model_loaded", "1 if the model is loaded, 0 if serving in degraded mode.", registry=registry
    )
    adapter_loaded_gauge = Gauge(
        "scientific_llm_adapter_loaded", "1 if a trained Step 4 LoRA adapter is loaded, 0 otherwise.", registry=registry
    )
    gpu_available_gauge = Gauge(
        "scientific_llm_gpu_available", "1 if a CUDA GPU is visible to this process, 0 otherwise.", registry=registry
    )
    # These three reflect fixed startup state (set once here, never
    # mutated again) rather than something that changes request to
    # request - matching how model_loaded/adapter_loaded/gpu_available
    # already work everywhere else in this file.
    model_loaded_gauge.set(1 if model_loaded else 0)
    adapter_loaded_gauge.set(1 if adapter_loaded else 0)
    gpu_available_gauge.set(1 if gpu_available else 0)

    @app.middleware("http")
    async def record_metrics(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start
        endpoint = request.url.path
        request_latency.labels(endpoint=endpoint).observe(elapsed)
        request_count.labels(endpoint=endpoint, method=request.method, status_code=str(response.status_code)).inc()
        return response

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if model_loaded else "degraded",
            model_loaded=model_loaded,
            adapter_loaded=adapter_loaded,
            gpu_available=gpu_available,
        )

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        if not model_loaded:
            raise HTTPException(status_code=503, detail="Model is not loaded - check server startup logs.")
        try:
            text = generate_fn(request.prompt, request.max_new_tokens)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Generation failed: {type(e).__name__}: {e}") from e
        return GenerateResponse(prompt=request.prompt, generated_text=text)

    return app


def _build_real_generate_fn(model, tokenizer) -> Callable[[str, int], str]:
    def generate_fn(prompt: str, max_new_tokens: int) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

    return generate_fn


def _load_model_and_tokenizer(adapter_dir: str):
    """Returns (model, tokenizer, adapter_loaded). Loads Step 2's 4-bit
    base model, then attaches a TRAINED adapter's weights via
    PeftModel.from_pretrained if adapter_dir looks like a real saved
    checkpoint (has adapter_config.json) - see module docstring for why
    this is a different call from attach_lora()."""
    from peft import PeftModel

    from src.model.base_model import load_base_model

    base_model, tokenizer = load_base_model()

    adapter_config_path = Path(adapter_dir) / "adapter_config.json"
    if adapter_config_path.exists():
        print(f"Found trained LoRA adapter at {adapter_dir} - loading its weights.")
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        adapter_loaded = True
    else:
        print(f"No trained adapter found at {adapter_dir} - serving the base model as-is.")
        model = base_model
        adapter_loaded = False

    model.eval()
    return model, tokenizer, adapter_loaded


def create_production_app() -> FastAPI:
    """Everything that can fail on a real machine before this project's
    own environment is fully set up - torch not importable yet, no CUDA
    device, a model load error, a corrupt adapter checkpoint - is caught
    here, not just the model load itself. The API should always be able
    to start and answer /health, even in a badly broken environment; a
    crash on import would take that diagnostic endpoint down with it,
    which defeats its purpose."""
    adapter_dir = os.environ.get("ADAPTER_DIR", DEFAULT_ADAPTER_DIR)

    model_loaded = False
    adapter_loaded = False
    gpu_available = False

    def _unavailable(prompt: str, max_new_tokens: int) -> str:
        raise RuntimeError("model failed to load at startup - see server logs")

    generate_fn = _unavailable

    try:
        import torch

        gpu_available = torch.cuda.is_available()
        if not gpu_available:
            # base_model.py's load_base_model() has no CPU path - it
            # always builds a BitsAndBytesConfig(load_in_4bit=True) and
            # hands it straight to from_pretrained(), which needs a
            # working CUDA device to actually quantize the weights.
            # Without this guard, a GPU-less environment (for example
            # Step 8c's minikube deployment - see its own README
            # section for why GPU passthrough is not available there
            # on Windows) would still attempt a real multi-gigabyte
            # load before failing, burning real time and memory for a
            # result that was never going to succeed. Skipping straight
            # to degraded mode here is strictly faster and safer, and
            # changes nothing on a real GPU machine (gpu_available is
            # True there, so this branch is simply never taken).
            print("No CUDA device visible - skipping model load, starting in degraded mode.")
        else:
            model, tokenizer, adapter_loaded = _load_model_and_tokenizer(adapter_dir)
            generate_fn = _build_real_generate_fn(model, tokenizer)
            model_loaded = True
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("WARNING: model failed to load - API starting in degraded mode (503 on /generate, /health still works).")

    return create_app(
        generate_fn, model_loaded=model_loaded, adapter_loaded=adapter_loaded, gpu_available=gpu_available
    )


# Module-level `app` - what `uvicorn src.api.main:app` imports. Building
# it at import time (rather than lazily) means the model load happens
# once at server startup, not on the first request.
app = create_production_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
