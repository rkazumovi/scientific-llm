"""
scientific-llm — Step 1 verification script.

Checks, in order:
  1. Python version is 3.13.x
  2. Running inside a virtual environment (not the system Python)
  3. Core packages import and report a version
  4. PyTorch sees the GPU (CUDA available, device name, VRAM)
  5. bitsandbytes can actually quantize a layer to 4-bit and run a forward
     pass on the GPU (not just "does it import" — a real functional check,
     since this is the piece most likely to silently misinstall on Windows)

Run directly:
    python scripts\\verify_environment.py

Exit code 0 = every check passed. Exit code 1 = at least one check failed;
scroll up to the first FAIL line, that's the one to fix first (later
checks often fail as a downstream consequence of an earlier one).
"""

import sys
import importlib

results: list[tuple[str, bool, str]] = []  # (label, passed, detail)


def record(label: str, passed: bool, detail: str = "") -> None:
    results.append((label, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))


def check_python_version() -> None:
    major, minor, micro = sys.version_info[:3]
    ok = (major, minor) == (3, 13)
    record(
        "Python version is 3.13.x",
        ok,
        f"found {major}.{minor}.{micro} at {sys.executable}",
    )


def check_venv() -> None:
    in_venv = sys.prefix != sys.base_prefix
    record(
        "Running inside a virtual environment",
        in_venv,
        sys.prefix if in_venv else "sys.prefix == sys.base_prefix (venv not active?)",
    )


def check_import(module_name: str, label: str | None = None) -> object | None:
    label = label or module_name
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown version")
        record(f"import {label}", True, version)
        return mod
    except Exception as e:  # noqa: BLE001 — we want to catch and report anything
        record(f"import {label}", False, f"{type(e).__name__}: {e}")
        return None


def check_cuda(torch_mod) -> None:
    if torch_mod is None:
        record("CUDA available", False, "torch failed to import")
        return

    available = torch_mod.cuda.is_available()
    if not available:
        record(
            "CUDA available",
            False,
            "torch.cuda.is_available() is False — check the nvidia-smi CUDA "
            "version vs. the CUDA build torch was installed with (see "
            "setup_step1.ps1 comments), and that torch was installed from "
            "the cu124 index-url, not plain PyPI.",
        )
        return

    name = torch_mod.cuda.get_device_name(0)
    total_vram_gb = torch_mod.cuda.get_device_properties(0).total_memory / (1024**3)
    record("CUDA available", True, f"{name}, {total_vram_gb:.1f} GB VRAM")

    if total_vram_gb < 7.5:
        print(
            "         Note: under ~8GB VRAM, 7B-parameter QLoRA fine-tuning is "
            "tight. We'll use batch_size=1 + gradient accumulation, short "
            "sequence lengths, and gradient checkpointing in later steps. "
            "This is expected and workable, not a failure."
        )


def check_bitsandbytes_functional(torch_mod, bnb_mod) -> None:
    label = "bitsandbytes 4-bit quantization functional test"
    if torch_mod is None or bnb_mod is None:
        record(label, False, "torch or bitsandbytes failed to import")
        return
    if not torch_mod.cuda.is_available():
        record(label, False, "skipped — no CUDA device available")
        return

    try:
        import torch
        import torch.nn as nn
        from bitsandbytes.nn import Linear4bit

        torch.manual_seed(0)
        ref = nn.Linear(64, 64)

        quant = Linear4bit(
            64,
            64,
            bias=True,
            compute_dtype=torch.float16,
        )
        quant.load_state_dict(ref.state_dict(), strict=False)
        quant = quant.to("cuda")

        x = torch.randn(4, 64, dtype=torch.float16, device="cuda")
        with torch.no_grad():
            out = quant(x)

        ok = out.shape == (4, 64) and torch.isfinite(out).all().item()
        record(
            label,
            ok,
            f"forward pass output shape {tuple(out.shape)}, all finite: "
            f"{torch.isfinite(out).all().item()}",
        )
    except Exception as e:  # noqa: BLE001
        record(label, False, f"{type(e).__name__}: {e}")


def main() -> int:
    print("=" * 70)
    print("scientific-llm — Step 1 environment verification")
    print("=" * 70)

    check_python_version()
    check_venv()

    torch_mod = check_import("torch")
    check_cuda(torch_mod)

    bnb_mod = check_import("bitsandbytes")
    check_bitsandbytes_functional(torch_mod, bnb_mod)

    check_import("transformers")
    check_import("peft")
    check_import("trl")
    check_import("accelerate")
    check_import("datasets")
    check_import("huggingface_hub")
    check_import("sentencepiece")
    check_import("scipy")
    check_import("sympy")

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

    print("All Step 1 checks passed. Ready for Step 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
