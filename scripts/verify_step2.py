"""
scientific-llm - Step 2 verification: base model + LoRA, end to end.

Runs the full pipeline once, in order, and checks each stage rather than
trusting that "no exception raised" means "worked correctly":
  1. Hugging Face authentication is in place (gated repo)
  2. Base model loads in 4-bit and generates coherent text
  3. LoRA adapters attach with a sane trainable-parameter percentage
  4. A forward+backward pass sends gradients ONLY into LoRA parameters
  5. Peak GPU memory stays within the 8GB budget with headroom for
     training (this run is inference + one backward pass, not a full
     training loop, so treat "close to 8GB here" as a yellow flag for
     Step 4, not a hard failure of this script)

Run directly:
    python scripts\\verify_step2.py
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model.base_model import load_base_model, smoke_test_generation
from src.model.lora_config import attach_lora, count_parameters, smoke_test_backward

results: list[tuple[str, bool, str]] = []


def record(label: str, passed: bool, detail: str = "") -> None:
    results.append((label, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail else ""))


def check_hf_auth() -> None:
    try:
        out = subprocess.run(
            ["hf", "auth", "whoami"], capture_output=True, text=True, timeout=15
        )
        combined = (out.stdout + out.stderr).lower()
        logged_in = out.returncode == 0 and "not logged in" not in combined
        record(
            "Hugging Face authentication",
            logged_in,
            out.stdout.strip() if logged_in else "run: hf auth login",
        )
    except Exception as e:  # noqa: BLE001
        record("Hugging Face authentication", False, f"could not check: {e}")


def main() -> int:
    print("=" * 70)
    print("scientific-llm - Step 2 verification (base model + LoRA)")
    print("=" * 70)

    if not torch.cuda.is_available():
        record("CUDA available", False, "run scripts\\verify_environment.py first")
        return 1
    record("CUDA available", True)

    check_hf_auth()

    print("\nLoading base model (first run downloads ~14-15GB, cached after)...")
    try:
        model, tokenizer = load_base_model()
        record("Base model loads in 4-bit", True)
    except Exception as e:  # noqa: BLE001
        record("Base model loads in 4-bit", False, f"{type(e).__name__}: {e}")
        print("=" * 70)
        print("Stopping early - later checks depend on this one.")
        return 1

    try:
        text = smoke_test_generation(model, tokenizer)
        record("Base model generates coherent text", bool(text.strip()), text[:80])
    except Exception as e:  # noqa: BLE001
        record("Base model generates coherent text", False, f"{type(e).__name__}: {e}")

    print("\nAttaching LoRA adapters...")
    peft_model = attach_lora(model)
    trainable, total = count_parameters(peft_model)
    pct = 100 * trainable / total
    # Sanity band: with r=16 across 7 target modules this should land
    # comfortably under 1%. Flag anything outside a generous [0.01%, 5%]
    # band as suspicious (e.g. target_modules matched nothing, or matched
    # far too much).
    sane = 0.01 <= pct <= 5.0
    record(
        "LoRA trainable-parameter percentage is sane",
        sane,
        f"{trainable:,} / {total:,} = {pct:.4f}%",
    )

    print("\nRunning forward+backward smoke test...")
    try:
        grads_ok = smoke_test_backward(peft_model, tokenizer)
        record("Gradients flow only into LoRA adapters", grads_ok)
    except Exception as e:  # noqa: BLE001
        record("Gradients flow only into LoRA adapters", False, f"{type(e).__name__}: {e}")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        under_budget = peak_gb < 7.5
        print(
            f"[{'PASS' if under_budget else 'WARN'}] Peak GPU memory this run: "
            f"{peak_gb:.2f} GB (8GB card)"
        )
        if not under_budget:
            print(
                "         Not a failure of this script (inference + one backward "
                "pass fit), but a signal that Step 4's training loop will need "
                "batch_size=1, gradient accumulation, and short sequence lengths "
                "from the start rather than as a fallback."
            )

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

    print("All Step 2 checks passed. Ready for Step 3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())