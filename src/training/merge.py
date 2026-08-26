"""
scientific-llm - Step 4d: merge the LoRA adapter into the base model.

Computes W' = W + (alpha/r) BA (see notebooks/lora_math.ipynb, Section 7)
and writes out a single, ordinary dense model - no LoRA-related
inference overhead, no base/adapter split, loadable without peft
installed at all.

Runs on CPU, not GPU, and that is deliberate, not a mistake: merging
needs the base model in fp16 (peft's merge_and_unload only merges into
an un-quantized dtype), and 7B params at fp16 is about 14GB - too big
for this project's 8GB card. The merge itself is a memory-bound weight
addition, not a compute-bound operation, so CPU is fine here - slower
(a few minutes), but it does not need the GPU at all.

Run directly:
    python src\\training\\merge.py --adapter-dir outputs\\checkpoints\\demo --output-dir outputs\\merged\\demo
"""

import argparse
import sys
import traceback
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model.base_model import DEFAULT_MODEL


def merge_adapter(
    adapter_dir: str, output_dir: str, base_model_name: str = DEFAULT_MODEL
) -> None:
    # No device_map here on purpose: without one, from_pretrained loads
    # straight to CPU, which is exactly what we want (see module
    # docstring) and avoids relying on device_map accepting a plain "cpu"
    # string, which is not the officially documented case for it.
    print(f"Loading {base_model_name} in fp16 on CPU (needs ~14GB RAM, a few minutes)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print(f"Loading LoRA adapter from {adapter_dir} ...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("Merging (W' = W + (alpha/r) BA) and unloading the adapter wrapper...")
    merged_model = peft_model.merge_and_unload()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {output_dir} ...")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def smoke_test_merged_model(output_dir: str) -> str:
    """Loads the merged model back (fp16, CPU) - a real inference sanity
    check, not just a file-exists check - and runs one short generation."""
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.float16)
    inputs = tokenizer("The heat equation describes", return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", default="outputs/checkpoints/demo")
    parser.add_argument("--output-dir", default="outputs/merged/demo")
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip reloading the merged model to generate (saves time/RAM on a full-size merge)",
    )
    args = parser.parse_args()

    if not Path(args.adapter_dir).exists():
        print(f"FAIL: adapter directory not found: {args.adapter_dir}")
        print("Run src\\training\\trainer.py (or the real training run) first.")
        return 1

    try:
        merge_adapter(args.adapter_dir, args.output_dir, args.base_model)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: merge raised {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    config_path = Path(args.output_dir) / "config.json"
    if not config_path.exists():
        print(f"FAIL: expected {config_path} not found after save_pretrained.")
        return 1
    print(f"\nPASS: merged model saved to {args.output_dir}")

    if not args.skip_smoke_test:
        print("\nReloading merged model to verify it actually generates (fp16, CPU)...")
        try:
            text = smoke_test_merged_model(args.output_dir)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL: merged model failed to load/generate: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
        print(f"Merged model output: {text}")
        if not text.strip():
            print("FAIL: merged model produced empty output.")
            return 1
        print("PASS: merged model loads standalone (no peft needed) and generates.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
