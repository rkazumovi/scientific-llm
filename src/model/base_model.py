"""
scientific-llm - Step 2a: load and 4-bit quantize the base model.

This is the QLoRA "Q" half: load Mistral-7B-Instruct-v0.3 with its weights
compressed to 4-bit NF4 (bitsandbytes), so it fits in ~4.5-5GB of VRAM
instead of ~14GB at fp16. The base weights stay frozen and quantized for
the whole project - only LoRA adapters (added in lora_config.py) ever get
trained.

Prerequisite (one-time, manual): Mistral-7B-Instruct-v0.3 is a gated
Hugging Face repo. Before running this file:
  1. Create a Hugging Face account if you do not have one:
     https://huggingface.co/join
  2. Open https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 and
     click "Agree and access repository" (Mistral auto-approves almost
     instantly).
  3. Create a read-access token: https://huggingface.co/settings/tokens
  4. In this venv, run once: hf auth login
     and paste the token when prompted. It is cached under your user
     profile, so you only do this once for the whole project.
     (older huggingface_hub versions call this huggingface-cli login -
     use hf auth login if that command is not found)

First run downloads ~14-15GB of model weights (cached afterward under
%USERPROFILE%\\.cache\\huggingface). Make sure you have that much free
disk space and a stable connection before running.

Run directly:
    python src\\model\\base_model.py
"""

import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Mistral-7B-Instruct-v0.3: ungated-friendly (instant approval), strong
# baseline, well-supported by peft/trl/bitsandbytes. Swap this constant if
# you later want to try LLaMA-3-8B-Instruct instead - everything else in
# this file is model-agnostic.
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def build_quant_config() -> BitsAndBytesConfig:
    """
    The QLoRA quantization recipe (Dettmers et al., 2023):
      - load_in_4bit: store each weight in 4 bits instead of 16/32
      - bnb_4bit_quant_type="nf4": NormalFloat4, a quantization grid
        matched to how pretrained weights are actually distributed
        (roughly Gaussian) rather than a plain linear/int4 grid - this is
        what keeps 4-bit accuracy close to fp16 in practice.
      - bnb_4bit_use_double_quant=True: quantizes the per-block
        quantization CONSTANTS themselves (a second, smaller quantization
        pass), saving roughly another 0.4 bits/parameter on top of NF4.
      - bnb_4bit_compute_dtype=bfloat16: weights are stored at 4-bit but
        de-quantized to bfloat16 on the fly for the actual matmuls
        (your RTX 4060 / Ada Lovelace supports bf16 natively).
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_base_model(model_name: str = DEFAULT_MODEL, use_4bit: bool = True):
    """
    Loads the tokenizer and the 4-bit quantized base model onto the GPU.
    Returns (model, tokenizer).
    """
    print(f"Loading tokenizer for {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # Mistral's tokenizer ships without a pad token; reuse eos_token
        # rather than inventing a new one, so we don't resize embeddings.
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {model_name} in 4-bit (this can take a few minutes on first run)...")
    t0 = time.time()
    quant_config = build_quant_config() if use_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    elapsed = time.time() - t0
    print(f"Loaded in {elapsed:.1f}s")

    return model, tokenizer


def report_gpu_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"[{label}] GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def smoke_test_generation(model, tokenizer) -> str:
    """Runs one short generation to confirm the loaded model actually
    produces coherent text, not just that from_pretrained() didn't crash."""
    prompt = (
        "Explain, in two sentences, the physical significance of the "
        "Einstein field equations."
    )
    messages = [{"role": "user", "content": prompt}]
    # return_dict=True: current transformers versions return a BatchEncoding
    # (dict-like: input_ids, attention_mask, ...) here, not a bare tensor.
    # BatchEncoding.to(device) moves every contained tensor at once, and
    # generate() takes it unpacked as **inputs so attention_mask comes
    # along too, rather than passing the whole dict as if it were input_ids.
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = tokenizer.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    )
    return generated


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device visible. Run scripts\\verify_environment.py first.")
        return 1

    try:
        model, tokenizer = load_base_model()
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: could not load {DEFAULT_MODEL}: {type(e).__name__}: {e}")
        print(
            "If this mentions authentication/gated access, re-check the "
            "Hugging Face login steps in this file's module docstring."
        )
        return 1

    report_gpu_memory("after load")

    print("\nRunning generation smoke test...")
    try:
        text = smoke_test_generation(model, tokenizer)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: generation smoke test raised {type(e).__name__}: {e}")
        print("\nFull traceback (this is the part we actually need to diagnose it):")
        traceback.print_exc()
        return 1

    print(f"\nModel output:\n{text}\n")
    report_gpu_memory("after generation")

    if not text.strip():
        print("FAIL: model produced empty output.")
        return 1

    print("PASS: base model loads in 4-bit and generates coherent text.")
    return 0


if __name__ == "__main__":
    sys.exit(main())