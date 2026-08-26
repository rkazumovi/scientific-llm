"""
scientific-llm - Step 2b: attach LoRA adapters to the 4-bit base model.

This is the "LoRA" half of QLoRA. The base model loaded by base_model.py
stays frozen and quantized; this file adds small trainable low-rank
"adapter" matrices next to a chosen set of linear layers, and only those
adapters ever receive gradients. See notebooks/lora_math.ipynb for the
full derivation of why this works and how the parameter counts below are
computed - this file is the implementation, that notebook is the math.

Run directly:
    python src\\model\\lora_config.py
(loads the base model, attaches LoRA, prints trainable-parameter stats,
runs a tiny forward+backward pass to prove gradients flow only into the
adapters, and reports peak GPU memory.)
"""

import sys
from pathlib import Path

# Allow `python src\model\lora_config.py` to import src.model.base_model
# regardless of the working directory it was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.model.base_model import load_base_model, report_gpu_memory

# Mistral's decoder blocks expose these 7 linear projections per layer.
# Targeting all 7 (not just q_proj/v_proj) is what the QLoRA paper found
# closes most of the gap to full fine-tuning, at a modest extra adapter
# parameter cost (see notebooks/lora_math.ipynb for the count).
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# rank r=16, alpha=32 (alpha/r = 2.0 scaling) is the QLoRA paper's default
# and a reasonable starting point for a 7B model on 8GB VRAM. Step 4
# (training) is where we would sweep this if results need tuning.
DEFAULT_R = 16
DEFAULT_ALPHA = 32
DEFAULT_DROPOUT = 0.05


def build_lora_config(
    r: int = DEFAULT_R,
    alpha: int = DEFAULT_ALPHA,
    dropout: float = DEFAULT_DROPOUT,
    target_modules=None,
) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or DEFAULT_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )


def attach_lora(model, lora_config: LoraConfig = None):
    """
    prepare_model_for_kbit_training does three things that matter for a
    quantized base model:
      1. casts LayerNorm / final-norm weights to fp32 (numerical
         stability - these are cheap and should not stay 4-bit)
      2. enables gradient checkpointing (trades compute for memory -
         needed to fit training, not just inference, in 8GB)
      3. makes the input embeddings require grad so gradient
         checkpointing has something to attach to, even though the
         embeddings themselves are not trained
    get_peft_model then wraps the model, freezing every original weight
    and inserting the trainable LoRA matrices from lora_config.
    """
    model = prepare_model_for_kbit_training(model)
    lora_config = lora_config or build_lora_config()
    peft_model = get_peft_model(model, lora_config)
    return peft_model


def count_parameters(peft_model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    return trainable, total


def smoke_test_backward(peft_model, tokenizer) -> bool:
    """Runs one forward + backward pass on a tiny dummy batch and checks
    that gradients landed on LoRA adapter parameters and nowhere else.

    Important subtlety (see notebooks/lora_math.ipynb, Sections 4 and 5):
    peft initializes lora_B to all zeros. Since dL/dlora_A = lora_B^T @ (...),
    that gradient is mathematically ZERO everywhere on this very first
    backward pass - not a bug, a direct consequence of B=0 at init. So we
    only require lora_A's gradient to EXIST (be connected to the
    autograd graph, i.e. not None) - we do not require it to be nonzero.
    lora_B's gradient depends on lora_A (randomly initialized, not zero),
    so lora_B genuinely should receive nonzero signal immediately, and we
    check that strictly.
    """
    peft_model.train()
    text = "The Schrodinger equation describes"
    inputs = tokenizer(text, return_tensors="pt").to(peft_model.device)

    outputs = peft_model(**inputs, labels=inputs["input_ids"])
    outputs.loss.backward()

    lora_a_missing = []
    lora_b_zero = []
    leaked = []
    for name, param in peft_model.named_parameters():
        if not param.requires_grad:
            continue
        has_grad = param.grad is not None
        nonzero = has_grad and torch.any(param.grad != 0).item()
        if "lora_A" in name:
            if not has_grad:
                lora_a_missing.append(name)
        elif "lora_B" in name:
            if not nonzero:
                lora_b_zero.append(name)
        else:
            if nonzero:
                leaked.append(name)

    peft_model.zero_grad()

    if lora_a_missing:
        print(f"  lora_A params with NO gradient at all (should not happen): {lora_a_missing[:3]}")
    if lora_b_zero:
        print(f"  lora_B params with zero/missing gradient (should be nonzero): {lora_b_zero[:3]}")
    if leaked:
        print(f"  Non-LoRA params that unexpectedly received a nonzero gradient: {leaked[:3]}")

    return not lora_a_missing and not lora_b_zero and not leaked


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device visible. Run scripts\\verify_environment.py first.")
        return 1

    try:
        base_model, tokenizer = load_base_model()
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: could not load base model: {type(e).__name__}: {e}")
        return 1

    report_gpu_memory("base model loaded")

    print("\nAttaching LoRA adapters...")
    peft_model = attach_lora(base_model)

    trainable, total = count_parameters(peft_model)
    pct = 100 * trainable / total
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")

    report_gpu_memory("after LoRA attach")

    print("\nRunning forward+backward smoke test...")
    try:
        grads_ok = smoke_test_backward(peft_model, tokenizer)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: forward/backward raised {type(e).__name__}: {e}")
        return 1

    report_gpu_memory("after backward pass (peak-ish)")

    if not grads_ok:
        print("FAIL: gradients did not land exclusively on LoRA adapter parameters.")
        return 1

    print("PASS: LoRA attached, gradients flow only into adapters, base model untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())