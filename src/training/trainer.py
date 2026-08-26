"""
scientific-llm - Step 4c: QLoRA fine-tuning loop.

A hand-written PyTorch training loop, deliberately not TRL's SFTTrainer
or plain HuggingFace Trainer. Two reasons: TRL is still installed and
fully usable later if you want its extra conveniences (packing,
DeepSpeed, ...) - but its high-level Trainer APIs have gone through
several breaking shape changes across versions, and this project already
hit exactly that kind of surprise once (Step 2's apply_chat_template
change). A raw loop keeps every mechanic - the forward pass, the loss
combination, gradient accumulation - fully visible and easy to verify,
which matters for a project this documentation-focused. The building
blocks (4-bit base model, LoRA adapters, physics loss) from Steps 2 and
4a plug into this loop exactly as they would into any Trainer.

Batch size is fixed at 1 microbatch (not configurable down further - it
is already the minimum), with gradient accumulation to reach a larger
effective batch size, and gradient checkpointing (enabled back in
lora_config.py's prepare_model_for_kbit_training call) - both required to
fit a 7B model's training activations in 8GB VRAM alongside the 4-bit
frozen weights.

Run directly:
    python src\\training\\trainer.py
(runs a short demo training run - a few steps on a tiny built-in dataset
- to prove the loop itself works. scripts\\verify_step4.py runs a
similarly small but fully real end-to-end check against the actual data
pipeline and physics loss.)
"""

import sys
import traceback
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model.physics_loss import physics_consistency_loss
from src.training.callbacks import TrainingLogger


def tokenize_example(tokenizer, text: str, max_length: int = 512) -> dict:
    encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    encoded["labels"] = encoded["input_ids"].clone()
    return encoded


def train(
    peft_model,
    tokenizer,
    train_texts: list[str],
    equations_pool: list[str] = None,
    output_dir: str = "outputs/checkpoints",
    num_steps: int = 5,
    grad_accum_steps: int = 4,
    learning_rate: float = 2e-4,
    physics_weight: float = 0.1,
    max_length: int = 512,
    log_dir: str = "outputs/logs",
    run_name: str = "train",
) -> dict:
    """Runs num_steps OPTIMIZER steps (each made of grad_accum_steps
    forward/backward passes at batch_size=1), combining cross-entropy SFT
    loss with the physics-consistency margin loss (Step 4a) once per
    optimizer step when an equation pool is provided. Saves the LoRA
    adapter to output_dir at the end and returns a summary dict."""
    if not train_texts:
        raise ValueError("train_texts is empty - nothing to train on.")

    equations_pool = equations_pool or []
    device = peft_model.device
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]

    optimizer = AdamW(trainable_params, lr=learning_rate)
    total_microsteps = num_steps * grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_microsteps // 10),
        num_training_steps=total_microsteps,
    )

    logger = TrainingLogger(log_dir=log_dir, run_name=run_name)
    peft_model.train()

    losses = []
    text_idx = 0
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        step_ce_total = 0.0

        for _ in range(grad_accum_steps):
            text = train_texts[text_idx % len(train_texts)]
            text_idx += 1

            batch = tokenize_example(tokenizer, text, max_length=max_length).to(device)
            outputs = peft_model(**batch)
            ce_loss = outputs.loss / grad_accum_steps
            ce_loss.backward()
            step_ce_total += ce_loss.item()

        physics_loss_value = None
        if equations_pool:
            p_loss = physics_consistency_loss(peft_model, tokenizer, equations_pool)
            if p_loss is not None:
                (physics_weight * p_loss).backward()
                physics_loss_value = p_loss.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        physics_contribution = (
            physics_weight * physics_loss_value if physics_loss_value is not None else 0.0
        )
        total_loss = step_ce_total + physics_contribution
        losses.append(total_loss)
        logger.log_step(
            step,
            ce_loss=step_ce_total,
            physics_loss=physics_loss_value,
            total_loss=total_loss,
            lr=scheduler.get_last_lr()[0],
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(output_dir)

    return {
        "steps": num_steps,
        "final_loss": losses[-1],
        "losses": losses,
        "output_dir": output_dir,
    }


def main() -> int:
    try:
        from src.model.base_model import load_base_model
        from src.model.lora_config import attach_lora

        demo_texts = [
            "### Instruction:\nSummarize the heat equation.\n\n### Response:\n"
            "The heat equation u_t = alpha u_xx describes how temperature diffuses over time.",
            "### Instruction:\nWhat does E = mc^2 mean?\n\n### Response:\n"
            "It relates energy and mass through the speed of light squared.",
        ]
        demo_equations = ["u_t = \\alpha u_{xx}", "E = mc^2"]

        base_model, tokenizer = load_base_model()
        peft_model = attach_lora(base_model)

        summary = train(
            peft_model,
            tokenizer,
            train_texts=demo_texts,
            equations_pool=demo_equations,
            output_dir="outputs/checkpoints/demo",
            num_steps=3,
            grad_accum_steps=2,
            run_name="trainer_demo",
        )

        print(
            f"\nDemo training finished: {summary['steps']} steps, "
            f"final loss {summary['final_loss']:.4f}"
        )

        if not all(torch.isfinite(torch.tensor(l)) for l in summary["losses"]):
            print("FAIL: a loss value was not finite (NaN/Inf).")
            return 1

        # adapter_config.json is guaranteed by peft's save_pretrained
        # regardless of whether weights land in .safetensors or .bin -
        # checking for it (rather than a specific weights filename) keeps
        # this check correct across peft versions.
        config_path = Path(summary["output_dir"]) / "adapter_config.json"
        if not config_path.exists():
            print(f"FAIL: expected {config_path} not found - checkpoint save may have failed.")
            return 1

        print("PASS: training loop runs, losses are finite, checkpoint saved.")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
