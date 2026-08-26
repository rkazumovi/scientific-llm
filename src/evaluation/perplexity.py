"""
scientific-llm - Step 5a: perplexity evaluation.

Perplexity is the standard held-out-data metric for a causal language
model: PPL = exp(average negative log-likelihood per token). A model
that on average assigns probability 1.0 to the next real token has
NLL = 0 and PPL = 1 (as low as it gets - "perfectly unsurprised"); a
model no better than guessing uniformly among the tokenizer's V possible
tokens has PPL near V (there is nothing this metric can score below 1).

Convenient in practice because it is exactly the number the model's own
forward pass already produces: HuggingFace's causal LM `forward(...,
labels=...)` returns the mean per-token cross-entropy loss when labels
are supplied, and that IS the average negative log-likelihood per token
by construction (cross-entropy of a softmax against a one-hot target
equals -log P(true token)) - so perplexity is just exp(that loss),
computed per text and macro-averaged, with no extra assumptions needed.

Run directly:
    python src\\evaluation\\perplexity.py
(loads the Step 2 base model - no LoRA weights needed for this file, it
measures the SAME kind of metric you would report after Step 4 training
on both the base and fine-tuned checkpoints to compare them - and
reports perplexity on a couple of small hardcoded example texts.)
"""

import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def compute_perplexity(model, tokenizer, texts: list[str], max_length: int = 512) -> dict:
    """Computes per-text and macro-averaged perplexity over `texts`.
    Uses no_grad + eval mode - this is a read-only evaluation, not a
    training step, so no gradients or dropout should be active."""
    if not texts:
        raise ValueError("texts is empty - nothing to evaluate perplexity on.")

    was_training = model.training
    model.eval()

    per_text_ppl = []
    per_text_loss = []
    device = model.device

    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            labels = encoded["input_ids"].clone()
            outputs = model(**encoded, labels=labels)
            loss = outputs.loss.item()
            per_text_loss.append(loss)
            per_text_ppl.append(float(torch.exp(torch.tensor(loss))))

    if was_training:
        model.train()

    mean_loss = sum(per_text_loss) / len(per_text_loss)
    return {
        "per_text_loss": per_text_loss,
        "per_text_perplexity": per_text_ppl,
        "mean_loss": mean_loss,
        # exp(mean of per-text losses), NOT mean of per-text perplexities:
        # perplexity is exponential in loss, so averaging already-
        # exponentiated numbers overweights the model's worst texts. The
        # standard definition averages loss (equivalently, log-space)
        # first and exponentiates once at the end.
        "perplexity": float(torch.exp(torch.tensor(mean_loss))),
    }


def main() -> int:
    try:
        from src.model.base_model import load_base_model

        demo_texts = [
            "The heat equation u_t = alpha u_xx describes how temperature diffuses over time.",
            "Einstein's mass-energy equivalence is expressed as E = mc^2.",
        ]

        print("Loading base model (cached from Step 2, should be quick)...")
        model, tokenizer = load_base_model()

        print(f"\nComputing perplexity on {len(demo_texts)} demo text(s)...")
        results = compute_perplexity(model, tokenizer, demo_texts)

        for text, loss, ppl in zip(demo_texts, results["per_text_loss"], results["per_text_perplexity"]):
            print(f"  loss={loss:.4f}  ppl={ppl:.2f}  | {text[:60]}...")

        print(f"\nMean loss: {results['mean_loss']:.4f}")
        print(f"Perplexity (macro-average): {results['perplexity']:.2f}")

        if not torch.isfinite(torch.tensor(results["perplexity"])):
            print("FAIL: perplexity is not finite.")
            return 1
        if results["perplexity"] < 1.0:
            print("FAIL: perplexity below 1.0 is not mathematically possible - something is wrong.")
            return 1

        print("PASS: perplexity computed and is a finite value >= 1.0.")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
