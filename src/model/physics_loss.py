"""
scientific-llm - Step 4a: physics-consistency loss.

The project's "novel contribution" extending PINN-style physics
constraints to LLM fine-tuning. A PINN adds a PDE-residual term to its
loss, directly penalizing candidate solutions that violate a known
physical law - the residual is continuous and differentiable, so it
plugs straight into backprop.

A causal language model has no such residual: its output is a sequence
of discrete tokens, not a continuous field. What IS differentiable is the
model's own log-probability of generating one sequence versus another.
So instead of a residual, this file builds an EQUATION-PREFERENCE loss:
given a real equation from the corpus (extracted by preprocessor.py) and
a deliberately CORRUPTED version of it (a flipped sign, a changed
exponent - the kind of error that makes an equation physically or
mathematically wrong while staying superficially plausible), we compute
the model's sequence log-probability of each completion and add a margin
loss that pushes P(correct) above P(corrupted):

    L_physics = max(0, margin - [logP(correct) - logP(corrupted)])

That is the same "prefer the physically valid one" principle a PINN
residual enforces, expressed as a log-likelihood margin instead of a PDE
residual, because the margin is the differentiable quantity actually
available during language model training.

Honesty about scope: this trains the model's PREFERENCE between a
correct and a corrupted equation - one additional training signal
alongside ordinary cross-entropy - not a guarantee of full physical
correctness for everything the model generates. Step 5's separate SymPy
verification layer checks actual correctness at evaluation time, after
generation; this file is a training-time signal, not a checker.

Run directly:
    python src\\model\\physics_loss.py
(demonstrates the corruption logic on a few example equations with no
model needed, then loads the Step 2 base model + LoRA to demonstrate the
full loss computing and backpropagating.)
"""

import re
import sys
import traceback

import torch


def corrupt_equation(equation: str) -> str | None:
    """
    Applies one simple, targeted corruption to a real equation string,
    chosen to be the kind of error that actually makes it wrong rather
    than merely different:
      1. Sign flip: the first top-level + or - not at position 0 (so a
         leading unary minus is left alone) becomes the other sign.
         (e.g. u_t = a u_xx + f becomes u_t = a u_xx - f, which does not
         just look different - a sign-flipped source term changes the
         physics, a textbook example of what a sign error actually
         breaks, not an arbitrary edit.)
      2. Exponent flip: if there is no top-level sign to flip, change the
         first ^N or **N exponent to a different small integer.
      3. Subscript swap: PDEs like the heat equation (u_t = alpha u_xx)
         often have neither of the above. If the equation contains two
         DIFFERENT derivative subscripts (e.g. _t and _{xx}), swap them -
         confusing which variable something is differentiated with
         respect to is itself a genuine, common physics/math error, and
         this is exactly the case most relevant to this project's PDE
         corpus.
    Returns None if none of the three patterns applies - the caller skips
    this equation rather than inventing a fake error to fill the gap.
    """
    sign_match = re.search(r"(?<!^)(?<![eE])([+-])", equation)
    if sign_match:
        pos = sign_match.start()
        flipped = "-" if equation[pos] == "+" else "+"
        return equation[:pos] + flipped + equation[pos + 1 :]

    exp_match = re.search(r"(\^|\*\*)\{?(\d+)\}?", equation)
    if exp_match:
        old_exp = exp_match.group(2)
        new_exp = str(int(old_exp) + 1)
        start, end = exp_match.span(2)
        return equation[:start] + new_exp + equation[end:]

    subscript_pattern = re.compile(r"_(\{[^}]+\}|\w+)")
    subscripts = subscript_pattern.findall(equation)
    unique_subscripts = list(dict.fromkeys(subscripts))  # order-preserving dedupe
    if len(unique_subscripts) >= 2:
        a, b = unique_subscripts[0], unique_subscripts[1]

        def _swap(m: re.Match) -> str:
            label = m.group(1)
            if label == a:
                return "_" + b
            if label == b:
                return "_" + a
            return m.group(0)

        corrupted = subscript_pattern.sub(_swap, equation)
        if corrupted != equation:
            return corrupted

    return None


EQUATION_PROMPT_TEMPLATE = (
    "### Instruction:\nComplete the following equation.\n\n"
    "### Input:\nEquation (left-hand side): {lhs}\n\n"
    "### Response:\n"
)


def split_equation(equation: str) -> tuple[str, str]:
    """Splits on the first '=' into (left-hand side for the prompt, full
    equation for the completion). No '=' present is a degenerate but
    harmless case - the whole string is used as the "left-hand side"
    prompt too, and corrupt_equation still gives two different
    completions to compare."""
    if "=" in equation:
        lhs = equation.split("=", 1)[0].strip()
        return lhs, equation
    return equation, equation


def sequence_logprob(model, tokenizer, prompt: str, completion: str) -> torch.Tensor:
    """Teacher-forces prompt+completion through the model and returns the
    summed log-probability of just the completion tokens, as a
    differentiable scalar gradient-connected to the model's LoRA
    parameters.

    Note: tokenizing `prompt` alone and `prompt + completion` together
    can occasionally tokenize the seam slightly differently (a known,
    widely-accepted imprecision in this pattern - the same one used
    throughout preference-tuning methods like DPO). That is acceptable
    here: the signal only needs to be directionally correct and
    differentiable, not exact to the token.
    """
    full_text = prompt + completion
    full_ids = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = prompt_ids["input_ids"].shape[1]

    outputs = model(**full_ids)
    logits = outputs.logits[:, :-1, :]  # position t predicts token t+1
    targets = full_ids["input_ids"][:, 1:]

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    completion_start = max(prompt_len - 1, 0)
    return token_log_probs[:, completion_start:].sum()


def physics_consistency_loss(
    model, tokenizer, equations: list[str], margin: float = 1.0
) -> torch.Tensor | None:
    """Samples equations from the pool until one can be corrupted, and
    returns the margin-ranking loss for that pair. Returns None if
    nothing in the pool can be corrupted (caller should skip adding a
    physics term for this step rather than fail the whole training
    step)."""
    import random

    candidates = list(equations)
    random.shuffle(candidates)

    for eq in candidates:
        corrupted = corrupt_equation(eq)
        if corrupted is None or corrupted == eq:
            continue

        lhs, correct_full = split_equation(eq)
        _, corrupted_full = split_equation(corrupted)
        prompt = EQUATION_PROMPT_TEMPLATE.format(lhs=lhs)

        logp_correct = sequence_logprob(model, tokenizer, prompt, correct_full)
        logp_corrupted = sequence_logprob(model, tokenizer, prompt, corrupted_full)

        return torch.clamp(margin - (logp_correct - logp_corrupted), min=0.0)

    return None


def main() -> int:
    print("Corruption logic demo (pure string/regex logic, no model needed):")
    examples = [
        "u_t = \\alpha u_{xx}",
        "E = mc^2",
        "\\nabla^2 \\phi = 4\\pi G \\rho",
        "no equals sign or sign or exponent here",
    ]
    for eq in examples:
        corrupted = corrupt_equation(eq)
        print(f"  {eq!r:45s} -> {corrupted!r}")

    print("\nLoading base model + LoRA to demo the full loss computation...")
    try:
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.model.base_model import load_base_model
        from src.model.lora_config import attach_lora

        base_model, tokenizer = load_base_model()
        peft_model = attach_lora(base_model)
        peft_model.train()

        loss = physics_consistency_loss(
            peft_model, tokenizer, ["u_t = \\alpha u_{xx}", "E = mc^2"]
        )
        if loss is None:
            print("FAIL: no corruptible equation found in the demo pool.")
            return 1

        print(f"physics_consistency_loss = {loss.item():.4f}")
        loss.backward()
        has_grad = any(
            p.grad is not None and torch.any(p.grad != 0).item()
            for p in peft_model.parameters()
            if p.requires_grad
        )
        peft_model.zero_grad()

        if not has_grad:
            print("FAIL: physics loss produced no gradient on any LoRA parameter.")
            return 1

        print("PASS: physics-consistency loss computes and backpropagates into LoRA params.")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
