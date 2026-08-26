"""
scientific-llm - Step 5d: MATH / SciQ / ARC-Challenge benchmark harness.

Runs the fine-tuned model against small subsets of three public
benchmarks and scores it:
  - SciQ (allenai/sciq): multiple-choice science questions.
  - ARC-Challenge (ai2_arc, config "ARC-Challenge"): harder multiple-
    choice science questions (the "Challenge" split specifically
    excludes questions a simple retrieval/co-occurrence baseline gets
    right, so it is a meaningfully harder bar than SciQ).
  - MATH (HuggingFaceH4/MATH-500): competition math problems with a
    known final answer, scored via Step 5c's SymPy equivalence checker
    rather than exact string match (so "1/2" and "0.5" both score
    correct).

Honesty about scope, worth reading before trusting a number:
  - Each benchmark's dataset schema was implemented from documented
    field names, not verified against a live download in this
    project's own build/test environment (no Hugging Face Hub network
    access there - see this file's module-level comment history / the
    project README for the general pattern of network-dependent code
    needing to be verified on your machine). load_benchmark_subset()
    therefore wraps the real download in a try/except and falls back to
    a small built-in example set per benchmark on ANY failure (network,
    renamed dataset, changed schema) - the same resilience pattern
    arxiv_loader.py/preprocessor.py already use for the arXiv pipeline.
    If verify_step5.py reports "using built-in fallback examples" for a
    benchmark, that benchmark's real schema needs checking against what
    is actually live on the Hub.
  - Default N per benchmark is intentionally small (a handful of
    examples) because generation with a 7B model, one example at a time,
    on an 8GB card is not fast - this is a spot-check, not a full
    benchmark run. Pass a larger --n for a real evaluation; expect it to
    take a while and log progress rather than running silently.
  - Answer extraction (multiple-choice letter, MATH's \\boxed{...}) is
    regex-based and will occasionally fail to find an answer in a
    poorly-formatted generation. Unparseable generations are counted and
    reported SEPARATELY from accuracy, never silently treated as wrong
    or silently dropped from the denominator - see run_benchmark()'s
    returned "unparseable" count.

Run directly:
    python src\\evaluation\\benchmarks.py
(loads the Step 2 base model and runs 2 examples per benchmark - a quick
smoke test, not a real evaluation number.)
"""

import argparse
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.math_verifier import expressions_equivalent

# ---------------------------------------------------------------------------
# Fallback examples (used only if the real dataset cannot be downloaded).
# Small and hand-picked so the harness is always exercisable offline.
# ---------------------------------------------------------------------------

_SCIQ_FALLBACK = [
    {
        "question": "What force pulls objects toward the center of the Earth?",
        "choices": ["magnetism", "friction", "gravity", "inertia"],
        "answer_index": 2,
    },
    {
        "question": "What is the smallest unit of an element that retains its properties?",
        "choices": ["molecule", "atom", "electron", "compound"],
        "answer_index": 1,
    },
]

_ARC_CHALLENGE_FALLBACK = [
    {
        "question": "A student wants to increase the rate of a chemical reaction. Which change would most likely accomplish this?",
        "choices": ["lowering the temperature", "increasing the temperature", "removing the catalyst", "decreasing the surface area"],
        "answer_index": 1,
    },
    {
        "question": "Which property of a wave is directly related to its energy?",
        "choices": ["wavelength only", "amplitude", "color", "direction of travel"],
        "answer_index": 1,
    },
]

_MATH_FALLBACK = [
    {"problem": "What is 12 + 7 * 3?", "answer": "33"},
    {"problem": "Simplify: (x+1)(x-1)", "answer": "x**2 - 1"},
]

_LETTERS = ["A", "B", "C", "D", "E", "F"]


def _to_standard_mc(question: str, choices: list[str], answer_index: int) -> dict:
    return {"question": question, "choices": list(choices), "answer_index": answer_index}


def _sciq_to_standard(example: dict) -> dict:
    correct = example["correct_answer"]
    distractors = [example["distractor1"], example["distractor2"], example["distractor3"]]
    choices = distractors + [correct]
    # Deterministic ordering (not shuffled) - keeps this reproducible
    # rather than depending on an unseeded random draw.
    return _to_standard_mc(example["question"], choices, answer_index=len(choices) - 1)


def _arc_to_standard(example: dict) -> dict:
    texts = example["choices"]["text"]
    labels = example["choices"]["label"]
    answer_key = example["answerKey"]
    if answer_key not in labels:
        raise ValueError(f"answerKey {answer_key!r} not found among choice labels {labels!r}")
    return _to_standard_mc(example["question"], texts, answer_index=labels.index(answer_key))


def _math_to_standard(example: dict) -> dict:
    # HuggingFaceH4/MATH-500 provides a plain final "answer" field
    # alongside the full worked "solution" - using it directly avoids
    # re-implementing \boxed{} extraction on the REFERENCE side (the
    # model's own generation still needs extract_boxed_answer below).
    return {"problem": example["problem"], "answer": str(example["answer"])}


_BENCHMARK_SPECS = {
    "sciq": {
        "hf_dataset": "sciq",
        "hf_config": None,
        "hf_split": "test",
        "to_standard": _sciq_to_standard,
        "fallback": _SCIQ_FALLBACK,
        "kind": "multiple_choice",
    },
    "arc_challenge": {
        "hf_dataset": "ai2_arc",
        "hf_config": "ARC-Challenge",
        "hf_split": "test",
        "to_standard": _arc_to_standard,
        "fallback": _ARC_CHALLENGE_FALLBACK,
        "kind": "multiple_choice",
    },
    "math": {
        "hf_dataset": "HuggingFaceH4/MATH-500",
        "hf_config": None,
        "hf_split": "test",
        "to_standard": _math_to_standard,
        "fallback": _MATH_FALLBACK,
        "kind": "math",
    },
}


def load_benchmark_subset(name: str, n: int = 5) -> tuple[list[dict], bool]:
    """Returns (examples, used_fallback). Tries the real Hugging Face
    dataset first; on ANY exception (network, renamed dataset, changed
    schema), falls back to the small built-in example set rather than
    crashing the whole evaluation run."""
    spec = _BENCHMARK_SPECS[name]
    try:
        from datasets import load_dataset

        ds = load_dataset(spec["hf_dataset"], spec["hf_config"], split=spec["hf_split"])
        examples = []
        for i in range(min(n, len(ds))):
            examples.append(spec["to_standard"](ds[i]))
        if not examples:
            raise ValueError("dataset loaded but yielded zero usable examples")
        return examples, False
    except Exception:  # noqa: BLE001 - deliberately broad: any failure falls back
        fallback = spec["fallback"][:n] if n <= len(spec["fallback"]) else spec["fallback"]
        return fallback, True


# ---------------------------------------------------------------------------
# Prompting and answer extraction
# ---------------------------------------------------------------------------


def format_multiple_choice_prompt(question: str, choices: list[str]) -> str:
    options = "\n".join(f"{_LETTERS[i]}. {choice}" for i, choice in enumerate(choices))
    return (
        "### Instruction:\nAnswer the following multiple-choice question. "
        "Respond with only the letter of the correct option.\n\n"
        f"### Input:\n{question}\n{options}\n\n"
        "### Response:\n"
    )


def extract_choice_letter(generated_text: str, num_choices: int) -> str | None:
    valid_letters = _LETTERS[:num_choices]
    match = re.search(r"\b([" + "".join(valid_letters) + r"])\b", generated_text.strip().upper())
    return match.group(1) if match else None


def format_math_prompt(problem: str) -> str:
    return (
        "### Instruction:\nSolve the following problem. Put only your final answer "
        "inside \\boxed{}.\n\n"
        f"### Input:\n{problem}\n\n"
        "### Response:\n"
    )


def extract_boxed_answer(generated_text: str) -> str | None:
    """Finds the content of the first \\boxed{...}, using a balanced-
    brace scan rather than a single regex - \\boxed{\\frac{1}{2}} has
    nested braces that a naive `\\{([^}]*)\\}` pattern would truncate
    at the first inner '}'."""
    marker = "\\boxed{"
    start = generated_text.find(marker)
    if start == -1:
        return None
    i = start + len(marker)
    depth = 1
    content_start = i
    while i < len(generated_text) and depth > 0:
        if generated_text[i] == "{":
            depth += 1
        elif generated_text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None  # unbalanced - no clean closing brace found
    return generated_text[content_start : i - 1].strip()


# ---------------------------------------------------------------------------
# Running a benchmark against a real model
# ---------------------------------------------------------------------------


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Only the newly generated continuation, not the echoed prompt.
    return full_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]


def run_benchmark(model, tokenizer, name: str, n: int = 5) -> dict:
    spec = _BENCHMARK_SPECS[name]
    examples, used_fallback = load_benchmark_subset(name, n)

    correct = 0
    unparseable = 0
    details = []

    for example in examples:
        if spec["kind"] == "multiple_choice":
            prompt = format_multiple_choice_prompt(example["question"], example["choices"])
            generated = _generate(model, tokenizer, prompt, max_new_tokens=8)
            predicted = extract_choice_letter(generated, len(example["choices"]))
            correct_letter = _LETTERS[example["answer_index"]]
            is_correct = predicted == correct_letter
            if predicted is None:
                unparseable += 1
            elif is_correct:
                correct += 1
            details.append({"predicted": predicted, "correct": correct_letter, "generated": generated})
        else:  # "math"
            prompt = format_math_prompt(example["problem"])
            generated = _generate(model, tokenizer, prompt, max_new_tokens=128)
            predicted = extract_boxed_answer(generated)
            if predicted is None:
                unparseable += 1
            else:
                equivalence = expressions_equivalent(predicted, example["answer"])
                if equivalence is None:
                    unparseable += 1  # SymPy could not parse one side - not scoreable
                elif equivalence:
                    correct += 1
            details.append({"predicted": predicted, "correct": example["answer"], "generated": generated})

    scored = len(examples) - unparseable
    accuracy = correct / scored if scored else None
    return {
        "benchmark": name,
        "n": len(examples),
        "used_fallback": used_fallback,
        "correct": correct,
        "unparseable": unparseable,
        "accuracy": accuracy,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2, help="Examples per benchmark (default: 2, a quick smoke test)")
    args = parser.parse_args()

    try:
        from src.model.base_model import load_base_model

        print("Loading base model (cached from Step 2, should be quick)...")
        model, tokenizer = load_base_model()

        all_ok = True
        for name in _BENCHMARK_SPECS:
            print(f"\nRunning benchmark: {name} (n={args.n})...")
            result = run_benchmark(model, tokenizer, name, n=args.n)
            fallback_note = " (using built-in fallback examples - could not download real dataset)" if result["used_fallback"] else ""
            print(
                f"  {name}: {result['correct']}/{result['n'] - result['unparseable']} scored correct, "
                f"{result['unparseable']} unparseable{fallback_note}"
            )
            if result["accuracy"] is None:
                print(f"  [WARN] {name}: no scoreable examples (all unparseable)")
                all_ok = False

        if not all_ok:
            print("\nFAIL: at least one benchmark produced zero scoreable examples.")
            return 1

        print("\nPASS: all benchmarks ran and produced at least one scoreable example.")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
