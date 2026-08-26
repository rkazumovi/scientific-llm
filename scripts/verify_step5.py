"""
scientific-llm - Step 5 verification: the evaluation suite, end to end
against real data.

Chains together everything built so far: a small real arXiv fetch (Step
3) provides a real abstract for perplexity/ROUGE and a real equation pool
for the math verifier's cross-check against Step 4a's corrupt_equation;
Step 2's base model generates real completions; Step 5's four evaluation
modules (perplexity, ROUGE, SymPy math verification, MATH/SciQ/ARC-
Challenge benchmarks) each get exercised against that real output.

This step is READ-ONLY evaluation - unlike Step 4, nothing here trains
or saves a checkpoint. It is also the slowest verification script so
far: it runs several real generations against a 7B model on an 8GB card
plus 6 benchmark examples (2 per benchmark), so expect a few minutes,
not seconds.

Run directly:
    python scripts\\verify_step5.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.arxiv_loader import fetch_papers
from src.data.preprocessor import preprocess_papers
from src.model.base_model import load_base_model, report_gpu_memory
from src.model.physics_loss import corrupt_equation
from src.evaluation.perplexity import compute_perplexity
from src.evaluation.rouge_eval import compute_rouge
from src.evaluation.math_verifier import verify_corruption_detected
from src.evaluation.benchmarks import run_benchmark, _BENCHMARK_SPECS

results: list[tuple[str, bool, str]] = []
FALLBACK_EQUATIONS = ["u_t = \\alpha u_{xx}", "E = mc^2", "\\nabla^2 \\phi = 4\\pi G \\rho"]


def record(label: str, passed: bool, detail: str = "") -> None:
    results.append((label, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail else ""))


def main() -> int:
    print("=" * 70)
    print("scientific-llm - Step 5 verification (evaluation suite)")
    print("=" * 70)

    if not torch.cuda.is_available():
        record("CUDA available", False, "run scripts\\verify_environment.py first")
        return 1
    record("CUDA available", True)

    print("\nFetching a few real papers from arXiv (cat: gr-qc) for real evaluation data...")
    try:
        papers = fetch_papers(categories=["gr-qc"], max_results=3, progress=False)
    except Exception as e:  # noqa: BLE001
        record("arXiv fetch", False, f"{type(e).__name__}: {e}")
        return 1
    record("arXiv fetch", len(papers) > 0, f"{len(papers)} paper(s)")

    cleaned = preprocess_papers(papers)
    if not cleaned:
        record("At least one usable paper after preprocessing", False)
        return 1
    record("Papers usable after preprocessing", True, f"{len(cleaned)} paper(s)")

    real_equations = [eq for p in cleaned for eq in p.get("equations", [])]
    equations_pool = real_equations if real_equations else FALLBACK_EQUATIONS
    record(
        "Equation pool for math verifier",
        len(equations_pool) > 0,
        f"{len(real_equations)} from real papers"
        + ("" if real_equations else " (none found - using built-in fallback equations)"),
    )

    print("\nLoading base model (cached from Step 2, should be quick)...")
    try:
        model, tokenizer = load_base_model()
    except Exception as e:  # noqa: BLE001
        record("Base model load", False, f"{type(e).__name__}: {e}")
        return 1
    record("Base model load", True)
    report_gpu_memory("before evaluation")

    # --- 5a: perplexity, on real abstracts -------------------------------
    print("\nComputing perplexity on real abstracts...")
    try:
        eval_texts = [p["clean_abstract"] for p in cleaned][:3]
        ppl_results = compute_perplexity(model, tokenizer, eval_texts)
        ppl_ok = torch.isfinite(torch.tensor(ppl_results["perplexity"])) and ppl_results["perplexity"] >= 1.0
        record("Perplexity finite and >= 1.0", bool(ppl_ok), f"perplexity={ppl_results['perplexity']:.2f}")
    except Exception as e:  # noqa: BLE001
        record("Perplexity computation", False, f"{type(e).__name__}: {e}")
        return 1

    # --- 5b: ROUGE, real model-generated summary vs real abstract --------
    print("\nGenerating a summary and scoring it against the real abstract with ROUGE...")
    try:
        reference_abstract = cleaned[0]["clean_abstract"]
        prompt = (
            "### Instruction:\nSummarize the following abstract in one sentence.\n\n"
            f"### Input:\n{reference_abstract}\n\n### Response:\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.pad_token_id
            )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        generated_summary = full_text[len(prompt_text):].strip()

        rouge_scores = compute_rouge(reference_abstract, generated_summary)
        record(
            "ROUGE computed on real generation",
            len(generated_summary) > 0,
            f"ROUGE-1 F1={rouge_scores['rouge1']['f1']:.3f}, generated: {generated_summary[:80]!r}",
        )
    except Exception as e:  # noqa: BLE001
        record("ROUGE computation", False, f"{type(e).__name__}: {e}")
        return 1

    # --- 5c: math verifier, cross-checking Step 4a's corruptions ---------
    print("\nCross-checking Step 4a's corrupt_equation against real extracted equations...")
    checked_any = False
    all_confirmed = True
    for eq in equations_pool:
        corrupted = corrupt_equation(eq)
        if corrupted is None:
            continue
        confirmed = verify_corruption_detected(eq, corrupted)
        if confirmed is None:
            continue  # could not parse this particular equation - skip, not a failure
        checked_any = True
        if not confirmed:
            all_confirmed = False
            print(f"  [FAIL] {eq!r} -> {corrupted!r} : SymPy did NOT confirm a difference")
        else:
            print(f"  [OK] {eq!r} -> {corrupted!r} : confirmed different")
    record(
        "Math verifier agrees with corrupt_equation on at least one real equation",
        checked_any and all_confirmed,
        f"checked {sum(1 for eq in equations_pool if corrupt_equation(eq))} corruptible equation(s)"
        if checked_any
        else "no equation in the pool was both corruptible and SymPy-parseable - inconclusive, not a failure",
    )

    # --- 5d: benchmarks, small real spot-check ----------------------------
    print("\nRunning a small (n=2 per benchmark) real spot-check on MATH/SciQ/ARC-Challenge...")
    try:
        benchmark_ok = True
        for name in _BENCHMARK_SPECS:
            result = run_benchmark(model, tokenizer, name, n=2)
            fallback_note = " (fallback examples - could not download real dataset)" if result["used_fallback"] else ""
            print(
                f"  {name}: {result['correct']} correct / "
                f"{result['n'] - result['unparseable']} scored, "
                f"{result['unparseable']} unparseable{fallback_note}"
            )
            if result["accuracy"] is None:
                benchmark_ok = False
        record("All benchmarks produced at least one scoreable example", benchmark_ok)
    except Exception as e:  # noqa: BLE001
        record("Benchmark harness runs", False, f"{type(e).__name__}: {e}")
        return 1

    report_gpu_memory("after evaluation")

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

    print("All Step 5 checks passed. Ready for Step 6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
