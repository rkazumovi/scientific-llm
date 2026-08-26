"""
scientific-llm - Step 4 verification: physics loss + training loop, end
to end, against real data.

Chains together everything built so far: a small real arXiv fetch (Step
3) provides real training text AND a real equation pool, Step 2's
4-bit base model + LoRA gets trained for a few real optimizer steps
(Step 4's trainer.py) combining cross-entropy with the physics-
consistency loss (Step 4a), and a checkpoint gets saved.

Deliberately NOT included here: merging the checkpoint (merge.py). That
step loads the full fp16 model on CPU and can take several minutes and
~14GB of RAM - a real but separate check, not something this quick GPU
verification should silently make slow. Run it yourself once this
passes:
    python src\\training\\merge.py --adapter-dir outputs\\checkpoints\\step4_verify

Run directly:
    python scripts\\verify_step4.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.arxiv_loader import fetch_papers
from src.data.preprocessor import preprocess_papers
from src.data.instruction_gen import generate_dataset
from src.data.dataset import _build_from_records
from src.model.base_model import load_base_model, report_gpu_memory
from src.model.lora_config import attach_lora, count_parameters
from src.model.physics_loss import physics_consistency_loss
from src.training.trainer import train

results: list[tuple[str, bool, str]] = []
FALLBACK_EQUATIONS = ["u_t = \\alpha u_{xx}", "E = mc^2", "\\nabla^2 \\phi = 4\\pi G \\rho"]


def record(label: str, passed: bool, detail: str = "") -> None:
    results.append((label, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail else ""))


def main() -> int:
    print("=" * 70)
    print("scientific-llm - Step 4 verification (physics loss + training loop)")
    print("=" * 70)

    if not torch.cuda.is_available():
        record("CUDA available", False, "run scripts\\verify_environment.py first")
        return 1
    record("CUDA available", True)

    print("\nFetching a few real papers from arXiv (cat: gr-qc) for real training data...")
    try:
        papers = fetch_papers(categories=["gr-qc"], max_results=5, progress=False)
    except Exception as e:  # noqa: BLE001
        record("arXiv fetch", False, f"{type(e).__name__}: {e}")
        return 1
    record("arXiv fetch", len(papers) > 0, f"{len(papers)} paper(s)")

    cleaned = preprocess_papers(papers)
    pairs = generate_dataset(cleaned)
    record("Instruction pairs generated from real papers", len(pairs) > 0, f"{len(pairs)} pair(s)")
    if not pairs:
        return 1

    dataset_dict = _build_from_records(pairs, val_fraction=0.1, seed=42)
    train_texts = dataset_dict["train"]["text"][:6]  # keep the demo run short

    real_equations = [eq for p in cleaned for eq in p.get("equations", [])]
    equations_pool = real_equations if real_equations else FALLBACK_EQUATIONS
    record(
        "Equation pool for physics loss",
        len(equations_pool) > 0,
        f"{len(real_equations)} from real papers"
        + ("" if real_equations else " (none found - using built-in fallback equations)"),
    )

    print("\nLoading base model + LoRA (cached from Step 2, should be quick)...")
    try:
        base_model, tokenizer = load_base_model()
        peft_model = attach_lora(base_model)
    except Exception as e:  # noqa: BLE001
        record("Base model + LoRA load", False, f"{type(e).__name__}: {e}")
        return 1
    record("Base model + LoRA load", True)
    report_gpu_memory("before training")

    print("\nRunning 3 real optimizer steps (grad_accum_steps=2)...")
    try:
        summary = train(
            peft_model,
            tokenizer,
            train_texts=train_texts,
            equations_pool=equations_pool,
            output_dir="outputs/checkpoints/step4_verify",
            num_steps=3,
            grad_accum_steps=2,
            run_name="step4_verify",
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        record("Training loop runs", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    losses_finite = all(torch.isfinite(torch.tensor(l)) for l in summary["losses"])
    record("Training loop runs, all losses finite", losses_finite, f"losses={summary['losses']}")

    config_path = Path(summary["output_dir"]) / "adapter_config.json"
    record("Checkpoint saved", config_path.exists(), str(config_path))

    report_gpu_memory("after training")
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        under_budget = peak_gb < 7.5
        print(
            f"[{'PASS' if under_budget else 'WARN'}] Peak GPU memory this run: "
            f"{peak_gb:.2f} GB (8GB card)"
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

    print("All Step 4 checks passed. Ready for Step 5.")
    print("\nOptional (slow, CPU, ~14GB RAM) - merge the checkpoint just produced:")
    print("  python src\\training\\merge.py --adapter-dir outputs\\checkpoints\\step4_verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())
