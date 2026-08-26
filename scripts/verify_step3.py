"""
scientific-llm - Step 3 verification: full data pipeline, end to end.

Runs a SMALL real run of the whole pipeline (fetch -> clean -> instruction
pairs -> HuggingFace dataset) against the live arXiv API, not the demo
fallbacks each individual file uses when run alone. Only 5 papers from one
category - this is about proving connectivity and correctness, not doing
the real 50,000-100,000 paper collection (that is a separate, deliberate,
much longer run - see README.md).

Run directly:
    python scripts\\verify_step3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.arxiv_loader import fetch_papers
from src.data.preprocessor import preprocess_papers
from src.data.instruction_gen import generate_dataset
from src.data.dataset import _build_from_records

results: list[tuple[str, bool, str]] = []


def record(label: str, passed: bool, detail: str = "") -> None:
    results.append((label, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail else ""))


def main() -> int:
    print("=" * 70)
    print("scientific-llm - Step 3 verification (arXiv data pipeline)")
    print("=" * 70)

    print("\nFetching 5 real papers from arXiv (cat: gr-qc)...")
    try:
        papers = fetch_papers(categories=["gr-qc"], max_results=5, progress=False)
        record("arXiv API fetch", len(papers) > 0, f"{len(papers)} paper(s)")
    except Exception as e:  # noqa: BLE001
        record("arXiv API fetch", False, f"{type(e).__name__}: {e}")
        print("=" * 70)
        print("Stopping early - later checks depend on this one.")
        print("(Check your internet connection - export.arxiv.org must be reachable.)")
        return 1

    print("\nPreprocessing...")
    cleaned = preprocess_papers(papers)
    record(
        "Preprocessing keeps at least one paper",
        len(cleaned) > 0,
        f"{len(cleaned)}/{len(papers)} survived",
    )
    if not cleaned:
        print("=" * 70)
        print("Stopping early - later checks depend on this one.")
        return 1
    record(
        "Cleaned papers have non-empty abstracts",
        all(p["clean_abstract"] for p in cleaned),
    )

    print("\nGenerating instruction pairs...")
    pairs = generate_dataset(cleaned)
    record("Instruction pairs generated", len(pairs) > 0, f"{len(pairs)} pair(s)")
    schema_ok = all(
        {"instruction", "input", "output"} <= set(p.keys()) and p["output"]
        for p in pairs
    )
    record("Every pair has instruction/input/output with non-empty output", schema_ok)

    print("\nBuilding HuggingFace dataset...")
    try:
        dataset_dict = _build_from_records(pairs, val_fraction=0.1, seed=42)
        record(
            "HuggingFace DatasetDict builds successfully",
            True,
            f"train={len(dataset_dict['train'])}, validation={len(dataset_dict['validation'])}",
        )
        has_text_field = "text" in dataset_dict["train"].column_names
        record("Formatted 'text' field present for training", has_text_field)
    except Exception as e:  # noqa: BLE001
        record("HuggingFace DatasetDict builds successfully", False, f"{type(e).__name__}: {e}")

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

    print("All Step 3 checks passed. Ready for Step 4.")
    print("\nWhen you are ready to do the real full-scale collection, run e.g.:")
    print("  python src\\data\\arxiv_loader.py --max-results 50000")
    print("  python src\\data\\preprocessor.py")
    print("  python src\\data\\instruction_gen.py")
    print("  python src\\data\\dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
