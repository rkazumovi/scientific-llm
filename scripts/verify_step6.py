"""
scientific-llm - Step 6 verification: RAG retrieval + RAFT dataset
construction, end to end against real data.

Chains together: a real arXiv fetch (Step 3, a slightly bigger pull than
prior steps - RAG needs enough papers that retrieval and distractor
selection are meaningful, not a single-document corpus) -> real
embeddings (Step 6a, downloads a small model on first run) -> a real
FAISS index (Step 6b/6c) -> real retrieval -> a real RAFT dataset (Step
6d), checked at each stage.

Unlike Step 4/5, this step does not touch the 7B model or the GPU at
all - embeddings.py runs the small sentence-transformers model on CPU
by design (see that file's docstring), so this script does not call
load_base_model(). It DOES need network access (arXiv API + downloading
the embedding model on first run).

Run directly:
    python scripts\\verify_step6.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.arxiv_loader import fetch_papers
from src.data.preprocessor import preprocess_papers
from src.rag.embeddings import load_embedding_model, embed_texts
from src.rag.retriever import build_index_from_papers, Retriever
from src.rag.raft import generate_raft_dataset

results: list[tuple[str, bool, str]] = []

# RAG/RAFT needs several papers to have meaningful distractors - a
# noticeably bigger pull than Step 3/4/5's 3-5 paper smoke tests.
FETCH_CATEGORIES = ["gr-qc", "math.AP"]
FETCH_MAX_RESULTS = 15
NUM_DISTRACTORS = 2


def record(label: str, passed: bool, detail: str = "") -> None:
    results.append((label, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail else ""))


def main() -> int:
    print("=" * 70)
    print("scientific-llm - Step 6 verification (RAG retrieval + RAFT)")
    print("=" * 70)

    print(f"\nFetching real papers from arXiv (cats: {FETCH_CATEGORIES})...")
    try:
        papers = fetch_papers(categories=FETCH_CATEGORIES, max_results=FETCH_MAX_RESULTS, progress=False)
    except Exception as e:  # noqa: BLE001
        record("arXiv fetch", False, f"{type(e).__name__}: {e}")
        return 1
    record("arXiv fetch", len(papers) > 0, f"{len(papers)} paper(s)")

    cleaned = preprocess_papers(papers)
    record("Papers usable after preprocessing", len(cleaned) >= 3, f"{len(cleaned)} paper(s)")
    if len(cleaned) < 3:
        print("Need at least 3 usable papers for a meaningful retrieval/distractor demo - stopping.")
        return 1

    print("\nLoading embedding model (small, CPU, downloads on first run)...")
    try:
        model = load_embedding_model()
        embed_fn = lambda texts: embed_texts(model, texts)  # noqa: E731
    except Exception as e:  # noqa: BLE001
        record("Embedding model load", False, f"{type(e).__name__}: {e}")
        return 1
    record("Embedding model load", True)

    print(f"\nBuilding FAISS index from {len(cleaned)} paper(s)...")
    try:
        store = build_index_from_papers(cleaned, embed_fn)
    except Exception as e:  # noqa: BLE001
        record("FAISS index build", False, f"{type(e).__name__}: {e}")
        return 1
    record("FAISS index build", len(store) == len(cleaned), f"{len(store)} vector(s) stored")

    print("\nRetrieving for a query built from the first paper's own title...")
    retriever = Retriever(store, embed_fn)
    query = cleaned[0]["title"]
    retrieval_results = retriever.retrieve(query, k=3)
    record(
        "Retrieval returns results, self ranks first",
        bool(retrieval_results) and retrieval_results[0]["paper_id"] == cleaned[0].get("id"),
        f"top result: {retrieval_results[0]['title'] if retrieval_results else None!r}",
    )

    print(f"\nGenerating RAFT dataset (num_distractors={NUM_DISTRACTORS})...")
    try:
        raft_examples = generate_raft_dataset(cleaned, retriever, num_distractors=NUM_DISTRACTORS)
    except Exception as e:  # noqa: BLE001
        record("RAFT dataset generation", False, f"{type(e).__name__}: {e}")
        return 1
    record("RAFT dataset generation", len(raft_examples) > 0, f"{len(raft_examples)} example(s)")

    if raft_examples:
        golden_labels_present = all(ex["golden_label"] in ex["text"] for ex in raft_examples)
        record("Every RAFT example's text cites its own golden_label", golden_labels_present)

        distractor_counts = [ex["num_distractors_used"] for ex in raft_examples]
        under_requested = sum(1 for c in distractor_counts if c < NUM_DISTRACTORS)
        print(
            f"  distractor counts: min={min(distractor_counts)}, max={max(distractor_counts)}"
            + (f" ({under_requested} example(s) got fewer than requested - corpus-size limited, not a bug)"
               if under_requested else "")
        )

        example = raft_examples[0]
        print(f"\nFirst RAFT example (truncated):\n{example['text'][:500]}...")

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

    print("All Step 6 checks passed. Ready for Step 7.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
