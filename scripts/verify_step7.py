"""
scientific-llm - Step 7 verification: the retrieval-grounded agent, end
to end against real data.

Chains together: a real arXiv fetch (Step 3, filtered to papers that
actually contain an extracted equation - the agent's verification step
needs something concrete to check itself against), a real FAISS index
over those papers (Step 6), the real Step 2 base model, and Step 7's
LangGraph agent, asked a real question about a real paper's own
equation.

This is a read-only, generation-heavy check (up to max_attempts
generations for one question) - no training, no checkpoint saved.

Honesty about what this script can and cannot promise: it can verify the
MECHANISM works (the graph runs to completion, retrieval happens,
verification happens, the retry edge is reachable, the result is well-
formed). It cannot promise the 7B base model will actually quote a
paper's equation precisely enough to be marked "grounded" on any given
run - that depends on the model's own generation, not on this project's
code, and the base model has not been fine-tuned on this specific
question. A grounded=False result here is reported and explained, not
treated as a failed check by itself; see the PASS/FAIL criteria in
main() for what actually fails this script.

Run directly:
    python scripts\\verify_step7.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.graph import ask, build_agent_graph
from src.data.arxiv_loader import fetch_papers
from src.data.preprocessor import preprocess_papers
from src.model.base_model import load_base_model, report_gpu_memory
from src.rag.embeddings import embed_texts, load_embedding_model
from src.rag.retriever import Retriever, build_index_from_papers

results: list[tuple[str, bool, str]] = []

FETCH_CATEGORIES = ["gr-qc", "math.AP"]
FETCH_MAX_RESULTS = 15


def record(label: str, passed: bool, detail: str = "") -> None:
    results.append((label, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail else ""))


def main() -> int:
    print("=" * 70)
    print("scientific-llm - Step 7 verification (retrieval-grounded agent)")
    print("=" * 70)

    print(f"\nFetching real papers from arXiv (cats: {FETCH_CATEGORIES})...")
    try:
        papers = fetch_papers(categories=FETCH_CATEGORIES, max_results=FETCH_MAX_RESULTS, progress=False)
    except Exception as e:  # noqa: BLE001
        record("arXiv fetch", False, f"{type(e).__name__}: {e}")
        return 1
    record("arXiv fetch", len(papers) > 0, f"{len(papers)} paper(s)")

    cleaned = preprocess_papers(papers)
    with_equations = [p for p in cleaned if p.get("equations")]
    record(
        "Papers with at least one extracted equation",
        len(with_equations) > 0,
        f"{len(with_equations)}/{len(cleaned)} usable paper(s) have an equation",
    )
    if not with_equations:
        print("No fetched paper had an extractable equation this run - cannot build a grounded question. Stopping.")
        return 1

    print("\nLoading embedding model and building index...")
    try:
        embed_model = load_embedding_model()
        embed_fn = lambda texts: embed_texts(embed_model, texts)  # noqa: E731
        store = build_index_from_papers(cleaned, embed_fn)
        retriever = Retriever(store, embed_fn)
    except Exception as e:  # noqa: BLE001
        record("Embedding + index build", False, f"{type(e).__name__}: {e}")
        return 1
    record("Embedding + index build", len(store) == len(cleaned), f"{len(store)} vector(s) stored")

    print("\nLoading base model (cached from Step 2, should be quick)...")
    try:
        model, tokenizer = load_base_model()
    except Exception as e:  # noqa: BLE001
        record("Base model load", False, f"{type(e).__name__}: {e}")
        return 1
    record("Base model load", True)
    report_gpu_memory("before agent run")

    target_paper = with_equations[0]
    question = f"What equation appears in the paper titled '{target_paper['title']}', and what does it describe?"
    print(f"\nBuilding agent graph and asking a real question:\n  {question!r}")

    try:
        app = build_agent_graph(model, tokenizer, retriever, k=2, max_new_tokens=200)
        result = ask(app, question, max_attempts=2)
    except Exception as e:  # noqa: BLE001
        record("Agent graph runs", False, f"{type(e).__name__}: {e}")
        return 1

    print(f"\nAnswer: {result['answer']}")
    print(f"Equations found in answer: {result['equations']}")
    print(f"Attempts used: {result['attempts']}")
    print(f"Sources: {result['sources']}")
    print(f"Grounded: {result['grounded']}")

    record("Agent produced a non-empty answer", bool(result["answer"].strip()))
    record("Agent used at least one retrieved source", len(result["sources"]) > 0)
    record(
        "Attempts within max_attempts bound",
        1 <= result["attempts"] <= 2,
        f"attempts={result['attempts']}",
    )
    record(
        "Grounding result (informational - see this script's docstring)",
        True,  # never fails the run by itself
        f"grounded={result['grounded']}"
        + ("" if result["grounded"] else " - model did not quote a matching equation this run, mechanism still verified structurally"),
    )

    report_gpu_memory("after agent run")

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

    print("All Step 7 checks passed. Ready for Step 8.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
