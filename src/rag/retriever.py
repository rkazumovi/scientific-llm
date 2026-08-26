"""
scientific-llm - Step 6c: retriever - ties embeddings.py and
vector_store.py together into "given a corpus of papers, answer a text
query with the most relevant ones."

Two entry points:
  - build_index_from_papers(): embeds each paper's clean_abstract
    (Step 3's preprocessor.py output) and stores it with title/id
    metadata - the corpus-building half of RAG.
  - Retriever.retrieve(): embeds a query the same way and searches the
    index - the query-time half of RAG.

Split from vector_store.py on purpose: vector_store.py only knows about
already-computed vectors (fully testable with synthetic data, no model
needed - see that file). This file is the layer that actually calls the
embedding model, so its query-embedding step cannot be exercised without
a real (or injected fake) model - see embed_fn on Retriever.__init__ for
how the tests in this project's build process cover its logic anyway.

Run directly:
    python src\\rag\\retriever.py
(builds a small real index from Step 3's DEMO_PAPERS-style fallback data
and the real embedding model, then retrieves for a demo query.)
"""

import sys
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag.vector_store import VectorStore


def build_index_from_papers(papers: list[dict], embed_fn: Callable[[list[str]], np.ndarray]) -> VectorStore:
    """embed_fn: any callable text-list -> (n, dim) embeddings, e.g.
    `lambda texts: embed_texts(model, texts)` from embeddings.py, or a
    fake for testing. Keeping this as an injected function rather than
    hardcoding embeddings.py's model here is what lets retriever.py's
    corpus-building logic (below) be unit tested without downloading or
    running a real embedding model."""
    texts = [p.get("clean_abstract", p.get("abstract", "")) for p in papers]
    embeddings = embed_fn(texts)
    dim = embeddings.shape[1]

    metadatas = [
        {
            "paper_id": p.get("id", f"paper-{i}"),
            "title": p.get("title", ""),
            "text": texts[i],
        }
        for i, p in enumerate(papers)
    ]

    store = VectorStore(dim=dim)
    store.add(embeddings, metadatas)
    return store


class Retriever:
    def __init__(self, store: VectorStore, embed_fn: Callable[[list[str]], np.ndarray]):
        self.store = store
        self.embed_fn = embed_fn

    def retrieve(self, query: str, k: int = 3, exclude_paper_id: str | None = None) -> list[dict]:
        """Retrieves up to k results for `query`. exclude_paper_id
        filters out one document AFTER search (e.g. the golden document
        itself, when building RAFT distractors in raft.py) - searching
        for k+1 first and dropping one afterward rather than k, so
        excluding a hit does not silently starve the result count."""
        query_embedding = self.embed_fn([query])[0]
        raw_k = k + 1 if exclude_paper_id is not None else k
        results = self.store.search(query_embedding, k=raw_k)

        if exclude_paper_id is not None:
            results = [r for r in results if r.get("paper_id") != exclude_paper_id]

        return results[:k]


def main() -> int:
    try:
        from src.data.preprocessor import DEMO_PAPERS, preprocess_papers
        from src.rag.embeddings import load_embedding_model, embed_texts

        print("Preprocessing built-in demo papers...")
        papers = preprocess_papers(DEMO_PAPERS)
        if not papers:
            print("FAIL: no usable demo papers.")
            return 1

        print("Loading embedding model (small, CPU, downloads on first run)...")
        model = load_embedding_model()
        embed_fn = lambda texts: embed_texts(model, texts)  # noqa: E731

        print(f"Building index from {len(papers)} paper(s)...")
        store = build_index_from_papers(papers, embed_fn)
        retriever = Retriever(store, embed_fn)

        query = "How does heat diffuse through a solid over time?"
        print(f"\nRetrieving for query: {query!r}")
        results = retriever.retrieve(query, k=2)

        for r in results:
            print(f"  score={r['score']:.3f}  {r['title']!r}")

        if not results:
            print("FAIL: no results returned for a query that should match the heat-equation demo paper.")
            return 1
        if "heat" not in results[0]["title"].lower():
            print(f"FAIL: expected the heat-equation paper to rank first, got {results[0]['title']!r}")
            return 1

        print("\nPASS: retriever builds an index and returns the semantically relevant paper first.")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
