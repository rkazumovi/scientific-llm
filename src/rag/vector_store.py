"""
scientific-llm - Step 6b: FAISS vector store.

A thin wrapper around a flat FAISS index plus a parallel Python list of
metadata dicts (title, abstract text, source paper id - whatever the
caller wants attached to each vector). "Flat" (IndexFlatIP - brute-force
inner product) rather than an approximate index (IVF, HNSW, ...) is
deliberate: this project's corpus is a few thousand paper abstracts at
most, and at that scale exact brute-force search is fast enough that
trading it away for an approximate index's extra complexity (training,
tuning nlist/nprobe) would not be buying anything real.

Inner product (not L2 distance) because embeddings.py already returns
L2-normalized vectors - for unit vectors, inner product IS cosine
similarity, and higher is better (a more intuitive score than a
distance where lower is better), so IndexFlatIP is the matching choice
metric-for-metric with how embeddings.py normalizes its output.

Run directly:
    python src\\rag\\vector_store.py
(exercises add/search/save/load with small synthetic vectors - no
embedding model or GPU needed, this file only deals in already-computed
vectors.)
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[dict] = []

    def add(self, embeddings: np.ndarray, metadatas: list[dict]) -> None:
        if embeddings.shape[0] != len(metadatas):
            raise ValueError(
                f"embeddings has {embeddings.shape[0]} rows but got {len(metadatas)} metadata dicts"
            )
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"embeddings are dim {embeddings.shape[1]}, store expects dim {self.dim}")

        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        faiss.normalize_L2(embeddings)  # idempotent if already normalized; cheap safety net
        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[dict]:
        """query_embedding: a single (dim,) or (1, dim) vector. Returns
        up to k results as [{"score": float, **metadata}], best first.
        k is clamped to the number of stored vectors so asking for more
        neighbors than exist does not raise or return padding rows."""
        if self.index.ntotal == 0:
            return []

        query = np.ascontiguousarray(query_embedding.astype(np.float32)).reshape(1, -1)
        faiss.normalize_L2(query)
        k = min(k, self.index.ntotal)

        scores, ids = self.index.search(query, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append({"score": float(score), **self.metadata[idx]})
        return results

    def save(self, path: str) -> None:
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out_dir / "index.faiss"))
        with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "metadata": self.metadata}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        in_dir = Path(path)
        with (in_dir / "metadata.json").open(encoding="utf-8") as f:
            saved = json.load(f)
        store = cls(dim=saved["dim"])
        store.index = faiss.read_index(str(in_dir / "index.faiss"))
        store.metadata = saved["metadata"]
        return store

    def __len__(self) -> int:
        return self.index.ntotal


def main() -> int:
    print("VectorStore demo (synthetic vectors, no embedding model needed):")

    dim = 4
    store = VectorStore(dim=dim)
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],  # close to vector 0
        ],
        dtype=np.float32,
    )
    metadatas = [
        {"title": "Doc A", "topic": "x-axis"},
        {"title": "Doc B", "topic": "y-axis"},
        {"title": "Doc C", "topic": "near x-axis"},
    ]
    store.add(embeddings, metadatas)

    if len(store) != 3:
        print(f"FAIL: expected 3 stored vectors, got {len(store)}.")
        return 1
    print(f"[PASS] added {len(store)} vectors")

    query = np.array([1.0, 0.05, 0.0, 0.0], dtype=np.float32)
    results = store.search(query, k=2)
    print(f"Top-2 results for a query near Doc A: {results}")

    if not results or results[0]["title"] != "Doc A":
        print("FAIL: nearest neighbor should have been Doc A.")
        return 1
    if len(results) != 2 or results[1]["title"] != "Doc C":
        print("FAIL: second-nearest neighbor should have been Doc C.")
        return 1
    print("[PASS] search ranks nearest neighbors correctly")

    over_k_results = store.search(query, k=100)
    if len(over_k_results) != 3:
        print(f"FAIL: k larger than store size should clamp to store size (3), got {len(over_k_results)}.")
        return 1
    print("[PASS] k larger than store size clamps instead of erroring")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store.save(tmp)
        reloaded = VectorStore.load(tmp)
        reloaded_results = reloaded.search(query, k=2)
        if reloaded_results != results:
            print(f"FAIL: reloaded store gave different results.\n  original: {results}\n  reloaded: {reloaded_results}")
            return 1
        print("[PASS] save/load round-trip preserves search results")

    print("\nPASS: VectorStore add/search/save/load all behave correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
