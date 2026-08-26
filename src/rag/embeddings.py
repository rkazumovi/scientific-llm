"""
scientific-llm - Step 6a: embedding model for retrieval.

Wraps a small sentence-embedding model (NOT Mistral-7B) that turns text
into a fixed-length vector for similarity search. Deliberately a
separate, much smaller model from the 7B model this whole project fine-
tunes: embedding a corpus of paper abstracts is a one-time (or
infrequent) batch job, not something that needs the 7B model's language
generation ability, and running it on CPU keeps the full 8GB of VRAM
free for Mistral - important, since Step 6c's retriever and Step 4's
trainer may both be in memory during RAFT dataset generation.

Model: sentence-transformers/all-MiniLM-L6-v2 - 384-dimensional
embeddings, ~80MB, a standard, well-tested choice for exactly this kind
of small-corpus semantic search (not a claim that it is the best
possible embedding model - it is a reasonable, lightweight default that
keeps this step's footprint small).

Run directly:
    python src\\rag\\embeddings.py
(downloads the embedding model on first run - much smaller than Step 2's
7B download - and embeds a couple of demo sentences to confirm shape and
that semantically similar sentences score higher than dissimilar ones.)
"""

import sys
import traceback

import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL, device: str = "cpu"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def embed_texts(model, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Returns an (n_texts, dim) float32 array, L2-normalized so that a
    plain dot product between two rows equals cosine similarity - this
    is what vector_store.py's FAISS IndexFlatIP expects (see that file's
    docstring)."""
    embeddings = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True
    )
    return embeddings.astype(np.float32)


def main() -> int:
    try:
        print(f"Loading embedding model ({DEFAULT_EMBEDDING_MODEL}, CPU)...")
        model = load_embedding_model()

        demo_texts = [
            "The heat equation describes how temperature diffuses over time.",
            "Diffusion of thermal energy is governed by a parabolic PDE.",
            "The stock market closed higher today on strong earnings.",
        ]
        print(f"Embedding {len(demo_texts)} demo sentences...")
        embeddings = embed_texts(model, demo_texts)

        print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
        if embeddings.shape[0] != len(demo_texts):
            print("FAIL: wrong number of embeddings returned.")
            return 1

        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            print(f"FAIL: embeddings are not L2-normalized (norms={norms}).")
            return 1

        # Sentences 0 and 1 are about the same physics topic in different
        # words; sentence 2 is unrelated (finance). A working embedding
        # model should put 0 and 1 closer together (higher dot product)
        # than either is to 2 - the actual semantic behavior this file
        # exists to provide, not just "did it run."
        sim_0_1 = float(np.dot(embeddings[0], embeddings[1]))
        sim_0_2 = float(np.dot(embeddings[0], embeddings[2]))
        print(f"cosine(heat eq, diffusion PDE) = {sim_0_1:.3f}")
        print(f"cosine(heat eq, stock market)  = {sim_0_2:.3f}")

        if not (sim_0_1 > sim_0_2):
            print("FAIL: embedding model did not rank the semantically related sentence higher.")
            return 1

        print("PASS: embedding model loads, produces normalized vectors, and ranks semantic similarity sensibly.")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
