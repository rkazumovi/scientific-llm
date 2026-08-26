"""
scientific-llm - Step 6d: RAFT (Retrieval-Augmented Fine-Tuning) dataset
construction.

Plain RAG only touches inference: retrieve documents, stuff them into
the prompt, generate. RAFT changes what happens at TRAINING time
instead - each training example is built with a mix of the actual
source document (the "golden" document, called "oracle" in the original
RAFT paper) and several retrieved DISTRACTOR documents that are
topically similar but do not contain the answer, all shuffled together
and labeled "Document 1".."Document k". The target output explicitly
names which document the answer came from. Training on that teaches the
model two things ordinary instruction-tuning on (question, answer)
pairs does not: to locate the relevant passage among plausible-looking
noise, and to say where an answer came from - the second half of what
"retrieval-augmented" is supposed to mean, not just "the context window
happened to contain the answer."

Distractors are chosen by retrieval, not at random - Step 6c's Retriever
naturally returns near-miss papers (similar topic, wrong specific
content) when you query with a paper's own instruction text and exclude
that paper itself. Those near-misses are exactly the kind of confusable,
useful-to-train-against distractor RAFT calls for, rather than an
unrelated paper drawn at random, which the model could reject by topic
alone without ever having to actually read the documents.

Honesty about scope: this reuses Step 3's instruction_gen.py templates
for the underlying (instruction, input, output) content - the golden
document is simply the paper each pair's content already came from - so
the same caveat instruction_gen.py's own docstring states applies here
too: the summarization and result-extraction templates are genuinely
grounded (real abstract text as output), the equation-role template is
weaker (a heuristic anchor sentence, not a real explanation). RAFT
changes how the material is PRESENTED for training (with distractors,
requiring the model to locate and cite its source) - it does not
upgrade the underlying answer quality beyond what instruction_gen.py
already produced.

The formatted "text" field this file produces is deliberately the same
Alpaca-style shape dataset.py already uses, so RAFT examples can be
passed straight into Step 4's trainer.train(train_texts=...) with no
changes needed there.

Run directly:
    python src\\rag\\raft.py
(builds a RAFT example from built-in demo papers with a fake in-memory
retriever - no embedding model or GPU needed to exercise the shuffling/
formatting logic itself.)
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.instruction_gen import generate_instruction_pairs

RAFT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Context:\n{documents_block}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\nBased on {golden_label}: {output}"
)


def _format_documents_block(labeled_documents: list[tuple[str, dict]]) -> str:
    parts = []
    for label, doc in labeled_documents:
        title = doc.get("title", "")
        text = doc.get("text", "")
        parts.append(f"[{label}]\nTitle: {title}\n{text}")
    return "\n\n".join(parts)


def build_raft_example(
    instruction_pair: dict,
    golden_doc: dict,
    distractor_docs: list[dict],
    rng: random.Random | None = None,
) -> dict:
    """golden_doc / distractor_docs: dicts with at least "title" and
    "text" (matching Retriever's result shape - see retriever.py).
    Shuffles golden + distractors together, labels them "Document 1"..
    "Document k", and formats a single Alpaca-style training string that
    names the golden document's label in the target output."""
    rng = rng or random.Random()

    all_docs = [golden_doc] + list(distractor_docs)
    order = list(range(len(all_docs)))
    rng.shuffle(order)

    labels = [f"Document {i + 1}" for i in range(len(all_docs))]
    labeled_documents = [(labels[position], all_docs[doc_idx]) for position, doc_idx in enumerate(order)]

    golden_position = order.index(0)
    golden_label = labels[golden_position]

    text = RAFT_TEMPLATE.format(
        instruction=instruction_pair["instruction"],
        documents_block=_format_documents_block(labeled_documents),
        input=instruction_pair["input"],
        golden_label=golden_label,
        output=instruction_pair["output"],
    )

    return {
        "text": text,
        "golden_label": golden_label,
        "num_documents": len(all_docs),
        "num_distractors_used": len(distractor_docs),
    }


def generate_raft_dataset(papers: list[dict], retriever, num_distractors: int = 2) -> list[dict]:
    """For every instruction pair generated from every paper (Step 3's
    templates, called per-paper here rather than via
    instruction_gen.generate_dataset() specifically so each pair's
    source paper - needed as the golden document - stays known), builds
    one RAFT example. Retrieves num_distractors near-miss papers per
    pair via the retriever, excluding the golden paper itself. If the
    corpus is too small to find that many distractors, uses however many
    are available and reports it in the returned example's
    "num_distractors_used" rather than silently padding or failing."""
    examples = []
    for paper in papers:
        paper_id = paper.get("id", paper.get("title", ""))
        golden_doc = {
            "title": paper.get("title", ""),
            "text": paper.get("clean_abstract", paper.get("abstract", "")),
        }

        pairs = generate_instruction_pairs(paper)
        for pair in pairs:
            query = f"{pair['instruction']} {pair['input']}"
            distractors = retriever.retrieve(query, k=num_distractors, exclude_paper_id=paper_id)
            example = build_raft_example(pair, golden_doc, distractors)
            example["source_paper_id"] = paper_id
            examples.append(example)

    return examples


def main() -> int:
    print("RAFT formatting demo (fake in-memory retriever, no embedding model needed):")

    from src.data.preprocessor import DEMO_PAPERS, preprocess_papers

    papers = preprocess_papers(DEMO_PAPERS)
    if len(papers) < 1:
        print("FAIL: no usable demo papers.")
        return 1

    class FakeRetriever:
        """Returns a couple of hardcoded, clearly-not-the-golden-doc
        distractors, regardless of query - enough to exercise
        build_raft_example's shuffling/labeling without needing a real
        embedding model or a multi-paper corpus."""

        def retrieve(self, query: str, k: int = 2, exclude_paper_id: str | None = None) -> list[dict]:
            fake_distractors = [
                {"title": "Unrelated paper on black hole thermodynamics", "text": "We study black hole entropy."},
                {"title": "Unrelated paper on quantum gravity", "text": "We study loop quantum gravity corrections."},
            ]
            return fake_distractors[:k]

    examples = generate_raft_dataset(papers, FakeRetriever(), num_distractors=2)
    print(f"Generated {len(examples)} RAFT example(s) from {len(papers)} paper(s)")

    if not examples:
        print("FAIL: no RAFT examples generated.")
        return 1

    example = examples[0]
    print(f"\nExample text field:\n{'-' * 60}\n{example['text']}\n{'-' * 60}")
    print(f"golden_label={example['golden_label']}  num_documents={example['num_documents']}")

    all_ok = True
    for ex in examples:
        if ex["golden_label"] not in ex["text"]:
            print(f"FAIL: golden_label {ex['golden_label']!r} not found in formatted text.")
            all_ok = False
        if "### Response:\nBased on" not in ex["text"]:
            print("FAIL: expected response format not found.")
            all_ok = False
        if ex["num_documents"] != 1 + ex["num_distractors_used"]:
            print("FAIL: num_documents should equal 1 (golden) + num_distractors_used.")
            all_ok = False

    if not all_ok:
        return 1

    print("\nPASS: RAFT examples cite their golden document and are formatted correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
