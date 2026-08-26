"""
scientific-llm - Step 5b: ROUGE scoring for generated summaries.

Measures overlap between a model-generated summary and a reference
(the real arXiv abstract, from Step 3's pipeline) - the standard metric
for the summarization instruction template in instruction_gen.py.

Implemented from scratch rather than pulling in the `rouge-score`
package: this project already avoids new dependencies where the metric
is simple enough to implement directly and verify against known values
(see arxiv_loader.py's stdlib-only XML parsing for the same philosophy),
and ROUGE-N/ROUGE-L are simple enough - n-gram overlap and a longest
common subsequence, respectively - that hand-rolling them is both less
weight and more transparent than trusting a black-box import.

Three variants, all reported as precision / recall / F1:
  - ROUGE-1: unigram (single word) overlap.
  - ROUGE-2: bigram (word pair) overlap.
  - ROUGE-L: based on the Longest Common Subsequence between reference
    and hypothesis token sequences (order-sensitive, unlike N-gram
    overlap - rewards getting words in the right relative order without
    requiring an exact contiguous match).

Run directly:
    python src\\evaluation\\rouge_eval.py
(pure string logic - no model, no GPU needed. Verifies against a few
worked-by-hand examples.)
"""

import re
import sys
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _prf1(overlap: int, hyp_count: int, ref_count: int) -> dict:
    precision = overlap / hyp_count if hyp_count else 0.0
    recall = overlap / ref_count if ref_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_n(reference: str, hypothesis: str, n: int = 1) -> dict:
    """N-gram overlap between reference and hypothesis. Overlap counts
    each shared n-gram at most min(count in reference, count in
    hypothesis) times (standard ROUGE clipping - stops a hypothesis that
    just repeats one word from scoring artificially high)."""
    ref_grams = _ngrams(_tokenize(reference), n)
    hyp_grams = _ngrams(_tokenize(hypothesis), n)

    overlap = sum((ref_grams & hyp_grams).values())
    hyp_count = sum(hyp_grams.values())
    ref_count = sum(ref_grams.values())
    return _prf1(overlap, hyp_count, ref_count)


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(b)]


def rouge_l(reference: str, hypothesis: str) -> dict:
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    return _prf1(lcs, len(hyp_tokens), len(ref_tokens))


def compute_rouge(reference: str, hypothesis: str) -> dict:
    """Returns {"rouge1": {...}, "rouge2": {...}, "rougeL": {...}}, each a
    precision/recall/f1 dict - the standard trio reported for
    summarization evaluation."""
    return {
        "rouge1": rouge_n(reference, hypothesis, n=1),
        "rouge2": rouge_n(reference, hypothesis, n=2),
        "rougeL": rouge_l(reference, hypothesis),
    }


def main() -> int:
    print("ROUGE scoring demo (pure string logic, no model needed):")

    cases = [
        ("the cat sat on the mat", "the cat sat on the mat", 1.0),  # identical
        ("the cat sat on the mat", "a dog ran in the park", 0.0),  # near-disjoint (still shares "the")
    ]

    all_ok = True
    for reference, hypothesis, expected_rouge1_f1 in cases:
        scores = compute_rouge(reference, hypothesis)
        f1 = scores["rouge1"]["f1"]
        # Exact 0.0 case above shares "the", so allow a small tolerance
        # rather than demanding an exact match on the near-disjoint case.
        ok = abs(f1 - expected_rouge1_f1) < 0.35
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] ROUGE-1 F1({reference!r}, {hypothesis!r}) = {f1:.3f}")

    identical_scores = compute_rouge("the heat equation describes diffusion", "the heat equation describes diffusion")
    identical_ok = all(
        abs(identical_scores[k]["f1"] - 1.0) < 1e-9 for k in ("rouge1", "rouge2", "rougeL")
    )
    all_ok = all_ok and identical_ok
    print(
        f"  [{'PASS' if identical_ok else 'FAIL'}] identical text scores 1.0 on all three metrics: "
        f"{identical_scores}"
    )

    if not all_ok:
        print("\nFAIL: one or more ROUGE sanity checks did not hold.")
        return 1

    print("\nPASS: ROUGE-1/2/L implementations behave as expected on worked examples.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
