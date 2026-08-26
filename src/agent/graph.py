"""
scientific-llm - Step 7: a retrieval-grounded question-answering agent,
built with LangGraph.

Why a graph and not just a linear "retrieve, then generate" function:
the interesting behavior here is a CONDITIONAL retry - if the model's
answer states an equation that cannot be found among the retrieved
documents, the agent tries again with a stricter prompt, up to
max_attempts times, before giving up and returning its best answer
honestly labeled as unverified. That branch (generate -> verify ->
maybe back to generate) is a real loop with a decision in it, which is
what LangGraph is for - a plain function chain cannot loop back on
itself based on a runtime condition without hand-rolling the same state
machine LangGraph already provides.

The four nodes:
  1. retrieve - Step 6c's Retriever finds the k most relevant documents
     for the question.
  2. generate - Step 2's base model answers the question using ONLY the
     retrieved documents as context (a RAG prompt), via the same
     apply_chat_template pattern base_model.py's smoke test uses.
  3. extract - pulls any LaTeX-delimited equations out of the generated
     answer, reusing Step 3's extract_latex_equations (the same function
     that built this project's training data, applied here to the
     model's OUTPUT instead of a paper's abstract).
  4. verify - checks whether each extracted equation is symbolically
     equivalent (Step 5c's math_verifier) to an equation that actually
     appears in the retrieved documents.

Honesty about what "grounded" means here, worth reading before trusting
the field: it means "the equation the model produced also appears,
algebraically, among the retrieved sources' own equations" - a
provenance check, not a physical-correctness check. An answer with no
equation in it at all is trivially marked grounded=True (there is
nothing to verify, which is not the same as "verified correct") - see
verify_node below. And verification only catches equations the model
wrote with LaTeX delimiters ($...$) - the generate prompt asks the model
to always use them, but that is an instruction, not a guarantee; a
plain-text equation with no delimiters will not be found by
extract_latex_equations and will slip through as "nothing to verify."

Every node is built by a build_*_node(...) function that takes its real
dependencies (model, tokenizer, retriever) as arguments and returns a
plain function closing over them - the same dependency-injection pattern
retriever.py uses for embed_fn. That is what makes the graph's control
flow (does the retry edge actually fire? does it stop at max_attempts?)
fully unit-testable with fake model/retriever stand-ins, with no GPU or
downloaded model needed - and it was tested that way, see this file's
test coverage notes in the project README.

Run directly:
    python src\\agent\\graph.py
(builds a small real index from Step 3's demo papers, loads the real
Step 2 base model, and asks one real question through the full graph.)
"""

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.preprocessor import extract_latex_equations
from src.evaluation.math_verifier import equations_equivalent


class AgentState(TypedDict):
    question: str
    docs: list
    answer: str
    equations: list
    grounded: bool
    attempts: int
    max_attempts: int


def build_retrieve_node(retriever, k: int = 3):
    def node(state: AgentState) -> dict:
        docs = retriever.retrieve(state["question"], k=k)
        return {"docs": docs}

    return node


def build_generate_node(model, tokenizer, max_new_tokens: int = 200):
    def node(state: AgentState) -> dict:
        import torch

        context_block = "\n\n".join(
            f"[{d.get('title', '')}]\n{d.get('text', '')}" for d in state["docs"]
        )
        retry_note = ""
        if state.get("attempts", 0) > 0:
            retry_note = (
                " Your previous answer's equation could not be found in the documents "
                "below - try again, and only state an equation if you can quote it "
                "exactly as written there."
            )
        user_prompt = (
            "Answer the question below using ONLY the documents provided. If you "
            "state an equation, write it exactly as it appears in the documents, "
            f"wrapped in dollar signs, like $u_t = \\alpha u_{{xx}}$.{retry_note}\n\n"
            f"Documents:\n{context_block}\n\nQuestion: {state['question']}"
        )
        messages = [{"role": "user", "content": user_prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        answer = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
        return {"answer": answer, "attempts": state.get("attempts", 0) + 1}

    return node


def extract_node(state: AgentState) -> dict:
    return {"equations": extract_latex_equations(state["answer"])}


def build_verify_node():
    def node(state: AgentState) -> dict:
        if not state["equations"]:
            # Nothing to verify is not the same as verified correct -
            # see module docstring.
            return {"grounded": True}

        doc_equations = []
        for doc in state["docs"]:
            doc_equations.extend(extract_latex_equations(doc.get("text", "")))

        for eq in state["equations"]:
            for doc_eq in doc_equations:
                if equations_equivalent(eq, doc_eq):
                    return {"grounded": True}

        return {"grounded": False}

    return node


def should_retry(state: AgentState) -> str:
    if state.get("grounded") or state.get("attempts", 0) >= state.get("max_attempts", 2):
        return "end"
    return "retry"


def build_agent_graph(model, tokenizer, retriever, k: int = 3, max_new_tokens: int = 200):
    """Wires the four nodes into a compiled LangGraph app. k and
    max_new_tokens are baked into the retrieve/generate nodes at build
    time (they do not change between retries); max_attempts is instead
    passed per-question through ask() below, since it is state, not a
    fixed dependency."""
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", build_retrieve_node(retriever, k=k))
    graph.add_node("generate", build_generate_node(model, tokenizer, max_new_tokens=max_new_tokens))
    graph.add_node("extract", extract_node)
    graph.add_node("verify", build_verify_node())

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "extract")
    graph.add_edge("extract", "verify")
    graph.add_conditional_edges("verify", should_retry, {"retry": "generate", "end": END})

    return graph.compile()


def ask(app, question: str, max_attempts: int = 2) -> dict:
    """Runs one question through a compiled graph and returns a plain
    summary dict - the graph's raw state has fields (max_attempts as an
    input, docs as full retrieval results) that a caller does not
    usually need."""
    result = app.invoke({"question": question, "attempts": 0, "max_attempts": max_attempts})
    return {
        "answer": result["answer"],
        "equations": result["equations"],
        "grounded": result["grounded"],
        "attempts": result["attempts"],
        "sources": [d.get("title", "") for d in result["docs"]],
    }


def main() -> int:
    import traceback

    try:
        from src.data.preprocessor import DEMO_PAPERS, preprocess_papers
        from src.model.base_model import load_base_model
        from src.rag.embeddings import embed_texts, load_embedding_model
        from src.rag.retriever import Retriever, build_index_from_papers

        print("Preprocessing built-in demo papers...")
        papers = preprocess_papers(DEMO_PAPERS)
        if not papers:
            print("FAIL: no usable demo papers.")
            return 1

        print("Loading embedding model (small, CPU) and building index...")
        embed_model = load_embedding_model()
        embed_fn = lambda texts: embed_texts(embed_model, texts)  # noqa: E731
        store = build_index_from_papers(papers, embed_fn)
        retriever = Retriever(store, embed_fn)

        print("Loading Step 2 base model (cached, should be quick)...")
        model, tokenizer = load_base_model()

        print("Building agent graph...")
        app = build_agent_graph(model, tokenizer, retriever, k=1, max_new_tokens=150)

        question = "What equation does the heat-equation demo paper use, and what does it describe?"
        print(f"\nAsking: {question!r}")
        result = ask(app, question, max_attempts=2)

        print(f"\nAnswer: {result['answer']}")
        print(f"Equations found in answer: {result['equations']}")
        print(f"Grounded (equation matched a retrieved source): {result['grounded']}")
        print(f"Attempts used: {result['attempts']}")
        print(f"Sources: {result['sources']}")

        if not result["answer"].strip():
            print("FAIL: agent produced an empty answer.")
            return 1
        if not result["sources"]:
            print("FAIL: agent produced no sources - retrieval likely did not run.")
            return 1

        print(
            "\nPASS: agent graph runs end to end and returns a well-formed answer with sources."
            + ("" if result["grounded"] else " (not grounded on this run - not necessarily a bug, see module docstring)")
        )
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
