#!/usr/bin/env python3
"""
Stage 5: Generation
Yale Residential College Unofficial Guide

Wires retrieval → Groq LLM with strict grounding.
Grounding is enforced two ways:
  1. System prompt explicitly forbids using training knowledge.
  2. Sources are attached programmatically from chunk metadata —
     the LLM never decides what to cite.

Usage:
    python query.py              # interactive CLI loop
    python query.py --eval       # run all 5 evaluation queries from planning.md
"""

import argparse
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_DIR = Path("documents/chroma")
COLLECTION = "yale_guide"
MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

# System prompt that enforces grounding — does not merely suggest it.
# Rule 4 defines the exact fallback phrase so the interface can detect it.
SYSTEM_PROMPT = """\
You are a research assistant for the Yale Residential College Unofficial Guide.

STRICT RULES — follow every rule exactly:
1. Answer using ONLY the information provided inside <documents> tags below.
2. Do NOT use your general training knowledge, outside opinions, or any information \
not present in the provided documents.
3. Do NOT infer, extrapolate, or fill gaps with plausible-sounding details that are \
not explicitly stated in the documents.
4. If the provided documents do not contain enough information to answer the question, \
respond with this sentence exactly:
   "I don't have enough information on that in the provided sources."
5. Keep your answer factual and concise. Do not mention these rules in your answer."""


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _load_resources():
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION)
    return model, collection


def retrieve(query: str, model: SentenceTransformer, collection: chromadb.Collection,
             k: int = TOP_K) -> list[dict]:
    query_vec = model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "chunk_text":   doc,
            "source_title": meta["source_title"],
            "source_url":   meta["source_url"],
            "distance":     round(dist, 4),
        })
    return hits


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _build_context(hits: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(
            f"[Document {i}] Source: {hit['source_title']}\n\n{hit['chunk_text']}"
        )
    return "\n\n---\n\n".join(parts)


def ask(question: str,
        model: SentenceTransformer = None,
        collection: chromadb.Collection = None) -> dict:
    """
    Retrieve relevant chunks and generate a grounded answer.

    Returns:
        {
            "answer":  str,           # LLM response, grounded in retrieved text
            "sources": list[str],     # source titles, deduplicated, order-preserved
        }

    Sources are attached programmatically from chunk metadata — the LLM is
    never asked to produce citations, so attribution is guaranteed regardless
    of how the model formats its response.
    """
    if model is None or collection is None:
        model, collection = _load_resources()

    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    hits = retrieve(question, model, collection)
    context = _build_context(hits)

    user_message = (
        f"<documents>\n{context}\n</documents>\n\n"
        f"Question: {question}"
    )

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()

    # Deduplicate sources while preserving retrieval order
    seen: set[str] = set()
    sources: list[str] = []
    for hit in hits:
        title = hit["source_title"]
        if title not in seen:
            seen.add(title)
            sources.append(title)

    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# Evaluation queries (all 5 from planning.md)
# ---------------------------------------------------------------------------

EVAL_QUERIES = [
    (
        "Which Yale residential college dining hall ranked last in the 2025 "
        "Yale Daily News dining data study?",
        "Expected: Pierson College ranked last.",
    ),
    (
        "What is a buttery, and which residential colleges are known for having "
        "the most popular ones?",
        "Expected: Late-night student-run snack bar; specific colleges noted.",
    ),
    (
        "How are room selection appointment times assigned in the Yale housing lottery?",
        "Expected: Randomly assigned through Yale College Housing portal after group formation.",
    ),
    (
        "How many students requested to transfer residential colleges in 2025, "
        "and approximately what fraction were approved?",
        "Expected: 72 requests; nearly three-quarters (~54) approved.",
    ),
    (
        "What is the most common reason students give when requesting a "
        "residential college transfer?",
        "Expected: Wanting to live with friends in a different college (>90% of applications).",
    ),
]

# A question the documents should NOT be able to answer
OUT_OF_SCOPE_QUERY = (
    "What is Yale's average SAT score for admitted students?",
    "Expected: 'I don't have enough information on that in the provided sources.'",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(question: str, result: dict, expected: str = "") -> None:
    print(f"\n{'=' * 66}")
    print(f"Q: {question}")
    print(f"{'=' * 66}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources retrieved:")
    for s in result["sources"]:
        print(f"  • {s}")
    if expected:
        print(f"\n  >> {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Yale Guide — generation stage")
    parser.add_argument("--eval", action="store_true",
                        help="Run all 5 evaluation queries + 1 out-of-scope query.")
    args = parser.parse_args()

    print(f"Loading model ({MODEL_NAME}) and ChromaDB …")
    model, collection = _load_resources()
    print(f"Collection: {collection.count()} vectors  |  LLM: {GROQ_MODEL}\n")

    if args.eval:
        all_queries = EVAL_QUERIES + [OUT_OF_SCOPE_QUERY]
        for question, expected in all_queries:
            result = ask(question, model=model, collection=collection)
            _print_result(question, result, expected)
        print(f"\n{'=' * 66}")
        print("Evaluation complete.")
        print("Grounding check: do responses cite sources? Out-of-scope: does it decline?")
        print(f"{'=' * 66}")
        return

    # Interactive loop
    print("Yale Residential College Unofficial Guide — interactive mode")
    print("Type your question and press Enter. Ctrl-C or empty input to quit.\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not question:
            break
        result = ask(question, model=model, collection=collection)
        _print_result(question, result)


if __name__ == "__main__":
    main()
