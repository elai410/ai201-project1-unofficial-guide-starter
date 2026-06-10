#!/usr/bin/env python3
"""
Stage 3 + 4: Embedding, Vector Store, and Retrieval
Yale Residential College Unofficial Guide

Pipeline position:
  Ingestion -> Chunking -> [Embedding + ChromaDB] -> [Retrieval] -> Generation

Spec (planning.md § Retrieval Approach):
  Embedding model : all-MiniLM-L6-v2  (sentence-transformers, local, no API key)
  Vector store    : ChromaDB persistent (local disk, documents/chroma/)
  Distance metric : cosine  (lower = more similar)
                    < 0.30 = strong match
                    < 0.50 = acceptable match
                   >= 0.50 = weak match — retrieval may be off-target
  Top-k           : 5

Run:
    python embed.py           # embed chunks + run 3 test queries
    python embed.py --reset   # wipe existing collection and re-embed
"""

import argparse
import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("documents/chunks.json")
CHROMA_DIR  = Path("documents/chroma")
COLLECTION  = "yale_guide"
MODEL_NAME  = "all-MiniLM-L6-v2"
TOP_K       = 5


# ------------------------------------------------------------------------------
# Stage 3 — Embedding + Vector Store
# ------------------------------------------------------------------------------

def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        sys.exit(f"ERROR: {CHUNKS_PATH} not found. Run python ingest.py first.")
    chunks = json.loads(CHUNKS_PATH.read_text())
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")
    return chunks


def build_collection(
    chunks: list[dict],
    model: SentenceTransformer,
    reset: bool = False,
) -> chromadb.Collection:
    """
    Embed all chunks and store them in a persistent ChromaDB collection.

    ChromaDB stores:
      - The raw chunk text  (documents=)
      - The 384-dim embedding vector  (embeddings=)
      - Source metadata per chunk  (metadatas=)
      - A unique string ID per chunk  (ids=)

    hnsw:space = "cosine" tells ChromaDB to use cosine distance rather than
    the default L2 (Euclidean). Cosine is better here because sentence
    embeddings are normalized unit vectors — cosine distance is more meaningful
    than raw Euclidean distance on the unit hypersphere.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION)
            print(f"Deleted collection '{COLLECTION}' for fresh embed.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0 and not reset:
        print(f"Collection '{COLLECTION}' already has {collection.count()} vectors.")
        print("Skipping re-embed. Pass --reset to wipe and re-embed.\n")
        return collection

    # Embed all chunk texts in one batched call
    texts = [c["chunk_text"] for c in chunks]
    print(f"\nEmbedding {len(texts)} chunks with {MODEL_NAME} ...")
    print("(First run downloads ~90 MB model weights — subsequent runs use cache.)\n")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    # IDs must be unique strings; use positional index across the full list
    ids       = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source_title": c["source_title"],
            "source_url":   c["source_url"],
            "chunk_index":  c["chunk_index"],   # position within that document
            "token_count":  c["token_count"],
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=[e.tolist() for e in embeddings],
        documents=texts,
        metadatas=metadatas,
    )
    print(f"\nStored {collection.count()} vectors in ChromaDB -> {CHROMA_DIR}/")
    return collection


# ------------------------------------------------------------------------------
# Stage 4 — Retrieval
# ------------------------------------------------------------------------------

def retrieve(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    k: int = TOP_K,
) -> list[dict]:
    """
    Embed the query with the same model used at index time, then ask ChromaDB
    for the k nearest neighbours by cosine distance.

    Returns a list of dicts:
      chunk_text   : the full text of the retrieved chunk
      source_title : human-readable source name (for attribution)
      source_url   : original URL
      chunk_index  : position of this chunk within its source document
      distance     : cosine distance (0 = identical, 1 = orthogonal)

    Why embed the query with the same model?
    The vector store only makes sense if the query lives in the same embedding
    space as the indexed chunks. Using a different model — or even the same
    model at a different version — would make distance scores meaningless.
    """
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
            "chunk_index":  meta["chunk_index"],
            "distance":     round(dist, 4),
        })
    return hits


# ------------------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------------------

def _distance_label(d: float) -> str:
    if d < 0.30:
        return "STRONG"
    if d < 0.50:
        return "OK    "
    return "WEAK  "


def print_results(query: str, hits: list[dict], expected: str = "") -> None:
    print(f"\n{'=' * 66}")
    print(f"QUERY: {query}")
    print(f"{'=' * 66}")

    for i, hit in enumerate(hits, 1):
        label = _distance_label(hit["distance"])
        print(f"\n  Result {i}  |  distance={hit['distance']}  [{label}]")
        print(f"  Source   : {hit['source_title']}")
        print(f"  chunk_index={hit['chunk_index']}")
        print(f"  {'- ' * 28}")
        # Show up to 500 chars; enough to judge relevance without flooding terminal
        preview = hit["chunk_text"][:500].replace("\n", " ").strip()
        if len(hit["chunk_text"]) > 500:
            preview += " ..."
        print(f"  {preview}")

    if expected:
        print(f"\n  >> Expected: {expected}")


# ------------------------------------------------------------------------------
# Evaluation queries (3 of 5 from planning.md)
# ------------------------------------------------------------------------------

TEST_QUERIES = [
    (
        "Which Yale residential college dining hall ranked last in the 2025 "
        "Yale Daily News dining data study?",
        "Pierson College ranked last.",
    ),
    (
        "How many students requested to transfer residential colleges in 2025, "
        "and approximately what fraction were approved?",
        "72 students requested; nearly three-quarters (~54) were approved.",
    ),
    (
        "What is the most common reason students give when requesting a "
        "residential college transfer?",
        "Wanting to live with friends in a different college (cited in >90% of applications).",
    ),
]


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks and test retrieval.")
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete the existing ChromaDB collection and re-embed from scratch.",
    )
    args = parser.parse_args()

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    chunks     = load_chunks()
    collection = build_collection(chunks, model, reset=args.reset)

    print(f"\n{'=' * 66}")
    print("RETRIEVAL TEST  —  3 of 5 evaluation queries")
    print(f"top-k={TOP_K}  |  cosine distance  |  STRONG<0.30  OK<0.50  WEAK>=0.50")
    print(f"{'=' * 66}")

    for query, expected in TEST_QUERIES:
        hits = retrieve(query, collection, model)
        print_results(query, hits, expected)

    print(f"\n{'=' * 66}")
    print("Done. If top results are STRONG or OK and on-topic, retrieval is working.")
    print("If results are WEAK or off-topic, check chunk size and cleaning.")
    print("Next: python generate.py  (Stage 5 — generation + CLI)")
    print(f"{'=' * 66}")


if __name__ == "__main__":
    main()
