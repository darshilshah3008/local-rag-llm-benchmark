import os
import json
import numpy as np
import faiss
import requests
from dataclasses import dataclass
from typing import List, Dict

from utils import INDEX_DIR, EMBEDDING_URL, EMBEDDING_MODEL
from ingest import collect_all_chunks


@dataclass
class DocumentIndex:
    index: faiss.IndexFlatIP
    metadata: List[Dict]


def embed_text_batch(texts: List[str]) -> np.ndarray:
    """
    Get embeddings from Ollama (nomic-embed-text or similar).
    Ollama embedding endpoint: POST /api/embed
    """
    response = requests.post(
        EMBEDDING_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": texts
        },
        timeout=300
    )
    response.raise_for_status()
    data = response.json()

    embeddings = np.array(data["embeddings"], dtype="float32")
    return embeddings


def ingest_and_build_index() -> DocumentIndex:
    """
    Extract chunks, embed them, and build FAISS index.
    Works with NotebookLM-style chunks produced by ingest.py.
    """
    os.makedirs(INDEX_DIR, exist_ok=True)

    chunks = collect_all_chunks()

    # *** FIXED: use "chunk_text" instead of "text" ***
    texts = [c["chunk_text"] for c in chunks]

    print(f"\nUsing Ollama embedding model: {EMBEDDING_MODEL}")
    print(f"Sending {len(texts)} chunks for embedding...")

    embeddings = embed_text_batch(texts)

    # Normalize for cosine similarity (FAISS IP = cosine when normalized)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save FAISS index and metadata
    faiss_path = os.path.join(INDEX_DIR, "faiss.index")
    metadata_path = os.path.join(INDEX_DIR, "metadata.json")

    faiss.write_index(index, faiss_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("\n[OK] Saved FAISS index and metadata.")
    return DocumentIndex(index=index, metadata=chunks)


def load_index() -> DocumentIndex:
    """
    Load FAISS index and chunk metadata created previously.
    """
    index_path = os.path.join(INDEX_DIR, "faiss.index")
    metadata_path = os.path.join(INDEX_DIR, "metadata.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError("FAISS index missing. Run ingest first.")

    index = faiss.read_index(index_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"[OK] Loaded index with {len(metadata)} chunks.")
    return DocumentIndex(index=index, metadata=metadata)
