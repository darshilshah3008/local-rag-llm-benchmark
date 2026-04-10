from typing import List, Tuple, Dict

import faiss
import numpy as np

from embedding import DocumentIndex, embed_text_batch


class Retriever:
    def __init__(self, doc_index: DocumentIndex):
        """
        Stores FAISS index + metadata.
        No local embedding model is loaded because we use Ollama.
        """
        self.doc_index = doc_index

    def retrieve(self, query: str, k: int = 8) -> List[Tuple[Dict, float]]:
        """
        Retrieve top-k most similar chunks to the query.
        Uses Ollama embedding model via embed_text_batch().
        """

        # Use the same embedding model as chunk ingestion
        q_vec = embed_text_batch([query])  # shape (1, dim)

        # Normalize for cosine similarity
        faiss.normalize_L2(q_vec)

        # FAISS search
        scores, idxs = self.doc_index.index.search(q_vec, k)

        scores = scores[0]
        idxs = idxs[0]

        results: List[Tuple[Dict, float]] = []
        for score, idx in zip(scores, idxs):
            if idx == -1:
                continue
            meta = self.doc_index.metadata[idx]
            results.append((meta, float(score)))

        return results
