"""
PsycheOS — FAISS Memory Store
Cross-session RAG for patient history retrieval.
"""

import os
import json
import hashlib
from typing import List, Optional
from datetime import datetime

try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class MemoryStore:
    """
    Stores and retrieves conversation history using FAISS vector search.
    Falls back to keyword-based retrieval if FAISS/sentence-transformers unavailable.
    """

    def __init__(self, dim: int = 384, index_path: str = "/tmp/psycheos_faiss"):
        self.dim = dim
        self.index_path = index_path
        self.memories = []  # list of {text, distress, timestamp, embedding_idx}
        self.faiss_available = FAISS_AVAILABLE
        self._encoder = None
        self._index = None

        if self.faiss_available:
            try:
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
                self._index = faiss.IndexFlatL2(dim)
            except Exception:
                self.faiss_available = False

    def store(self, message: str, distress_level: int, session: int = 1):
        """Store a message with its distress level and metadata."""
        entry = {
            "text": message,
            "distress": distress_level,
            "session": session,
            "timestamp": datetime.now().isoformat(),
        }

        if self.faiss_available and self._encoder and self._index:
            try:
                emb = self._encoder.encode([message])
                self._index.add(emb)
                entry["embedding_idx"] = self._index.ntotal - 1
            except Exception:
                pass

        self.memories.append(entry)

    def retrieve(self, query: str, k: int = 3) -> str:
        """Retrieve top-k relevant past memories."""
        if not self.memories:
            return ""

        if self.faiss_available and self._encoder and self._index and self._index.ntotal > 0:
            try:
                import numpy as np
                q_emb = self._encoder.encode([query])
                k_actual = min(k, self._index.ntotal)
                distances, indices = self._index.search(q_emb, k_actual)
                retrieved = []
                for idx in indices[0]:
                    if 0 <= idx < len(self.memories):
                        m = self.memories[idx]
                        retrieved.append(f"[Distress {m['distress']}] {m['text'][:80]}")
                return " | ".join(retrieved)
            except Exception:
                pass

        # Keyword fallback
        query_words = set(query.lower().split())
        scored = []
        for m in self.memories[-20:]:
            mem_words = set(m["text"].lower().split())
            overlap = len(query_words & mem_words)
            scored.append((overlap, m))

        scored.sort(reverse=True)
        top = scored[:k]
        if not top:
            return ""

        return " | ".join(
            f"[Distress {m['distress']}] {m['text'][:60]}"
            for _, m in top if m["text"]
        )

    def get_session_summary(self) -> dict:
        """Return a summary of stored memories."""
        if not self.memories:
            return {"total": 0, "avg_distress": 0, "max_distress": 0}
        distress_levels = [m["distress"] for m in self.memories]
        return {
            "total": len(self.memories),
            "avg_distress": round(sum(distress_levels) / len(distress_levels), 2),
            "max_distress": max(distress_levels),
        }
