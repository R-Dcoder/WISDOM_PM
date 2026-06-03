"""
rag/vector_store.py
ChromaDB-backed vector store for analyst PDFs and concall transcripts.
The LLM retrieves qualitative signals from here — never computes numbers from it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import List, Optional

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from config import SAMPLE_ANALYST_DOCS


@dataclass
class RetrievedChunk:
    source: str
    ticker: str
    date: str
    text: str
    distance: float
    doc_id: str


class WisdomVectorStore:
    """
    Manages a persistent ChromaDB collection.
    Documents are chunked analyst reports and concall transcripts.
    Used exclusively by Agent 2 (Qualitative Researcher).
    """

    COLLECTION_NAME = "wisdom_analyst_docs"

    def __init__(self, persist_dir: str = ".wisdom_chroma"):
        self._fallback_docs: list[dict] = SAMPLE_ANALYST_DOCS
        self.client = None
        self.collection = None

        if not CHROMA_AVAILABLE:
            return

        try:
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._seed_sample_docs()
        except Exception:
            # Graceful fallback — keyword search works without the embedding model
            self.client = None
            self.collection = None

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def add_document(self, doc_id: str, text: str, metadata: dict) -> None:
        if self.collection is None:
            return
        chunks = self._chunk(text, size=512, overlap=64)
        ids, texts, metas = [], [], []
        for i, chunk in enumerate(chunks):
            cid = f"{doc_id}_chunk{i}"
            ids.append(cid)
            texts.append(chunk)
            metas.append({**metadata, "chunk_index": i})
        try:
            self.collection.upsert(ids=ids, documents=texts, metadatas=metas)
        except Exception:
            pass

    def add_pdf_text(self, pdf_text: str, ticker: str, source: str, date: str) -> int:
        doc_id = hashlib.md5(f"{ticker}{source}{date}".encode()).hexdigest()[:12]
        self.add_document(
            doc_id=doc_id,
            text=pdf_text,
            metadata={"ticker": ticker, "source": source, "date": date},
        )
        return doc_id

    def _seed_sample_docs(self) -> None:
        """Load built-in analyst note samples if collection is empty."""
        if self.collection is None:
            return
        try:
            count = self.collection.count()
            if count == 0:
                for doc in SAMPLE_ANALYST_DOCS:
                    self.add_document(
                        doc_id=doc["id"],
                        text=doc["text"],
                        metadata={
                            "ticker": doc["ticker"],
                            "source": doc["source"],
                            "date":   doc["date"],
                        },
                    )
        except Exception:
            pass

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def query(self, query_text: str, ticker: Optional[str] = None, n_results: int = 4) -> List[RetrievedChunk]:
        """Semantic search. Falls back to keyword match if ChromaDB unavailable."""
        if self.collection is not None:
            return self._chroma_query(query_text, ticker, n_results)
        return self._fallback_query(query_text, ticker, n_results)

    def _chroma_query(self, query: str, ticker: Optional[str], n: int) -> List[RetrievedChunk]:
        where = {"ticker": ticker} if ticker else None
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n, self.collection.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            chunks = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunks.append(RetrievedChunk(
                    source=meta.get("source", ""),
                    ticker=meta.get("ticker", ""),
                    date=meta.get("date", ""),
                    text=doc,
                    distance=round(dist, 4),
                    doc_id=meta.get("doc_id", ""),
                ))
            return chunks
        except Exception as e:
            return self._fallback_query(query, ticker, n)

    def _fallback_query(self, query: str, ticker: Optional[str], n: int) -> List[RetrievedChunk]:
        """Simple keyword overlap fallback when ChromaDB is unavailable."""
        query_words = set(query.lower().split())
        scored = []
        for doc in self._fallback_docs:
            if ticker and doc["ticker"] != ticker:
                continue
            overlap = len(query_words & set(doc["text"].lower().split()))
            scored.append((overlap, doc))
        scored.sort(key=lambda x: -x[0])
        return [
            RetrievedChunk(
                source=d["source"], ticker=d["ticker"], date=d["date"],
                text=d["text"], distance=round(1 - s / max(len(query_words), 1), 3),
                doc_id=d["id"],
            )
            for s, d in scored[:n]
        ]

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk(text: str, size: int = 512, overlap: int = 64) -> List[str]:
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunks.append(" ".join(words[i: i + size]))
            i += size - overlap
        return chunks

    def collection_size(self) -> int:
        if self.collection:
            try:
                return self.collection.count()
            except Exception:
                pass
        return len(self._fallback_docs)
