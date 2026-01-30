# agent/rag/retrieval.py
import os
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DocChunk:
    doc_id: str        # e.g. "marketing_calendar::chunk0"
    text: str
    source: str        # filename, e.g. "marketing_calendar.md"


class SimpleTfidfRetriever:
    def __init__(self, docs_dir: str = "docs", chunk_separator: str = "\n\n"):
        self.docs_dir = docs_dir
        self.chunk_separator = chunk_separator
        self.chunks: List[DocChunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    def load_docs(self):
        """Read all .md files in docs/ and create small chunks."""
        for fname in os.listdir(self.docs_dir):
            if not fname.endswith(".md"):
                continue
            path = os.path.join(self.docs_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            raw_chunks = [c.strip() for c in text.split(self.chunk_separator) if c.strip()]
            for i, chunk in enumerate(raw_chunks):
                doc_id = f"{fname.replace('.md', '')}::chunk{i}"
                self.chunks.append(DocChunk(doc_id=doc_id, text=chunk, source=fname))

    def build_index(self):
        """Fit TF-IDF on chunk texts."""
        texts = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(texts)

    def init(self):
        self.load_docs()
        self.build_index()

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[DocChunk, float]]:
        """Return top-k chunks with similarity score."""
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("Retriever not initialized. Call init() first.")

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        idxs = sims.argsort()[::-1][:k]

        results: List[Tuple[DocChunk, float]] = []
        for idx in idxs:
            results.append((self.chunks[idx], float(sims[idx])))
        return results
