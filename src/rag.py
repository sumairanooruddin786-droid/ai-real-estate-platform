"""RAG knowledge base: read documents, split them into chunks, embed them and search.

Vector store:
  * ChromaDB with its default sentence-embedding model when `chromadb` is installed (preferred).
  * A TF-IDF store as an automatic fallback so the app still runs without it.

Run `python -m src.rag` to index the starter documents in data/knowledge/.
"""
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import KNOWLEDGE_DIR, UPLOAD_DIR, VECTOR_DIR

SUPPORTED = {".pdf", ".txt", ".md"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int


# --------------------------------------------------------------------------- reading + chunking
def read_document(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text)]. Text and Markdown files count as page 1."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{ext}'. Upload a PDF, TXT or MD file.")
    if ext == ".pdf":
        from pypdf import PdfReader
        pages = []
        for i, page in enumerate(PdfReader(str(path)).pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((i, text))
        return pages
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [(1, text)] if text else []


def chunk_text(text: str, source: str, page: int, size: int = 160, overlap: int = 30) -> list[Chunk]:
    words = re.sub(r"\s+", " ", text).split(" ")
    step = max(1, size - overlap)
    chunks = []
    for n, start in enumerate(range(0, len(words), step)):
        piece = " ".join(words[start:start + size]).strip()
        if len(piece.split()) >= 12 or (n == 0 and piece):
            chunks.append(Chunk(f"{source}::p{page}::{n}", piece, source, page))
        if start + size >= len(words):
            break
    return chunks


# --------------------------------------------------------------------------- stores
class TfidfStore:
    name = "TF-IDF keyword search (fallback)"
    min_score = 0.08

    def __init__(self):
        self.path = VECTOR_DIR / "chunks.json"
        self.chunks: list[Chunk] = []
        if self.path.exists():
            self.chunks = [Chunk(**c) for c in json.loads(self.path.read_text())]
        self._vec = None
        self._matrix = None

    def _save(self):
        self.path.write_text(json.dumps([c.__dict__ for c in self.chunks]))
        self._vec = self._matrix = None

    def upsert(self, chunks: list[Chunk]):
        ids = {c.id for c in chunks}
        self.chunks = [c for c in self.chunks if c.id not in ids] + chunks
        self._save()

    def has_source(self, source: str) -> bool:
        return any(c.source == source for c in self.chunks)

    def delete_source(self, source: str):
        self.chunks = [c for c in self.chunks if c.source != source]
        self._save()

    def sources(self) -> dict:
        out: dict = {}
        for c in self.chunks:
            out[c.source] = out.get(c.source, 0) + 1
        return out

    def search(self, query: str, k: int = 4) -> list[dict]:
        if not self.chunks:
            return []
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel
        if self._vec is None:
            self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            self._matrix = self._vec.fit_transform([c.text for c in self.chunks])
        scores = linear_kernel(self._vec.transform([query]), self._matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [{"text": self.chunks[i].text, "source": self.chunks[i].source,
                 "page": self.chunks[i].page, "score": float(scores[i])} for i in order]


class ChromaStore:
    name = "ChromaDB with sentence embeddings"
    min_score = 0.20

    def __init__(self):
        import chromadb
        client = chromadb.PersistentClient(path=str(VECTOR_DIR / "chroma"))
        self.col = client.get_or_create_collection("real_estate_docs", metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: list[Chunk]):
        if chunks:
            self.col.upsert(ids=[c.id for c in chunks], documents=[c.text for c in chunks],
                            metadatas=[{"source": c.source, "page": c.page} for c in chunks])

    def has_source(self, source: str) -> bool:
        return len(self.col.get(where={"source": source}, limit=1)["ids"]) > 0

    def delete_source(self, source: str):
        self.col.delete(where={"source": source})

    def sources(self) -> dict:
        out: dict = {}
        for m in self.col.get(include=["metadatas"])["metadatas"]:
            out[m["source"]] = out.get(m["source"], 0) + 1
        return out

    def search(self, query: str, k: int = 4) -> list[dict]:
        if self.col.count() == 0:
            return []
        res = self.col.query(query_texts=[query], n_results=min(k, self.col.count()))
        return [{"text": d, "source": m["source"], "page": m["page"], "score": float(1 - dist)}
                for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]


def make_store():
    try:
        return ChromaStore()
    except Exception as exc:  # chromadb missing or its embedding model could not be loaded
        print(f"[rag] ChromaDB unavailable ({type(exc).__name__}); using TF-IDF fallback.")
        return TfidfStore()


# --------------------------------------------------------------------------- service
class RAGService:
    def __init__(self, store=None):
        self.store = store or make_store()
        self.ingest_directory(KNOWLEDGE_DIR)
        self.ingest_directory(UPLOAD_DIR)

    @property
    def backend(self) -> str:
        return self.store.name

    def ingest_directory(self, folder: Path):
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() in SUPPORTED and not self.store.has_source(path.name):
                try:
                    self.ingest_file(path)
                except Exception as exc:
                    print(f"[rag] skipped {path.name}: {exc}")

    def ingest_file(self, path: Path) -> int:
        pages = read_document(path)
        chunks = [c for page, text in pages for c in chunk_text(text, path.name, page)]
        if not chunks:
            raise ValueError("No readable text found. Scanned PDFs need OCR before they can be searched.")
        if self.store.has_source(path.name):
            self.store.delete_source(path.name)
        self.store.upsert(chunks)
        return len(chunks)

    def add_upload(self, filename: str, data: bytes) -> int:
        safe = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(filename).name)
        if Path(safe).suffix.lower() not in SUPPORTED:
            raise ValueError("Unsupported file type. Upload a PDF, TXT or MD file.")
        if not data:
            raise ValueError("The file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("The file is larger than 10 MB.")
        dest = UPLOAD_DIR / safe
        dest.write_bytes(data)
        try:
            return self.ingest_file(dest)
        except Exception:
            dest.unlink(missing_ok=True)
            raise

    def remove(self, source: str):
        target = UPLOAD_DIR / Path(source).name
        if not target.exists():
            raise ValueError("Only documents you uploaded can be removed.")
        target.unlink()
        self.store.delete_source(source)

    def documents(self) -> list[dict]:
        uploaded = {p.name for p in UPLOAD_DIR.glob("*")}
        return [{"name": n, "chunks": c, "uploaded": n in uploaded} for n, c in sorted(self.store.sources().items())]

    def search(self, query: str, k: int = 4) -> list[dict]:
        hits = self.store.search(query, k)
        return [h for h in hits if h["score"] >= self.store.min_score]


if __name__ == "__main__":
    svc = RAGService()
    print(f"Backend: {svc.backend}")
    for d in svc.documents():
        print(f"  {d['name']}: {d['chunks']} chunks")
