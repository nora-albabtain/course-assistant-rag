"""Ingestion pipeline: load course files -> split into chunks -> embed -> FAISS index.

Usage:
    python -m app.ingest            # builds the index from settings.data_dir
"""
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".md", ".txt"}
PDF_SUFFIXES = {".pdf"}


def load_documents(data_dir: str | Path) -> list[Document]:
    """Load every supported file (PDF, Markdown, text) under data_dir."""
    docs: list[Document] = []
    for path in sorted(Path(data_dir).rglob("*")):
        suffix = path.suffix.lower()
        if suffix in PDF_SUFFIXES:
            docs.extend(PyPDFLoader(str(path)).load())
        elif suffix in TEXT_SUFFIXES:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
    logger.info("Loaded %d documents from %s", len(docs), data_dir)
    return docs


def split_documents(docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 120) -> list[Document]:
    """Split documents into overlapping chunks and tag each chunk with an id."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", "، ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = Path(chunk.metadata.get("source", "unknown")).name
    return chunks


def build_index(settings: Settings, embeddings: Embeddings) -> FAISS:
    """Build a FAISS index from the data folder and save it to disk."""
    docs = load_documents(settings.data_dir)
    if not docs:
        raise ValueError(f"No supported files found in '{settings.data_dir}'.")
    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    Path(settings.index_dir).parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(settings.index_dir)
    logger.info("Indexed %d chunks into %s", len(chunks), settings.index_dir)
    return vectorstore


def load_index(settings: Settings, embeddings: Embeddings) -> FAISS:
    """Load a FAISS index that this application created itself."""
    return FAISS.load_local(settings.index_dir, embeddings, allow_dangerous_deserialization=True)


if __name__ == "__main__":
    from app.providers import get_embeddings

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = get_settings()
    build_index(s, get_embeddings(s))
