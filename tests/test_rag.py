"""Tests run fully offline: fake embeddings and a fake LLM replace the real models."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.api import app, get_assistant
from app.ingest import load_documents, split_documents
from app.rag import RAGAssistant, format_docs

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.fixture
def assistant() -> RAGAssistant:
    chunks = split_documents(load_documents(SAMPLE_DIR), chunk_size=400, chunk_overlap=50)
    vectorstore = FAISS.from_documents(chunks, DeterministicFakeEmbedding(size=64))
    llm = FakeListChatModel(responses=["Overfitting means the model memorizes the training data [1]."])
    return RAGAssistant(vectorstore, llm, top_k=3)


def test_sample_documents_load():
    docs = load_documents(SAMPLE_DIR)
    assert len(docs) >= 2
    assert any("Overfitting" in d.page_content for d in docs)
    assert any("فرط التخصيص" in d.page_content for d in docs)


def test_split_adds_metadata():
    docs = [Document(page_content="word " * 500, metadata={"source": "/tmp/notes.md"})]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["source"] == "notes.md" for c in chunks)


def test_format_docs_numbers_sources():
    docs = [
        Document(page_content="A", metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="B", metadata={"source": "b.md"}),
    ]
    text = format_docs(docs)
    assert "[1] (a.pdf, page 1)" in text
    assert "[2] (b.md)" in text


def test_assistant_returns_answer_and_sources(assistant):
    result = assistant.ask("What is overfitting?")
    assert "Overfitting" in result["answer"]
    assert len(result["sources"]) == 3
    assert result["sources"][0]["id"] == 1


def test_api_ask(assistant):
    app.dependency_overrides[get_assistant] = lambda: assistant
    try:
        client = TestClient(app)
        resp = client.post("/ask", json={"question": "ما هو فرط التخصيص؟"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"]
        assert body["sources"]
    finally:
        app.dependency_overrides.clear()


def test_api_rejects_empty_question():
    client = TestClient(app)
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}
