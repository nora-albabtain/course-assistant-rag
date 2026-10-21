"""FastAPI service exposing the course assistant."""
import logging
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingest import build_index, load_index
from app.providers import get_embeddings, get_llm
from app.rag import RAGAssistant

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

app = FastAPI(
    title="Course Assistant RAG API",
    description="Answers trainees' questions from course material, in Arabic and English, with citations.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, examples=["What is overfitting?"])


class Source(BaseModel):
    id: int
    source: str
    page: int | None = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


_assistant: RAGAssistant | None = None


def _create_assistant(rebuild: bool = False) -> RAGAssistant:
    settings = get_settings()
    embeddings = get_embeddings(settings)
    if rebuild or not Path(settings.index_dir).exists():
        vectorstore = build_index(settings, embeddings)
    else:
        vectorstore = load_index(settings, embeddings)
    return RAGAssistant(vectorstore, get_llm(settings), settings.top_k)


def get_assistant() -> RAGAssistant:
    global _assistant
    if _assistant is None:
        _assistant = _create_assistant()
    return _assistant


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, assistant: Annotated[RAGAssistant, Depends(get_assistant)]) -> dict:
    return assistant.ask(request.question)


@app.post("/ingest")
def ingest() -> dict:
    """Rebuild the index after adding or changing files in the data folder."""
    global _assistant
    _assistant = _create_assistant(rebuild=True)
    return {"status": "index rebuilt"}
