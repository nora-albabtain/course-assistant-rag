"""Factories for the embedding model and the chat LLM.

Imports are done lazily so the heavy libraries (PyTorch / sentence-transformers)
are only loaded when actually needed.
"""
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.config import Settings


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.llm_base_url,
        )

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_llm(settings: Settings) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key or "not-needed",  # local servers (e.g. Ollama) ignore the key
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
    )
