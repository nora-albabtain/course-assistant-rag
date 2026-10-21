"""Application settings, loaded from environment variables or a .env file."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Data & index
    data_dir: str = "data"
    index_dir: str = "storage/faiss_index"

    # Embeddings: "huggingface" (free, local, multilingual) or "openai"
    embedding_provider: str = "huggingface"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # LLM: any OpenAI-compatible endpoint (OpenAI, Azure-compatible gateways, Ollama, vLLM...)
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    openai_api_key: str | None = None
    llm_temperature: float = 0.1

    # Retrieval
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
