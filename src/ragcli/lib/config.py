import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_VECTOR_COLLECTION = "ragcli-default-collection"


@dataclass
class Config:
    """Configuration class for RAG CLI system."""

    openai_base_url: str = "http://calyrex-neb.abifog.com:7884/v1/"
    openai_api_key: str | None = None
    llm_model: str = "gpt-oss-20b"
    embedding_model: str = "nomic-embed-text"
    reranker_model: str = "bge-reranker-v2-m3"
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333
    chunk_size: int = 1000
    chunk_overlap: int = 200
    vector_collection: str = DEFAULT_VECTOR_COLLECTION
    search_limit: int = 20
    rerank_limit: int = 5

    @classmethod
    def from_env(cls) -> "Config":
        """Create Config instance from environment variables."""
        # Load .env file from current directory or parent directories
        load_dotenv()

        return cls(
            openai_base_url=os.getenv("OPENAI_BASE_URL", cls.openai_base_url),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            reranker_model=os.getenv("RERANKER_MODEL", cls.reranker_model),
            qdrant_host=os.getenv("QDRANT_HOST", cls.qdrant_host),
            qdrant_port=int(os.getenv("QDRANT_PORT", str(cls.qdrant_port))),
            chunk_size=int(os.getenv("CHUNK_SIZE", str(cls.chunk_size))),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", str(cls.chunk_overlap))),
            vector_collection=os.getenv("VECTOR_COLLECTION", cls.vector_collection),
            search_limit=int(os.getenv("SEARCH_LIMIT", str(cls.search_limit))),
            rerank_limit=int(os.getenv("RERANK_LIMIT", str(cls.rerank_limit))),
        )
