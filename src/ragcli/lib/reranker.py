import httpx
from abc import ABC, abstractmethod

from .config import Config
from .store import Document
from .log import get_logger

logger = get_logger(__name__)


class BaseReranker(ABC):
    """Abstract base class for document rerankers."""

    @abstractmethod
    def rerank(self, query: str, documents: list[Document], limit: int) -> list[Document]:
        """Rerank documents based on relevance to query."""
        pass


class JinaReranker(BaseReranker):
    """Jina-style reranker implementation for llama.cpp reranker API."""

    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.openai_base_url.rstrip("/")
        self.model = config.reranker_model
        self.client = httpx.Client(timeout=30.0)
        logger.info(f"initialized jina reranker with model {self.model} at {self.base_url}")

    def rerank(self, query: str, documents: list[Document], limit: int) -> list[Document]:
        """Rerank documents using llama.cpp Jina-style API."""
        if not documents:
            logger.warning("no documents to rerank")
            return documents

        try:
            # Prepare API request
            endpoint = f"{self.base_url}/v1/rerank"
            texts = [doc.text for doc in documents]

            payload = {
                "model": self.model,
                "query": query,
                "top_n": min(limit, len(documents)),
                "documents": texts,
            }

            logger.info(f"reranking {len(documents)} documents with limit {limit}")
            logger.debug(f"reranker endpoint: {endpoint}")
            logger.debug(f"reranker payload: {payload}")

            # Make API request
            response = self.client.post(endpoint, json=payload)
            response.raise_for_status()

            result = response.json()
            logger.debug(f"reranker response: {result}")

            # Parse reranking results
            if "results" not in result:
                logger.warning("no results in reranker response")
                return documents[:limit]

            reranked_docs: list[Document] = []
            for item in result["results"]:
                index = item.get("index")
                if isinstance(index, int) and 0 <= index < len(documents):
                    selected_doc = documents[index]
                    reranked_docs.append(selected_doc)

            # Fill in remaining docs if fewer results than requested
            if len(reranked_docs) < limit and len(documents) > len(reranked_docs):
                seen_ids: set[str] = {doc.id for doc in reranked_docs if doc.id is not None}
                for doc in documents:
                    if len(reranked_docs) >= limit:
                        break
                    if doc.id is not None and doc.id not in seen_ids:
                        reranked_docs.append(doc)
                        seen_ids.add(doc.id)

            logger.info(f"reranked to {len(reranked_docs)} documents")
            return reranked_docs[:limit]

        except httpx.HTTPStatusError as e:
            logger.error(f"reranker http error: {e.response.status_code} - {e.response.text}")
            logger.info("falling back to original document order")
            return documents[:limit]
        except httpx.RequestError as e:
            logger.error(f"reranker request error: {e}")
            logger.info("falling back to original document order")
            return documents[:limit]
        except Exception as e:
            logger.error(f"reranker error: {e}")
            logger.info("falling back to original document order")
            return documents[:limit]

    def __del__(self) -> None:
        """Cleanup HTTP client."""
        if hasattr(self, "client"):
            self.client.close()
