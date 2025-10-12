from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from .config import Config
from .log import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    """Document data structure."""

    text: str
    metadata: dict[str, Any]
    id: str | None = None


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def upsert_with_vectors(self, documents: list[Document], vectors: list[list[float]]) -> list[str]:
        """Add documents with their vectors to the vector store."""
        pass

    @abstractmethod
    def search(self, query_vector: list[float], limit: int = 5) -> list[Document]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def create_collection(self, collection_name: str, vector_size: int) -> None:
        """Create a new collection."""
        pass

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        pass


class QdrantStore(VectorStore):
    """Qdrant implementation of VectorStore."""

    def __init__(self, config: Config, collection_name: str):
        self.config = config
        self.collection_name = collection_name
        self.client = QdrantClient(
            host=config.qdrant_host,
            port=config.qdrant_port,
        )
        logger.info(f"initialized qdrant client at {config.qdrant_host}:{config.qdrant_port}")

    def create_collection(self, collection_name: str, vector_size: int) -> None:
        """Create a new collection in Qdrant."""
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"created collection {collection_name}")
        except Exception as e:
            logger.error(f"failed to create collection {collection_name}: {e}")
            raise

    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists in Qdrant."""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            logger.error(f"failed to check collection existence: {e}")
            return False

    def upsert_with_vectors(self, documents: list[Document], vectors: list[list[float]]) -> list[str]:
        """Upsert documents with their vectors to Qdrant."""
        if len(documents) != len(vectors):
            raise ValueError("number of documents and vectors must match")

        if not documents:
            logger.warning("no documents provided for upsert")
            return []

        # Validate vectors
        if not vectors or any(not v for v in vectors):
            raise ValueError("vectors cannot be empty")

        # Check all vectors have the same dimension
        vector_size = len(vectors[0])
        if any(len(v) != vector_size for v in vectors):
            raise ValueError("all vectors must have the same dimension")

        if not self.collection_exists(self.collection_name):
            self.create_collection(self.collection_name, vector_size)

        points = []
        for i, (doc, vector) in enumerate(zip(documents, vectors)):
            point_id = doc.id or str(i)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"text": doc.text, **doc.metadata},
                )
            )

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"successfully upserted {len(documents)} documents")
            return [str(point.id) for point in points]
        except Exception as e:
            logger.error(f"failed to upsert documents: {e}")
            raise

    def search(self, query_vector: list[float], limit: int = 5) -> list[Document]:
        """Search for similar documents in Qdrant."""
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
            )

            documents = []
            for hit in search_result:
                if hit.payload is None:
                    logger.warning(f"hit {hit.id} has no payload, skipping")
                    continue

                text = hit.payload.get("text", "")
                metadata = {k: v for k, v in hit.payload.items() if k != "text"}

                documents.append(
                    Document(
                        text=text,
                        metadata=metadata,
                        id=str(hit.id),
                    )
                )

            logger.info(f"found {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"failed to search documents: {e}")
            raise

    def delete_collection(self) -> None:
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"deleted collection {self.collection_name}")
        except Exception as e:
            logger.error(f"failed to delete collection: {e}")
            raise
