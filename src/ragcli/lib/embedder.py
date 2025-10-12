import openai

from .config import Config
from .chunker import Chunk
from .log import get_logger

logger = get_logger(__name__)


class Embedder:
    """Service for generating text embeddings using OpenAI."""

    def __init__(self, config: Config):
        self.config = config
        self.client = openai.OpenAI(
            base_url=config.openai_base_url,
            api_key=config.openai_api_key or "dummy-key",
        )
        self.model = config.embedding_model
        logger.info(f"initialized embedding service with model {self.model}")

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            embedding = response.data[0].embedding
            logger.debug(f"generated embedding for text length {len(text)}, size {len(embedding)}")
            logger.debug(f"vector preview: {embedding[:5]}...{embedding[-5:]}")
            logger.debug(
                f"vector stats: min={min(embedding):.6f}, max={max(embedding):.6f}, mean={sum(embedding) / len(embedding):.6f}"
            )
            return embedding
        except Exception as e:
            logger.error(f"failed to generate embedding: {e}")
            raise

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in batch."""
        if not texts:
            return []

        try:
            # Process in batches to avoid rate limits
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                logger.debug(f"processing batch {i // batch_size + 1}, size {len(batch)}")

                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )

                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

            logger.info(f"generated embeddings for {len(texts)} texts")
            return all_embeddings
        except Exception as e:
            logger.error(f"failed to generate embeddings: {e}")
            raise

    def embed_chunks(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        """Generate embeddings for text chunks."""
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)

        result = []
        for chunk, embedding in zip(chunks, embeddings):
            result.append((chunk, embedding))

        logger.info(f"embedded {len(chunks)} chunks")
        return result
