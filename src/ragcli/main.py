import typer
import uuid
from pathlib import Path
from typing import List, Optional

from .lib.config import Config, DEFAULT_VECTOR_COLLECTION
from .lib.store import QdrantStore, Document
from .lib.chunker import TextChunker
from .lib.embedder import Embedder
from .lib.agent import QueryAgent
from .lib.log import get_logger

logger = get_logger(__name__)
app = typer.Typer(help="RAG CLI - Insert and query documents using vector search.")


@app.command()
def insert(
    files: List[Path] = typer.Argument(..., help="List of files to ingest"),
    collection: str = typer.Option(DEFAULT_VECTOR_COLLECTION, "--collection", "-c", help="Collection name"),
    chunk_size: Optional[int] = typer.Option(None, "--chunk-size", help="Chunk size in characters"),
    chunk_overlap: Optional[int] = typer.Option(None, "--chunk-overlap", help="Chunk overlap in characters"),
) -> None:
    """Insert documents into the vector store."""
    try:
        config = Config.from_env()

        # Override config with CLI args
        if chunk_size:
            config.chunk_size = chunk_size
        if chunk_overlap:
            config.chunk_overlap = chunk_overlap

        logger.info(f"starting ingestion for {len(files)} files")

        # Initialize components
        chunker = TextChunker(config)
        embedder = Embedder(config)
        vector_store = QdrantStore(config, collection)

        # Process files
        chunks = chunker.chunk_files(files)
        if not chunks:
            logger.warning("no chunks created from files")
            return

        logger.info(f"created {len(chunks)} chunks")

        # Generate embeddings
        embedded_chunks = embedder.embed_chunks(chunks)

        # Convert to Document objects
        documents: List[Document] = []
        vectors: List[List[float]] = []
        for chunk, embedding in embedded_chunks:
            doc = Document(
                text=chunk.text,
                metadata={
                    **chunk.metadata,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                },
                id=str(uuid.uuid4()),  # Use UUID
            )
            documents.append(doc)
            vectors.append(embedding)

        # Store in vector database
        vector_store.upsert_with_vectors(documents, vectors)

        logger.info(f"successfully inserted {len(documents)} documents into collection '{collection}'")
        typer.echo(f"✅ Successfully inserted {len(documents)} documents")

    except Exception as e:
        logger.error(f"ingestion failed: {e}")
        typer.echo(f"❌ Ingestion failed: {e}")
        raise typer.Exit(1)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    collection: str = typer.Option(DEFAULT_VECTOR_COLLECTION, "--collection", "-c", help="Collection name"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of documents to retrieve"),
) -> None:
    """Ask a question using RAG."""
    try:
        config = Config.from_env()

        logger.info(f"processing query: {question[:100]}...")

        # Initialize components
        vector_store = QdrantStore(config, collection)
        agent = QueryAgent(config, vector_store)

        # Check if collection exists
        if not vector_store.collection_exists(collection):
            logger.error(f"collection '{collection}' does not exist")
            typer.echo(f"❌ Collection '{collection}' does not exist. Run 'insert' command first.")
            raise typer.Exit(1)

        # Process query
        response = agent.query(question, limit)

        typer.echo(f"\n🤖 Answer:\n{response}")

    except Exception as e:
        logger.error(f"query failed: {e}")
        typer.echo(f"❌ Query failed: {e}")
        raise typer.Exit(1)


@app.command()
def status(
    collection: str = typer.Option(DEFAULT_VECTOR_COLLECTION, "--collection", "-c", help="Collection name"),
) -> None:
    """Check system status."""
    try:
        config = Config.from_env()
        vector_store = QdrantStore(config, collection)

        typer.echo("🔍 System Status:")
        typer.echo(f"Qdrant: {config.qdrant_host}:{config.qdrant_port}")
        typer.echo(f"LLM Model: {config.llm_model}")
        typer.echo(f"Embedding Model: {config.embedding_model}")
        typer.echo(f"OpenAI Endpoint: {config.openai_base_url}")

        if vector_store.collection_exists(collection):
            typer.echo(f"✅ Collection '{collection}' exists")
        else:
            typer.echo(f"❌ Collection '{collection}' does not exist")

    except Exception as e:
        logger.error(f"status check failed: {e}")
        typer.echo(f"❌ Status check failed: {e}")
        raise typer.Exit(1)


@app.command()
def delete_collection(
    collection: str = typer.Option(DEFAULT_VECTOR_COLLECTION, "--collection", "-c", help="Collection name"),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm deletion"),
) -> None:
    """Delete a collection."""
    if not confirm:
        typer.echo("⚠️  This will delete all data in the collection. Use --confirm to proceed.")
        raise typer.Exit(1)

    try:
        config = Config.from_env()
        vector_store = QdrantStore(config, collection)

        vector_store.delete_collection()
        typer.echo(f"✅ Collection '{collection}' deleted")

    except Exception as e:
        logger.error(f"collection deletion failed: {e}")
        typer.echo(f"❌ Deletion failed: {e}")
        raise typer.Exit(1)


def main() -> None:
    app()
