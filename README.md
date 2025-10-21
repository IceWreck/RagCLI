# RAG CLI System

A modular CLI-based Retrieval-Augmented Generation (RAG) system for ingesting documents and querying them using vector search and LLMs.

## Quick Start

```bash
make develop
# Configure .env with OPENAI_API_KEY
make qdrant-start

# Ingest documents
uv run ragcli insert file.txt --collection mydocs

# Query with reranking
uv run ragcli query "your question" --collection mydocs --rerank true
```

The system automatically generates optimal search queries and can rerank results using advanced models for improved precision.

## Configuration

The system is fully configurable through environment variables (`OPENAI_API_KEY`, `QDRANT_URL`, `JINA_API_KEY`) or CLI arguments. Key settings include embedding models, reranking limits, and custom API endpoints. All features have sensible defaults, making it easy to get started while remaining flexible for advanced use cases.

## Development

Use `make format`, `make lint`, and `make typecheck` for code quality. Qdrant can be managed with `make qdrant-start/stop/remove`. Dependencies are managed with `uv add package-name`. The codebase follows modern Python conventions with full type safety and comprehensive error handling.
