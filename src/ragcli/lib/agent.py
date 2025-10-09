import textwrap
from typing import List
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import Config
from .store import VectorStore, Document
from .embeddings import EmbeddingService
from .log import get_logger

logger = get_logger(__name__)


class QueryAgent:
    """RAG query agent using Pydantic AI."""

    def __init__(self, config: Config, vector_store: VectorStore):
        self.config = config
        self.vector_store = vector_store

        # Setup Pydantic AI agent
        provider = OpenAIProvider(
            base_url=config.openai_base_url,
            api_key=config.openai_api_key or "dummy-key",
        )
        model = OpenAIChatModel(config.llm_model, provider=provider)

        self.agent = Agent(
            model,
            system_prompt=textwrap.dedent("""\
                You are a helpful assistant that answers questions based on the provided context.
                Use only the information from the context to answer the question.
                If the context doesn't contain enough information to answer the question, say so.
                Be concise and accurate in your responses.
            """),
        )

        self.embedding_service = EmbeddingService(config)
        logger.info(f"initialized query agent with model {config.llm_model}")

    async def query(self, question: str, limit: int = 5) -> str:
        """Answer a question using RAG."""
        try:
            # Generate embedding for the question
            logger.info(f"generating embedding for question: {question[:100]}...")
            query_embedding = self.embedding_service.embed_text(question)

            # Search for relevant documents
            logger.info(f"searching for {limit} relevant documents")
            relevant_docs = self.vector_store.search(query_embedding, limit=limit)

            if not relevant_docs:
                logger.warning("no relevant documents found")
                return "I couldn't find any relevant information to answer your question."

            # Build context from relevant documents
            context = self._build_context(relevant_docs)
            logger.info(f"built context from {len(relevant_docs)} documents")

            # Generate response using Pydantic AI
            prompt = textwrap.dedent(f"""\
                Context information is below.

                ---------------------
                {context}
                ---------------------

                Given the context information, answer the following question.

                Question: {question}

                Answer:""")

            logger.info("generating response with pydantic ai")
            result = await self.agent.run(prompt)
            response = str(result)

            logger.info("successfully generated response")
            return response

        except Exception as e:
            logger.error(f"failed to process query: {e}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}"

    def _build_context(self, documents: List[Document]) -> str:
        """Build context string from documents."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            metadata_str = ", ".join(f"{k}: {v}" for k, v in doc.metadata.items())
            context_parts.append(
                f"Document {i}:\nSource: {doc.metadata.get('source', 'unknown')}\n{metadata_str}\nContent: {doc.text}\n"
            )
        return "\n".join(context_parts)

    def query_sync(self, question: str, limit: int = 5) -> str:
        """Synchronous version of query method."""
        import asyncio

        return asyncio.run(self.query(question, limit))
