import textwrap
from typing import List
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import Config
from .store import VectorStore, Document
from .embeddings import EmbeddingService
from .log import get_logger

logger = get_logger(__name__)


class SearchTerms(BaseModel):
    """Search terms generated from natural language queries."""
    terms: List[str]


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

        # Setup search term generation agent with structured output
        self.search_agent = Agent(
            model,
            output_type=SearchTerms,
            system_prompt=textwrap.dedent("""\
                You are an expert at converting natural language questions into effective search terms.
                Analyze the user's question and extract 3-5 relevant search terms.

                Focus on semantic meaning rather than exact wording.
                Example terms: "machine learning algorithms", "neural networks", "deep learning models", "AI training"
            """),
        )

        self.embedding_service = EmbeddingService(config)
        logger.info(f"initialized query agent with model {config.llm_model}")

    def _generate_search_terms(self, question: str) -> List[str]:
        """Generate search terms from natural language question."""
        try:
            logger.info(f"generating search terms for question: {question[:100]}...")

            result = self.search_agent.run_sync(question)
            search_terms = result.output.terms

            logger.info(f"generated {len(search_terms)} search terms")
            logger.debug(f"search terms: {search_terms}")
            return search_terms
        except Exception as e:
            logger.error(f"failed to generate search terms: {e}")
            # Fallback to using the question as single search term
            return [question]

    def query(self, question: str, limit: int = 5) -> str:
        """Answer a question using RAG."""
        try:
            # Generate search terms from the natural language question
            search_terms = self._generate_search_terms(question)
            logger.info(f"searching with {len(search_terms)} queries")

            # Generate embeddings for all search queries
            all_embeddings = []
            for query in search_terms:
                embedding = self.embedding_service.embed_text(query)
                all_embeddings.append(embedding)
                logger.debug(f"generated embedding for search query: {query[:50]}...")

            # Search for relevant documents using all embeddings
            all_docs = []
            for embedding in all_embeddings:
                docs = self.vector_store.search(embedding, limit=limit)
                all_docs.extend(docs)

            # Remove duplicates while preserving order
            seen_texts = set()
            relevant_docs = []
            for doc in all_docs:
                if doc.text not in seen_texts:
                    seen_texts.add(doc.text)
                    relevant_docs.append(doc)

            # Limit to requested number of documents
            relevant_docs = relevant_docs[:limit]

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
            result = self.agent.run_sync(prompt)
            response = result.output

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
