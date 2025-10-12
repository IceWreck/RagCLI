from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from .config import Config
from .log import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Text chunk data structure."""

    text: str
    metadata: Dict[str, Any]
    source: str
    chunk_index: int


class TextChunker:
    """Text chunking utility for splitting documents into manageable pieces."""

    def __init__(self, config: Config):
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        logger.info(f"initialized chunker with size {self.chunk_size} and overlap {self.chunk_overlap}")

    def chunk_file(self, file_path: Path) -> List[Chunk]:
        """Chunk a single file into text chunks."""
        if not file_path.exists():
            raise FileNotFoundError(f"file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            logger.info(f"read file {file_path}, length: {len(content)} characters")
            return self.chunk_text(content, str(file_path))
        except Exception as e:
            logger.error(f"failed to read file {file_path}: {e}")
            raise

    def chunk_text(self, text: str, source: str) -> List[Chunk]:
        """Split text into chunks with overlap."""
        if not text.strip():
            logger.warning("empty text provided for chunking")
            return []

        # Normalize whitespace
        text = text.strip()
        if len(text) <= self.chunk_size:
            logger.info(f"text length {len(text)} is within chunk size, creating single chunk")
            return [Chunk(text=text, metadata={"length": len(text)}, source=source, chunk_index=0)]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            # If this is the last chunk, take everything remaining
            if end >= len(text):
                chunk_text = text[start:]
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={"length": len(chunk_text), "start_char": start, "end_char": len(text)},
                        source=source,
                        chunk_index=chunk_index,
                    )
                )
                break

            # Try to break at word boundary
            chunk_text = text[start:end]
            last_space = chunk_text.rfind(" ")
            last_newline = chunk_text.rfind("\n")

            # Prefer breaking at newline, then space
            break_point = max(last_newline, last_space)

            if break_point > 0 and break_point > self.chunk_size * 0.8:
                # Adjust chunk to break at word boundary
                actual_end = start + break_point + 1
                chunk_text = text[start:actual_end]
            else:
                actual_end = end

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={"length": len(chunk_text), "start_char": start, "end_char": actual_end},
                    source=source,
                    chunk_index=chunk_index,
                )
            )

            # Calculate next start position with overlap
            start = actual_end - self.chunk_overlap
            if start < 0:
                start = 0
            chunk_index += 1

        logger.info(f"created {len(chunks)} chunks from {len(text)} characters")
        return chunks

    def chunk_files(self, file_paths: List[Path]) -> List[Chunk]:
        """Chunk multiple files."""
        all_chunks = []
        for file_path in file_paths:
            try:
                chunks = self.chunk_file(file_path)
                all_chunks.extend(chunks)
                logger.info(f"processed {file_path}, created {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"failed to process file {file_path}: {e}")
                continue

        logger.info(f"total chunks created: {len(all_chunks)}")
        return all_chunks
