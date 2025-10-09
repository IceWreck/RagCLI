import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Disable httpx logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx._client").setLevel(logging.WARNING)
    logging.getLogger("httpx._transports.default").setLevel(logging.WARNING)

    return logging.getLogger(name)
