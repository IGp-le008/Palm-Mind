"""Centralized application logging configuration."""

from __future__ import annotations

import logging
from logging.config import dictConfig

from app.core.config import settings


def setup_logging() -> None:
    """Configure application-wide logging.

    Safe to call more than once: dictConfig replaces the previous configuration
    rather than adding duplicate handlers.
    """
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": settings.log_level,
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": settings.log_level,
                "handlers": ["console"],
            },
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the calling module. Pass `__name__`."""
    return logging.getLogger(name)
