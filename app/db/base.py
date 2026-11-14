"""Declarative base shared by all ORM models. Alembic reads its metadata for migrations."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all ORM models."""
