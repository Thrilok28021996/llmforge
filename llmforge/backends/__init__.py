"""Backend adapters for LLM inference."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from llmforge.domain.models import (
        InferenceRequest,
        ModelDescriptor,
        TokenChunk,
    )


@runtime_checkable
class InferenceBackend(Protocol):
    """Protocol that all inference backends must implement."""

    @property
    def id(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    async def generate(self, request: InferenceRequest) -> AsyncIterator[TokenChunk]:
        """Stream token chunks for the given request. Yields TokenChunk objects."""
        ...

    async def list_models(self) -> list[ModelDescriptor]:
        """List available models from this backend."""
        ...

    async def cancel(self) -> None:
        """Cancel any in-progress inference."""
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...
