from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str


@runtime_checkable
class LLMClient(Protocol):
    def health_check(self) -> bool:
        """Return True when the model endpoint is reachable."""

    def generate(self, messages: list[ChatMessage]) -> LLMResponse:
        """Generate a completion from chat messages."""
