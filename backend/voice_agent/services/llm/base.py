"""Base class for Language Model services."""
import abc
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class GenerationOptions:
    """Optional generation controls shared across LLM backends."""

    stop: Sequence[str] | None = None
    max_new_tokens: int | None = None
    temperature: float | None = None
    json_schema: Mapping[str, Any] | None = None
    grammar: str | None = None
    messages: Sequence[Mapping[str, str]] | None = None


class LLMGenerationError(RuntimeError):
    """Base runtime error for LLM generation failures."""


class LLMContextBudgetExceededError(LLMGenerationError):
    """Raised when prompt + output budget cannot fit model context."""


class LLMDecodeError(LLMGenerationError):
    """Raised when decoder fails to produce output."""


def apply_stop_sequences(text: str, stop: Sequence[str] | None) -> str:
    """Truncate output at the first encountered stop sequence."""
    if not stop:
        return text

    stop_indexes = [text.find(sequence) for sequence in stop if sequence]
    valid_indexes = [index for index in stop_indexes if index >= 0]
    if not valid_indexes:
        return text
    return text[: min(valid_indexes)].strip()


class BaseLLMModel(abc.ABC):
    """Abstract base class for LLM (Language Model) services."""
    
    @abc.abstractmethod
    def generate(self, prompt: str, options: GenerationOptions | None = None) -> str:
        """
        Generate text response from a prompt.
        
        :param prompt: Input text prompt
        :param options: Optional generation controls
        :return: Generated text response
        """
        pass
