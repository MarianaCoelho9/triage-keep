"""Common type definitions for voice agent."""
from typing import Union, AsyncIterator, List, Dict

# Type aliases for clarity
AudioChunk = Union[bytes, str]
AudioStream = AsyncIterator[AudioChunk]
ChatMessage = Dict[str, str]  # {"role": "user"|"assistant", "content": "text"}
ChatHistory = List[ChatMessage]
