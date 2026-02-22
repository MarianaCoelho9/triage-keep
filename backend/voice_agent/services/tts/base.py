"""Base class for Text-to-Speech services."""
import abc


class BaseTTSModel(abc.ABC):
    """Abstract base class for TTS (Text-to-Speech) services."""
    
    @abc.abstractmethod
    def synthesize(self, text: str) -> bytes:
        """
        Synthesize speech from text.
        
        :param text: Text to convert to speech
        :return: Audio bytes (WAV format)
        """
        pass
