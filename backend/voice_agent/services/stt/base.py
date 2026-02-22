"""Base class for Speech-to-Text services."""
import abc


class BaseSTTModel(abc.ABC):
    """Abstract base class for STT (Speech-to-Text) services."""
    
    @abc.abstractmethod
    def transcribe(self, audio_chunk: bytes) -> str:
        """
        Transcribe audio input to text.
        
        :param audio_chunk: Raw audio bytes or numpy array
        :return: Transcribed text
        """
        pass
