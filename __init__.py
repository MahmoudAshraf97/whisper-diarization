"""
Whisper Diarization Library

A comprehensive speech transcription and diarization library built on
Whisper and NeMo MSDD models.

Example:
    >>> from processor import SpeechProcessor
    >>> processor = SpeechProcessor(device='cuda')
    >>> result = processor.diarize('audio.mp3')
    >>> print(result.to_txt())
"""

from models import (
    DiarizationResult,
    Segment,
    SpeechTimestamp,
    TranscriptionResult,
    WordTimestamp,
)
from preprocessor import AudioPreprocessor
from processor import SpeechProcessor

__version__ = "2.0.0"
__all__ = [
    "SpeechProcessor",
    "AudioPreprocessor",
    "TranscriptionResult",
    "DiarizationResult",
    "WordTimestamp",
    "SpeechTimestamp",
    "Segment",
]

