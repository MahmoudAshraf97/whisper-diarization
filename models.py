"""
Data models for speech processing results.

This module contains dataclasses that represent the various outputs
from speech transcription and diarization operations.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from io import StringIO


@dataclass
class WordTimestamp:
    """Represents a single word with its timing information.
    
    Attributes:
        text: The word text.
        start_ms: Start time in milliseconds.
        end_ms: End time in milliseconds.
    """
    text: str
    start_ms: int
    end_ms: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class SpeechTimestamp:
    """Represents a speech segment with speaker information.
    
    Attributes:
        start_ms: Start time in milliseconds.
        end_ms: End time in milliseconds.
        speaker_id: Numeric speaker identifier.
    """
    start_ms: int
    end_ms: int
    speaker_id: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class Segment:
    """Represents a speech segment with speaker and text.
    
    Attributes:
        speaker: Speaker label (e.g., "Speaker 0").
        text: The transcribed text for this segment.
        start_ms: Start time in milliseconds.
        end_ms: End time in milliseconds.
    """
    speaker: str
    text: str
    start_ms: int
    end_ms: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class TranscriptionResult:
    """Result of audio transcription with word-level timestamps.
    
    Attributes:
        text: Full transcribed text.
        language: Detected or specified language code.
        words: List of words with their timestamps.
    """
    text: str
    language: str
    words: List[WordTimestamp] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary with all fields, words converted to dicts.
        """
        return {
            "text": self.text,
            "language": self.language,
            "words": [w.to_dict() for w in self.words]
        }


@dataclass
class DiarizationResult:
    """Result of audio diarization with speaker-separated segments.
    
    Attributes:
        text: Full transcribed text.
        language: Detected or specified language code.
        segments: List of segments with speaker labels and text.
    """
    text: str
    language: str
    segments: List[Segment] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary with all fields, segments converted to dicts.
        """
        return {
            "text": self.text,
            "language": self.language,
            "segments": [s.to_dict() for s in self.segments]
        }

    def to_srt(self) -> str:
        """Convert to SRT subtitle format.
        
        Returns:
            String in SRT format with speaker labels.
        """
        output = StringIO()
        
        for i, segment in enumerate(self.segments, start=1):
            start_time = self._format_timestamp(segment.start_ms)
            end_time = self._format_timestamp(segment.end_ms)
            text = segment.text.strip().replace('-->', '->')
            
            output.write(f"{i}\n")
            output.write(f"{start_time} --> {end_time}\n")
            output.write(f"{segment.speaker}: {text}\n\n")
        
        return output.getvalue()

    def to_txt(self) -> str:
        """Convert to plain text format with speaker labels.
        
        Returns:
            String with speaker-aware transcript.
        """
        if not self.segments:
            return ""
        
        output = StringIO()
        previous_speaker = self.segments[0].speaker
        output.write(f"{previous_speaker}: ")

        for segment in self.segments:
            speaker = segment.speaker
            text = segment.text

            if speaker != previous_speaker:
                output.write(f"\n\n{speaker}: ")
                previous_speaker = speaker

            output.write(text + " ")

        return output.getvalue()

    @staticmethod
    def _format_timestamp(
        milliseconds: float,
        always_include_hours: bool = True,
        decimal_marker: str = ","
    ) -> str:
        """Format milliseconds as SRT timestamp.
        
        Args:
            milliseconds: Time in milliseconds.
            always_include_hours: Whether to always show hours.
            decimal_marker: Character to use for decimal separator.
            
        Returns:
            Formatted timestamp string (HH:MM:SS,mmm).
        """
        hours = int(milliseconds // 3_600_000)
        milliseconds -= hours * 3_600_000

        minutes = int(milliseconds // 60_000)
        milliseconds -= minutes * 60_000

        seconds = int(milliseconds // 1_000)
        milliseconds -= seconds * 1_000

        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return (
            f"{hours_marker}{minutes:02d}:{seconds:02d}"
            f"{decimal_marker}{int(milliseconds):03d}"
        )

