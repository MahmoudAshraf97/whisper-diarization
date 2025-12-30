"""
Core speech processing functionality.

This module provides the main SpeechProcessor class that orchestrates
transcription, alignment, and diarization operations.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Union

import faster_whisper
import torch
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel

from .diarization import MSDDDiarizer
from .helpers import (
    LANGUAGES,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
)
from .models import (
    DiarizationResult,
    Segment,
    SpeechTimestamp,
    TranscriptionResult,
    WordTimestamp,
)
from .preprocessor import AudioPreprocessor


logger = logging.getLogger(__name__)


class SpeechProcessor:
    """Main processor for speech transcription and diarization.
    
    This class provides a high-level interface for processing audio files,
    supporting transcription, speaker diarization, and timestamp extraction.
    Models are loaded lazily on first use to optimize memory usage.
    
    Attributes:
        model_name: Name of the Whisper model to use.
        device: Device for computation ('cuda' or 'cpu').
        batch_size: Batch size for inference.
        suppress_numerals: Whether to suppress numerical digits in transcription.
        enable_stemming: Whether to perform vocal separation preprocessing.
        temp_dir: Directory for temporary files.
    
    Example:
        >>> processor = SpeechProcessor(
        ...     model_name='medium.en',
        ...     device='cuda',
        ...     temp_dir='/tmp/whisper_temp'
        ... )
        >>> result = processor.diarize('audio.mp3')
        >>> print(result.to_txt())
    """

    def __init__(
        self,
        model_name: str = "medium.en",
        device: str = "cuda",
        batch_size: int = 8,
        suppress_numerals: bool = False,
        enable_stemming: bool = True,
        diarizer_type: str = "msdd",
        temp_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the speech processor.
        
        Args:
            model_name: Whisper model name (e.g., 'medium.en', 'large-v2').
            device: Device to use ('cuda' or 'cpu').
            batch_size: Batch size for inference. Set to 0 for original
                       Whisper longform inference.
            suppress_numerals: If True, suppresses numerical digits in output.
                             Helps diarization but converts digits to text.
            enable_stemming: If True, performs vocal separation preprocessing.
            diarizer_type: Type of diarizer to use (currently only 'msdd').
            temp_dir: Directory for temporary files (e.g., separated vocals).
                     If None, creates a temp_outputs directory in the current
                     working directory.
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.suppress_numerals = suppress_numerals
        self.enable_stemming = enable_stemming
        self.diarizer_type = diarizer_type
        self.temp_dir = Path(temp_dir) if temp_dir else None

        # Compute type based on device
        self.compute_type = "float16" if device == "cuda" else "int8"

        # Lazy-loaded models
        self._whisper_model = None
        self._whisper_pipeline = None
        self._alignment_model = None
        self._alignment_tokenizer = None
        self._diarizer_model = None
        self._punct_model = None

        # Preprocessor
        self.preprocessor = AudioPreprocessor(device=device, temp_dir=temp_dir)

        logger.info(
            f"SpeechProcessor initialized: model={model_name}, "
            f"device={device}, batch_size={batch_size}"
        )

    def _load_whisper_model(self):
        """Load Whisper model and pipeline lazily."""
        if self._whisper_model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self._whisper_model = faster_whisper.WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            self._whisper_pipeline = faster_whisper.BatchedInferencePipeline(
                self._whisper_model
            )
            logger.info("Whisper model loaded successfully")

    def _load_alignment_model(self):
        """Load alignment model lazily."""
        if self._alignment_model is None:
            logger.info("Loading alignment model")
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._alignment_model, self._alignment_tokenizer = (
                load_alignment_model(self.device, dtype=dtype)
            )
            logger.info("Alignment model loaded successfully")

    def _load_diarizer(self):
        """Load diarizer model lazily."""
        if self._diarizer_model is None:
            logger.info(f"Loading diarizer: {self.diarizer_type}")
            if self.diarizer_type == "msdd":
                self._diarizer_model = MSDDDiarizer(device=self.device)
            else:
                raise ValueError(f"Unknown diarizer type: {self.diarizer_type}")
            logger.info("Diarizer loaded successfully")

    def _load_punctuation_model(self):
        """Load punctuation restoration model lazily."""
        if self._punct_model is None:
            logger.info("Loading punctuation model")
            self._punct_model = PunctuationModel(model="kredor/punctuate-all")
            logger.info("Punctuation model loaded successfully")

    def _clear_whisper_model(self):
        """Clear Whisper model from memory."""
        if self._whisper_model is not None:
            del self._whisper_model
            del self._whisper_pipeline
            self._whisper_model = None
            self._whisper_pipeline = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.debug("Whisper model cleared from memory")

    def _clear_alignment_model(self):
        """Clear alignment model from memory."""
        if self._alignment_model is not None:
            del self._alignment_model
            self._alignment_model = None
            self._alignment_tokenizer = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.debug("Alignment model cleared from memory")

    def _clear_diarizer(self):
        """Clear diarizer from memory."""
        if self._diarizer_model is not None:
            del self._diarizer_model
            self._diarizer_model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.debug("Diarizer cleared from memory")

    def _preprocess_audio(self, audio_path: Union[str, Path]) -> str:
        """Preprocess audio file with optional vocal separation.
        
        Args:
            audio_path: Path to input audio file.
            
        Returns:
            Path to processed audio file.
        """
        return self.preprocessor.process(audio_path, self.enable_stemming)

    def _transcribe_audio(
        self, audio_waveform, language: Optional[str] = None
    ):
        """Transcribe audio waveform.
        
        Args:
            audio_waveform: Audio waveform array.
            language: Language code or None for auto-detection.
            
        Returns:
            Tuple of (transcript_segments, info).
        """
        self._load_whisper_model()

        suppress_tokens = (
            find_numeral_symbol_tokens(self._whisper_model.hf_tokenizer)
            if self.suppress_numerals
            else [-1]
        )

        if self.batch_size > 0:
            transcript_segments, info = self._whisper_pipeline.transcribe(
                audio_waveform,
                language,
                suppress_tokens=suppress_tokens,
                batch_size=self.batch_size,
            )
        else:
            transcript_segments, info = self._whisper_model.transcribe(
                audio_waveform,
                language,
                suppress_tokens=suppress_tokens,
                vad_filter=True,
            )

        return transcript_segments, info

    def _align_words(self, full_transcript: str, audio_waveform, language: str):
        """Perform forced alignment on transcript.
        
        Args:
            full_transcript: Full transcript text.
            audio_waveform: Audio waveform array.
            language: Language code.
            
        Returns:
            List of word timestamps.
        """
        self._load_alignment_model()

        emissions, stride = generate_emissions(
            self._alignment_model,
            torch.from_numpy(audio_waveform)
            .to(self._alignment_model.dtype)
            .to(self._alignment_model.device),
            batch_size=self.batch_size,
        )

        tokens_starred, text_starred = preprocess_text(
            full_transcript,
            romanize=True,
            language=langs_to_iso[language],
        )

        segments, scores, blank_token = get_alignments(
            emissions,
            tokens_starred,
            self._alignment_tokenizer,
        )

        spans = get_spans(tokens_starred, segments, blank_token)
        word_timestamps = postprocess_results(text_starred, spans, stride, scores)

        return word_timestamps

    def _restore_punctuation(self, wsm: List[dict], language: str) -> List[dict]:
        """Restore punctuation to word mappings.
        
        Args:
            wsm: Word-speaker mapping list.
            language: Language code.
            
        Returns:
            Word-speaker mapping with restored punctuation.
        """
        if language not in punct_model_langs:
            logger.warning(
                f"Punctuation restoration not available for {language}. "
                "Using original punctuation."
            )
            return wsm

        self._load_punctuation_model()

        words_list = [x["word"] for x in wsm]
        labeled_words = self._punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # Don't punctuate acronyms like U.S.A.
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labeled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

        return wsm

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio file with word-level timestamps.
        
        This method transcribes the audio and provides word-level timing
        information through forced alignment, but does not perform speaker
        diarization.
        
        Args:
            audio_path: Path to the audio file.
            language: Language code (e.g., 'en', 'pt'). If None, language
                     is automatically detected.
            
        Returns:
            TranscriptionResult containing the full text, detected language,
            and word-level timestamps.
            
        Example:
            >>> processor = SpeechProcessor()
            >>> result = processor.transcribe('audio.mp3', language='en')
            >>> print(f"Language: {result.language}")
            >>> print(f"Text: {result.text}")
            >>> for word in result.words[:5]:
            ...     print(f"{word.text}: {word.start_ms}-{word.end_ms}ms")
        """
        logger.info(f"Starting transcription: {audio_path}")

        # Preprocess audio
        processed_audio = self._preprocess_audio(audio_path)
        audio_waveform = faster_whisper.decode_audio(processed_audio)

        # Validate language
        if language:
            language = process_language_arg(language, self.model_name)

        # Transcribe
        transcript_segments, info = self._transcribe_audio(
            audio_waveform, language
        )
        full_transcript = "".join(segment.text for segment in transcript_segments)

        # Clear Whisper from memory
        self._clear_whisper_model()

        # Perform forced alignment
        word_timestamps = self._align_words(
            full_transcript, audio_waveform, info.language
        )

        # Clear alignment model
        self._clear_alignment_model()

        # Convert to WordTimestamp objects
        words = [
            WordTimestamp(
                text=wt["text"],
                start_ms=int(wt["start"] * 1000),
                end_ms=int(wt["end"] * 1000),
            )
            for wt in word_timestamps
        ]

        logger.info(
            f"Transcription complete: {len(words)} words, "
            f"language={info.language}"
        )

        return TranscriptionResult(
            text=full_transcript,
            language=info.language,
            words=words,
        )

    def get_timestamps(
        self, audio_path: Union[str, Path]
    ) -> List[SpeechTimestamp]:
        """Extract speech timestamps with speaker labels.
        
        This method performs diarization to identify when each speaker is
        talking, without transcribing the actual words.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            List of SpeechTimestamp objects indicating when each speaker
            is speaking.
            
        Example:
            >>> processor = SpeechProcessor()
            >>> timestamps = processor.get_timestamps('audio.mp3')
            >>> for ts in timestamps[:5]:
            ...     print(f"Speaker {ts.speaker_id}: "
            ...           f"{ts.start_ms}-{ts.end_ms}ms")
        """
        logger.info(f"Extracting timestamps: {audio_path}")

        # Preprocess audio
        processed_audio = self._preprocess_audio(audio_path)
        audio_waveform = faster_whisper.decode_audio(processed_audio)

        # Perform diarization
        self._load_diarizer()
        speaker_ts = self._diarizer_model.diarize(
            torch.from_numpy(audio_waveform).unsqueeze(0)
        )
        self._clear_diarizer()

        # Convert to SpeechTimestamp objects
        timestamps = [
            SpeechTimestamp(start_ms=start, end_ms=end, speaker_id=speaker)
            for start, end, speaker in speaker_ts
        ]

        logger.info(f"Extracted {len(timestamps)} speech segments")

        return timestamps

    def diarize(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
    ) -> DiarizationResult:
        """Perform full transcription with speaker diarization.
        
        This method combines transcription, forced alignment, and speaker
        diarization to produce a transcript with speaker labels and timing
        information for each segment.
        
        Args:
            audio_path: Path to the audio file.
            language: Language code (e.g., 'en', 'pt'). If None, language
                     is automatically detected.
            
        Returns:
            DiarizationResult containing the full text, detected language,
            and speaker-labeled segments with timestamps.
            
        Example:
            >>> processor = SpeechProcessor()
            >>> result = processor.diarize('meeting.mp3')
            >>> # Save as SRT subtitle file
            >>> with open('output.srt', 'w') as f:
            ...     f.write(result.to_srt())
            >>> # Save as plain text
            >>> with open('output.txt', 'w') as f:
            ...     f.write(result.to_txt())
        """
        logger.info(f"Starting diarization: {audio_path}")

        # Preprocess audio
        processed_audio = self._preprocess_audio(audio_path)
        audio_waveform = faster_whisper.decode_audio(processed_audio)

        # Validate language
        if language:
            language = process_language_arg(language, self.model_name)

        # Transcribe
        transcript_segments, info = self._transcribe_audio(
            audio_waveform, language
        )
        full_transcript = "".join(segment.text for segment in transcript_segments)

        # Clear Whisper from memory
        self._clear_whisper_model()

        # Perform forced alignment
        word_timestamps = self._align_words(
            full_transcript, audio_waveform, info.language
        )

        # Clear alignment model
        self._clear_alignment_model()

        # Perform diarization
        self._load_diarizer()
        speaker_ts = self._diarizer_model.diarize(
            torch.from_numpy(audio_waveform).unsqueeze(0)
        )
        self._clear_diarizer()

        # Map words to speakers
        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

        # Restore punctuation
        wsm = self._restore_punctuation(wsm, info.language)

        # Realign with punctuation
        wsm = get_realigned_ws_mapping_with_punctuation(wsm)

        # Get sentence-level speaker mapping
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

        # Convert to Segment objects
        segments = [
            Segment(
                speaker=s["speaker"],
                text=s["text"].strip(),
                start_ms=s["start_time"],
                end_ms=s["end_time"],
            )
            for s in ssm
        ]

        logger.info(
            f"Diarization complete: {len(segments)} segments, "
            f"language={info.language}"
        )

        return DiarizationResult(
            text=full_transcript,
            language=info.language,
            segments=segments,
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._clear_whisper_model()
        self._clear_alignment_model()
        self._clear_diarizer()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self._clear_whisper_model()
            self._clear_alignment_model()
            self._clear_diarizer()
        except Exception:
            pass

    @staticmethod
    def get_supported_languages() -> dict:
        """Get all supported languages for transcription.
        
        Returns a dictionary mapping language codes to their full names.
        These are all the languages supported by the Whisper model.
        
        Returns:
            Dictionary with language code as key and full name as value.
            
        Example:
            >>> langs = SpeechProcessor.get_supported_languages()
            >>> print(langs['pt'])  # 'portuguese'
            >>> print(langs['en'])  # 'english'
            >>> print(f"Total languages: {len(langs)}")
        """
        return LANGUAGES.copy()

    @staticmethod
    def get_punctuation_languages() -> List[str]:
        """Get languages that support punctuation restoration.
        
        Punctuation restoration is only available for a subset of languages.
        For other languages, original punctuation from transcription is used.
        
        Returns:
            List of language codes that support punctuation restoration.
            
        Example:
            >>> punct_langs = SpeechProcessor.get_punctuation_languages()
            >>> print(punct_langs)  # ['en', 'fr', 'de', 'es', ...]
            >>> 'pt' in punct_langs  # True
        """
        return punct_model_langs.copy()

    @staticmethod
    def is_language_supported(language: str) -> bool:
        """Check if a language is supported for transcription.
        
        Args:
            language: Language code (e.g., 'en', 'pt') or name (e.g., 'english').
            
        Returns:
            True if language is supported, False otherwise.
            
        Example:
            >>> SpeechProcessor.is_language_supported('pt')  # True
            >>> SpeechProcessor.is_language_supported('portuguese')  # True
            >>> SpeechProcessor.is_language_supported('xyz')  # False
        """
        lang_lower = language.lower()
        return lang_lower in LANGUAGES or lang_lower in LANGUAGES.values()

    @staticmethod
    def has_punctuation_support(language: str) -> bool:
        """Check if a language supports punctuation restoration.
        
        Args:
            language: Language code (e.g., 'en', 'pt').
            
        Returns:
            True if punctuation restoration is available, False otherwise.
            
        Example:
            >>> SpeechProcessor.has_punctuation_support('en')  # True
            >>> SpeechProcessor.has_punctuation_support('ja')  # False
        """
        return language.lower() in punct_model_langs

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SpeechProcessor(model='{self.model_name}', "
            f"device='{self.device}', batch_size={self.batch_size})"
        )

