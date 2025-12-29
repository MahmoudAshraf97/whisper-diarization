"""
Audio preprocessing utilities for speech processing.

This module handles audio preprocessing tasks such as vocal separation
using the Demucs source separation model.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Union


logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Handles audio preprocessing operations.
    
    This class manages vocal separation from audio files using the Demucs
    model, which helps improve transcription and diarization accuracy by
    isolating speech from background music and noise.
    
    Attributes:
        device: Device to use for processing ('cuda' or 'cpu').
        temp_dir: Directory for temporary output files.
    """

    def __init__(
        self,
        device: str = "cpu",
        temp_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize the audio preprocessor.
        
        Args:
            device: Device to use ('cuda' or 'cpu'). Defaults to 'cpu'.
            temp_dir: Directory for temporary files. If None, creates a
                     temp_outputs directory in the current working directory.
        """
        self.device = device
        
        if temp_dir is None:
            pid = os.getpid()
            self.temp_dir = Path.cwd() / f"temp_outputs_{pid}"
        else:
            self.temp_dir = Path(temp_dir)
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"AudioPreprocessor initialized with device={device}")

    def separate_vocals(self, audio_path: Union[str, Path]) -> Optional[str]:
        """Separate vocals from an audio file using Demucs.
        
        This method uses the htdemucs model to isolate vocals from the audio,
        which can improve transcription accuracy for files with background music.
        
        Args:
            audio_path: Path to the input audio file.
            
        Returns:
            Path to the separated vocals file if successful, None if separation
            failed. Returns None rather than raising to allow fallback to
            original audio.
            
        Example:
            >>> preprocessor = AudioPreprocessor(device='cuda')
            >>> vocals_path = preprocessor.separate_vocals('audio.mp3')
            >>> if vocals_path:
            ...     print(f"Vocals saved to: {vocals_path}")
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return None
        
        logger.info(f"Starting vocal separation for: {audio_path.name}")
        
        # Build the demucs command
        cmd = [
            "python", "-m", "demucs.separate",
            "-n", "htdemucs",
            "--two-stems=vocals",
            str(audio_path),
            "-o", str(self.temp_dir),
            "--device", self.device
        ]
        
        try:
            # Run demucs as subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Construct path to the output vocals file
            audio_stem = audio_path.stem
            vocals_path = (
                self.temp_dir / "htdemucs" / audio_stem / "vocals.wav"
            )
            
            if vocals_path.exists():
                logger.info(f"Vocal separation successful: {vocals_path}")
                return str(vocals_path)
            else:
                logger.warning("Vocal separation completed but output not found")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.warning(
                f"Vocal separation failed (exit code {e.returncode}). "
                f"Error: {e.stderr}"
            )
            return None
        except Exception as e:
            logger.warning(f"Vocal separation failed with error: {e}")
            return None

    def process(
        self,
        audio_path: Union[str, Path],
        enable_stemming: bool = True
    ) -> str:
        """Process audio file with optional vocal separation.
        
        This is a convenience method that handles the decision of whether
        to perform vocal separation or use the original audio.
        
        Args:
            audio_path: Path to the input audio file.
            enable_stemming: Whether to attempt vocal separation.
                           If False or if separation fails, returns original path.
            
        Returns:
            Path to the processed audio (vocals if separation succeeded,
            original audio otherwise).
            
        Example:
            >>> preprocessor = AudioPreprocessor()
            >>> # With stemming
            >>> audio = preprocessor.process('music.mp3', enable_stemming=True)
            >>> # Without stemming
            >>> audio = preprocessor.process('speech.mp3', enable_stemming=False)
        """
        audio_path = Path(audio_path)
        
        if not enable_stemming:
            logger.info("Stemming disabled, using original audio")
            return str(audio_path)
        
        vocals_path = self.separate_vocals(audio_path)
        
        if vocals_path:
            logger.info("Using separated vocals for processing")
            return vocals_path
        else:
            logger.info(
                "Vocal separation failed or disabled, using original audio"
            )
            return str(audio_path)

    def get_temp_dir(self) -> Path:
        """Get the temporary directory path.
        
        Returns:
            Path object for the temporary directory.
        """
        return self.temp_dir

    def __repr__(self) -> str:
        """String representation of the preprocessor."""
        return (
            f"AudioPreprocessor(device='{self.device}', "
            f"temp_dir='{self.temp_dir}')"
        )

