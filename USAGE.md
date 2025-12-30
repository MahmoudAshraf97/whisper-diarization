# Whisper Diarization - Usage Guide

A comprehensive speech transcription and diarization library built on Whisper and NeMo MSDD models. This library provides both a Python API and command-line interface for processing audio files.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Python API](#python-api)
  - [Basic Usage](#basic-usage)
  - [Diarization](#diarization)
  - [Transcription Only](#transcription-only)
  - [Timestamp Extraction](#timestamp-extraction)
- [Command-Line Interface](#command-line-interface)
- [Advanced Usage](#advanced-usage)
- [Configuration Options](#configuration-options)
- [Output Formats](#output-formats)

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/MahmoudAshraf97/whisper-diarization.git

# Or install in development mode
git clone https://github.com/MahmoudAshraf97/whisper-diarization.git
cd whisper-diarization
pip install -e .
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## Quick Start

### Command Line

```bash
# Full diarization with speaker labels
python diarize.py --audio audio.mp3

# Transcription only
python diarize.py --audio audio.mp3 --mode transcribe

# Extract speaker timestamps only
python diarize.py --audio audio.mp3 --mode timestamps
```

### Python API

```python
from whisper_diarization import SpeechProcessor

# Initialize processor
processor = SpeechProcessor(
    model_name='medium.en',
    device='cuda'
)

# Perform diarization
result = processor.diarize('audio.mp3')

# Save outputs
with open('output.txt', 'w') as f:
    f.write(result.to_txt())

with open('output.srt', 'w') as f:
    f.write(result.to_srt())
```

## Python API

### Basic Usage

The `SpeechProcessor` class is the main interface for all operations:

```python
from whisper_diarization import SpeechProcessor

# Initialize with default settings
processor = SpeechProcessor()

# Or customize settings
processor = SpeechProcessor(
    model_name='large-v2',      # Whisper model
    device='cuda',               # 'cuda' or 'cpu'
    batch_size=8,                # Batch size for inference
    suppress_numerals=False,     # Convert numbers to text
    enable_stemming=True,        # Separate vocals from music
    diarizer_type='msdd'         # Diarization model
)
```

### Diarization

Full transcription with speaker identification:

```python
# Process audio file
result = processor.diarize('meeting.mp3', language='en')

# Access results
print(f"Language: {result.language}")
print(f"Full text: {result.text}")
print(f"Number of segments: {len(result.segments)}")

# Iterate through segments
for segment in result.segments:
    print(f"{segment.speaker} ({segment.start_ms}ms - {segment.end_ms}ms):")
    print(f"  {segment.text}")

# Export to different formats
txt_output = result.to_txt()      # Plain text with speaker labels
srt_output = result.to_srt()      # SRT subtitle format
dict_output = result.to_dict()    # Dictionary representation
```

### Transcription Only

Transcribe without speaker diarization (faster):

```python
result = processor.transcribe('audio.mp3', language='en')

# Access transcription
print(f"Text: {result.text}")
print(f"Language: {result.language}")

# Access word-level timestamps
for word in result.words:
    print(f"{word.text}: {word.start_ms}ms - {word.end_ms}ms")

# Export to dictionary
data = result.to_dict()
```

### Timestamp Extraction

Extract only speaker timing information (no transcription):

```python
timestamps = processor.get_timestamps('audio.mp3')

# Each timestamp has speaker_id, start_ms, end_ms
for ts in timestamps:
    duration = (ts.end_ms - ts.start_ms) / 1000
    print(f"Speaker {ts.speaker_id}: {duration:.2f}s")
```

### Context Manager

Use as a context manager for automatic cleanup:

```python
with SpeechProcessor(device='cuda') as processor:
    result = processor.diarize('audio.mp3')
    # Models are automatically cleaned up on exit
```

## Command-Line Interface

### Basic Commands

```bash
# Full diarization (default)
python diarize.py --audio audio.mp3

# Specify output location
python diarize.py --audio audio.mp3 --output results/meeting

# Use specific Whisper model
python diarize.py --audio audio.mp3 --whisper-model large-v2

# Specify language
python diarize.py --audio audio.mp3 --language pt
```

### Processing Modes

```bash
# Mode 1: Full diarization (default)
python diarize.py --audio audio.mp3 --mode diarize
# Outputs: audio.txt, audio.srt

# Mode 2: Transcription only
python diarize.py --audio audio.mp3 --mode transcribe
# Outputs: audio.txt, audio_words.json

# Mode 3: Timestamps only
python diarize.py --audio audio.mp3 --mode timestamps
# Outputs: audio_timestamps.json
```

### Performance Options

```bash
# Use CPU instead of GPU
python diarize.py --audio audio.mp3 --device cpu

# Adjust batch size (reduce if out of memory)
python diarize.py --audio audio.mp3 --batch-size 4

# Disable vocal separation (faster for speech-only files)
python diarize.py --audio audio.mp3 --no-stem

# Suppress numerals (better diarization, converts numbers to text)
python diarize.py --audio audio.mp3 --suppress_numerals
```

### Parallel Processing

For faster processing on multi-core systems:

```bash
python diarize_parallel.py --audio audio.mp3 --whisper-model large-v2
```

This runs diarization and transcription in parallel processes.

## Advanced Usage

### Custom Audio Preprocessing

```python
from whisper_diarization import AudioPreprocessor

# Initialize preprocessor
preprocessor = AudioPreprocessor(device='cuda')

# Separate vocals from music
vocals_path = preprocessor.separate_vocals('audio.mp3')

# Process with custom settings
audio_path = preprocessor.process(
    'audio.mp3',
    enable_stemming=True
)
```

### Working with Results

```python
# Diarization result manipulation
result = processor.diarize('audio.mp3')

# Filter segments by speaker
speaker_0_segments = [
    s for s in result.segments 
    if s.speaker == 'Speaker 0'
]

# Calculate speaking time per speaker
from collections import defaultdict

speaking_time = defaultdict(int)
for segment in result.segments:
    duration = segment.end_ms - segment.start_ms
    speaking_time[segment.speaker] += duration

for speaker, time_ms in speaking_time.items():
    print(f"{speaker}: {time_ms/1000:.2f}s")

# Export specific time range
start_time = 60000  # 1 minute in ms
end_time = 120000   # 2 minutes in ms

filtered_segments = [
    s for s in result.segments
    if s.start_ms >= start_time and s.end_ms <= end_time
]
```

### Batch Processing

```python
from pathlib import Path

# Process multiple files
audio_files = Path('audio_folder').glob('*.mp3')

processor = SpeechProcessor(device='cuda')

for audio_file in audio_files:
    print(f"Processing: {audio_file.name}")
    
    result = processor.diarize(audio_file)
    
    # Save with same name as input
    output_base = audio_file.stem
    with open(f'{output_base}.txt', 'w') as f:
        f.write(result.to_txt())
    
    print(f"Completed: {audio_file.name}")
```

## Configuration Options

### Whisper Models

Available models (ordered by size/accuracy):

- `tiny`, `tiny.en` - Fastest, least accurate
- `base`, `base.en` - Fast, basic accuracy
- `small`, `small.en` - Balanced
- `medium`, `medium.en` - Good accuracy (default)
- `large-v1`, `large-v2`, `large-v3` - Best accuracy, slower

Models with `.en` suffix are English-only and faster for English audio.

### Language Codes

Common language codes:

- `en` - English
- `pt` - Portuguese
- `es` - Spanish
- `fr` - French
- `de` - German
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean

Set to `None` for automatic detection.

### Device Selection

```python
# Use GPU (requires CUDA)
processor = SpeechProcessor(device='cuda')

# Use CPU
processor = SpeechProcessor(device='cpu')

# Auto-select (GPU if available)
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = SpeechProcessor(device=device)
```

## Output Formats

### Text Format (.txt)

Plain text with speaker labels:

```
Speaker 0: Hello, how are you doing today?

Speaker 1: I'm doing great, thanks for asking. How about you?

Speaker 0: Pretty good, thanks.
```

### SRT Format (.srt)

Standard subtitle format:

```
1
00:00:00,000 --> 00:00:03,500
Speaker 0: Hello, how are you doing today?

2
00:00:03,500 --> 00:00:07,200
Speaker 1: I'm doing great, thanks for asking.
```

### JSON Format (.json)

Structured data for programmatic access:

```json
{
  "text": "Full transcript...",
  "language": "en",
  "segments": [
    {
      "speaker": "Speaker 0",
      "text": "Hello, how are you doing today?",
      "start_ms": 0,
      "end_ms": 3500
    }
  ]
}
```

## Performance Tips

1. **Use GPU**: CUDA-enabled GPU provides 10-20x speedup
2. **Choose appropriate model**: Smaller models are faster but less accurate
3. **Disable stemming**: Use `--no-stem` for speech-only audio
4. **Adjust batch size**: Increase for faster processing, decrease if out of memory
5. **Use parallel processing**: `diarize_parallel.py` for multi-core systems
6. **Specify language**: Avoid auto-detection overhead when language is known

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python diarize.py --audio audio.mp3 --batch-size 2

# Use smaller model
python diarize.py --audio audio.mp3 --whisper-model small.en

# Use CPU
python diarize.py --audio audio.mp3 --device cpu
```

### Poor Diarization Quality

```bash
# Enable numeral suppression
python diarize.py --audio audio.mp3 --suppress_numerals

# Use larger Whisper model
python diarize.py --audio audio.mp3 --whisper-model large-v2
```

### Slow Processing

```bash
# Disable vocal separation
python diarize.py --audio audio.mp3 --no-stem

# Use smaller model
python diarize.py --audio audio.mp3 --whisper-model small.en

# Increase batch size (if memory allows)
python diarize.py --audio audio.mp3 --batch-size 16
```

## Examples

### Meeting Transcription

```python
from whisper_diarization import SpeechProcessor

processor = SpeechProcessor(
    model_name='large-v2',
    device='cuda',
    enable_stemming=False  # Meeting audio is usually clean
)

result = processor.diarize('meeting.mp3', language='en')

# Generate meeting minutes
with open('meeting_minutes.txt', 'w') as f:
    f.write(f"Meeting Transcript\n")
    f.write(f"Language: {result.language}\n\n")
    f.write(result.to_txt())

# Generate subtitles for video
with open('meeting_subtitles.srt', 'w') as f:
    f.write(result.to_srt())
```

### Podcast Processing

```python
processor = SpeechProcessor(
    model_name='medium.en',
    device='cuda',
    enable_stemming=True  # Remove intro/outro music
)

result = processor.diarize('podcast.mp3', language='en')

# Calculate host vs guest speaking time
speaking_time = {}
for segment in result.segments:
    speaker = segment.speaker
    duration = (segment.end_ms - segment.start_ms) / 1000
    speaking_time[speaker] = speaking_time.get(speaker, 0) + duration

print("Speaking time:")
for speaker, seconds in speaking_time.items():
    print(f"{speaker}: {seconds/60:.1f} minutes")
```

### Lecture Transcription

```python
# Transcribe lecture without diarization
processor = SpeechProcessor(model_name='large-v2')

result = processor.transcribe('lecture.mp3', language='en')

# Save full transcript
with open('lecture_transcript.txt', 'w') as f:
    f.write(result.text)

# Save with timestamps for note-taking
import json
with open('lecture_timestamps.json', 'w') as f:
    json.dump(result.to_dict(), f, indent=2)
```

## API Reference

For detailed API documentation, see the docstrings in:

- `processor.py` - Main SpeechProcessor class
- `models.py` - Result data classes
- `preprocessor.py` - Audio preprocessing utilities

## License

See LICENSE file for details.

