# Architecture Documentation

This document describes the refactored architecture of the Whisper Diarization project.

## Overview

The project has been refactored from a procedural script into a modular, object-oriented architecture with clear separation of concerns. The new design follows SOLID principles and provides both a Python API and CLI interface.

**Version 2.0** introduces pip installability, allowing the package to be used as a library in other projects.

## Installation

The package can be installed via pip:

```bash
# Install from GitHub
pip install git+https://github.com/MahmoudAshraf97/whisper-diarization.git

# Or install in development mode
git clone https://github.com/MahmoudAshraf97/whisper-diarization.git
cd whisper-diarization
pip install -e .
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Package: whisper_diarization             │
├─────────────────────────────────────────────────────────────┤
│                         CLI Layer                            │
│  ┌──────────────┐              ┌────────────────────┐       │
│  │  diarize.py  │              │ diarize_parallel.py│       │
│  └──────┬───────┘              └─────────┬──────────┘       │
└─────────┼────────────────────────────────┼──────────────────┘
          │                                 │
          └────────────┬────────────────────┘
                       │
┌──────────────────────┼────────────────────────────────────┐
│                      ▼         Core Layer                  │
│           ┌──────────────────────┐                         │
│           │  SpeechProcessor     │                         │
│           │  (processor.py)      │  ← Main API Entry      │
│           └──────────┬───────────┘                         │
│                      │                                      │
│        ┌─────────────┼─────────────┐                       │
│        ▼             ▼              ▼                       │
│  ┌──────────┐ ┌────────────┐ ┌──────────┐                 │
│  │ Whisper  │ │ Alignment  │ │Diarizer  │                 │
│  │  Model   │ │   Model    │ │  (MSDD)  │                 │
│  └──────────┘ └────────────┘ └──────────┘                 │
└────────────────────────────────────────────────────────────┘
          │                                 │
┌─────────┼─────────────────────────────────┼──────────────┐
│         ▼         Support Layer           ▼               │
│  ┌──────────────────┐         ┌──────────────────┐       │
│  │ AudioPreprocessor│         │     models.py     │       │
│  │ (preprocessor.py)│         │   (Data Models)   │       │
│  └──────────────────┘         └──────────────────┘       │
│                                                            │
│  ┌──────────────────────────────────────────────┐        │
│  │            helpers.py                         │        │
│  │         (Utility Functions)                   │        │
│  └──────────────────────────────────────────────┘        │
└────────────────────────────────────────────────────────────┘

Package Structure:
  whisper_diarization/
  ├── __init__.py          (Package exports)
  ├── processor.py         (Main processor)
  ├── models.py            (Data models)
  ├── preprocessor.py      (Audio preprocessing)
  ├── helpers.py           (Utilities)
  └── diarization/         (Diarization module)
      └── msdd/
          ├── msdd.py
          └── diar_infer_telephonic.yaml
```

## Package Structure and Imports

### Package Organization

The project is organized as a Python package named `whisper_diarization`:

```
whisper_diarization/
├── __init__.py           # Public API exports
├── processor.py          # SpeechProcessor (main class)
├── models.py             # Data models (results)
├── preprocessor.py       # AudioPreprocessor
├── helpers.py            # Utility functions
├── diarization/          # Diarization subpackage
│   ├── __init__.py
│   └── msdd/
│       ├── __init__.py
│       ├── msdd.py       # MSDD diarizer
│       └── diar_infer_telephonic.yaml  # Config
├── pyproject.toml        # Package metadata
├── setup.py              # Git dependencies
└── MANIFEST.in           # Additional files
```

### Public API

The package exports the following classes from `__init__.py`:

```python
from whisper_diarization import (
    SpeechProcessor,       # Main processor
    AudioPreprocessor,     # Audio preprocessing
    TranscriptionResult,   # Transcription output
    DiarizationResult,     # Diarization output
    WordTimestamp,         # Word with timing
    SpeechTimestamp,       # Speaker timing
    Segment,               # Speaker segment with text
)
```

### Import Strategy

**Internal imports** (within the package) use relative imports:
```python
# In processor.py
from .diarization import MSDDDiarizer
from .helpers import get_words_speaker_mapping
from .models import DiarizationResult
```

**External usage** (by users) uses absolute package imports:
```python
# In user code
from whisper_diarization import SpeechProcessor
```

This ensures the package works correctly whether installed via pip or used in development mode.

## Module Structure

### Core Modules

#### `processor.py` - SpeechProcessor Class

The main orchestrator that coordinates all processing operations.

**Responsibilities:**
- Manages model lifecycle (lazy loading, memory cleanup)
- Coordinates transcription, alignment, and diarization
- Provides three main public methods:
  - `transcribe()` - Transcription with word timestamps
  - `diarize()` - Full diarization with speaker labels
  - `get_timestamps()` - Speaker timing information only

**Key Features:**
- Lazy model loading (loads on first use)
- Automatic memory management
- Context manager support
- GPU memory optimization

**Design Patterns:**
- Facade Pattern: Simplifies complex subsystem interactions
- Lazy Initialization: Defers expensive operations
- Context Manager: Ensures proper resource cleanup

#### `models.py` - Data Models

Type-safe data structures for results.

**Classes:**
- `WordTimestamp` - Single word with timing
- `SpeechTimestamp` - Speaker segment timing
- `Segment` - Speaker segment with text
- `TranscriptionResult` - Transcription output
- `DiarizationResult` - Diarization output with export methods

**Key Features:**
- Dataclass-based for clarity and type safety
- Built-in serialization (`to_dict()`)
- Format conversion (`to_srt()`, `to_txt()`)
- Immutable by design

#### `preprocessor.py` - AudioPreprocessor Class

Handles audio preprocessing operations.

**Responsibilities:**
- Vocal separation using Demucs
- Temporary file management
- Audio format handling

**Key Features:**
- Graceful degradation (fallback to original audio)
- Automatic temp directory management
- Subprocess isolation for Demucs

### Support Modules

#### `helpers.py`

Pure utility functions for:
- Language code mappings
- Word-to-speaker alignment
- Sentence segmentation
- Punctuation restoration
- Timestamp formatting
- File cleanup

**Design Philosophy:**
- Stateless functions
- No side effects
- Easily testable
- Maintained for backward compatibility

#### `diarization/`

Existing diarization module:
- `MSDDDiarizer` - NeMo MSDD-based diarization
- Configuration files

### Interface Modules

#### `diarize.py` - CLI Interface

Command-line interface with three modes:
- `diarize` - Full processing (default)
- `transcribe` - Transcription only
- `timestamps` - Timing information only

**Features:**
- Argument parsing and validation
- Progress logging
- Multiple output formats
- Backward compatible with original script

#### `diarize_parallel.py` - Parallel Processing

Optimized version that runs diarization and transcription in parallel.

**Optimization:**
- Multiprocessing for CPU-bound tasks
- Reduces total processing time
- Handles inter-process communication

## Data Flow

### Diarization Flow

```
Audio File
    │
    ▼
AudioPreprocessor
    │ (optional vocal separation)
    ▼
Whisper Model
    │ (transcription)
    ▼
Alignment Model
    │ (word-level timestamps)
    ▼
Diarizer (MSDD)
    │ (speaker identification)
    ▼
Word-Speaker Mapping
    │
    ▼
Punctuation Restoration
    │
    ▼
Sentence Segmentation
    │
    ▼
DiarizationResult
    │
    ├─► to_txt() → Plain text
    ├─► to_srt() → Subtitles
    └─► to_dict() → JSON
```

### Transcription Flow

```
Audio File
    │
    ▼
AudioPreprocessor
    │
    ▼
Whisper Model
    │
    ▼
Alignment Model
    │
    ▼
TranscriptionResult
    │
    └─► to_dict() → JSON with words
```

### Timestamps Flow

```
Audio File
    │
    ▼
AudioPreprocessor
    │
    ▼
Diarizer (MSDD)
    │
    ▼
List[SpeechTimestamp]
```

## Memory Management

### GPU Memory Optimization

The `SpeechProcessor` implements aggressive memory management:

1. **Lazy Loading**: Models loaded only when needed
2. **Explicit Cleanup**: Models deleted after use
3. **Cache Clearing**: `torch.cuda.empty_cache()` after deletions
4. **Sequential Processing**: Only one heavy model in memory at a time

### Memory Flow

```
Load Whisper → Transcribe → Delete Whisper → Clear Cache
                                   │
                                   ▼
                          Load Alignment → Align → Delete Alignment → Clear Cache
                                                          │
                                                          ▼
                                                 Load Diarizer → Diarize → Delete → Clear Cache
```

## Design Decisions

### 1. Lazy Loading

**Rationale**: Models are large (1-5GB each). Loading all at once wastes memory.

**Implementation**: Private `_load_*` methods check if model is None before loading.

### 2. Dataclasses for Results

**Rationale**: Type safety, clarity, and IDE support.

**Benefits**:
- Autocomplete in IDEs
- Type checking with mypy
- Clear API contracts
- Easy serialization

### 3. Separate Preprocessor

**Rationale**: Audio preprocessing is independent and reusable.

**Benefits**:
- Single responsibility
- Testable in isolation
- Reusable across different workflows
- Graceful failure handling

### 4. Context Manager Support

**Rationale**: Ensures cleanup even if exceptions occur.

**Usage**:
```python
with SpeechProcessor() as processor:
    result = processor.diarize('audio.mp3')
# Automatic cleanup
```

### 5. CLI Modes

**Rationale**: Different use cases need different outputs.

**Modes**:
- `diarize`: Full processing for meetings/conversations
- `transcribe`: Fast transcription for single-speaker content
- `timestamps`: Quick speaker timing analysis

## Extension Points

### Adding New Diarizers

1. Create diarizer class with `diarize(audio_tensor)` method
2. Add to `SpeechProcessor._load_diarizer()`
3. Add choice to CLI arguments

### Adding New Output Formats

Add methods to `DiarizationResult`:
```python
def to_vtt(self) -> str:
    """Export as WebVTT format."""
    # Implementation
```

### Custom Preprocessing

Extend `AudioPreprocessor`:
```python
class CustomPreprocessor(AudioPreprocessor):
    def denoise(self, audio_path):
        # Custom denoising logic
        pass
```

## Testing Strategy

### Unit Tests

- `models.py`: Test serialization, format conversion
- `preprocessor.py`: Test vocal separation, fallback
- `helpers.py`: Test utility functions

### Integration Tests

- `processor.py`: Test full workflows with sample audio
- CLI: Test argument parsing and file outputs

### Performance Tests

- Memory usage monitoring
- Processing time benchmarks
- GPU utilization metrics

## Migration Guide

### From Old Script to New API

**Old (procedural):**
```python
# Run entire script
python diarize.py --audio audio.mp3
```

**New (API):**
```python
from whisper_diarization import SpeechProcessor

processor = SpeechProcessor()
result = processor.diarize('audio.mp3')
```

The package is now installable via pip, making it easy to use in other projects:
```bash
pip install git+https://github.com/MahmoudAshraf97/whisper-diarization.git
```

### Backward Compatibility

The CLI interface remains fully backward compatible:
```bash
# Still works exactly as before
python diarize.py --audio audio.mp3 --whisper-model medium.en
```

## Packaging and Distribution

### Package Configuration

The package uses modern Python packaging standards:

**`pyproject.toml`** (PEP 517/518):
- Package metadata (name, version, description)
- Dependencies (torch, faster-whisper, nemo-toolkit, etc.)
- Build system configuration (setuptools)
- Package data inclusion (YAML config files)

**`setup.py`**:
- Handles git-based dependencies:
  - `demucs` (vocal separation)
  - `deepmultilingualpunctuation` (punctuation restoration)
  - `ctc-forced-aligner` (forced alignment)

**`MANIFEST.in`**:
- Includes non-Python files in distribution:
  - Documentation files (README, LICENSE)
  - Configuration files (YAML)

### Installation Methods

**From GitHub:**
```bash
pip install git+https://github.com/MahmoudAshraf97/whisper-diarization.git
```

**Development mode:**
```bash
git clone https://github.com/MahmoudAshraf97/whisper-diarization.git
cd whisper-diarization
pip install -e .
```

**Specific version:**
```bash
pip install git+https://github.com/MahmoudAshraf97/whisper-diarization.git@v2.0.0
```

### Resource Management

**Configuration Files:**
- YAML configs are included as package data
- Loaded using `Path(__file__).parent / "config.yaml"`
- Fallback to `importlib.resources` for better compatibility

**Temporary Files:**
- Audio preprocessing creates temp directories
- Cleaned up automatically or on exit
- Configurable via `temp_dir` parameter

## Performance Characteristics

### Time Complexity

For audio of length T:
- Transcription: O(T)
- Alignment: O(T)
- Diarization: O(T)
- Total: O(T) - linear in audio length

### Space Complexity

- Models: ~5GB GPU memory
- Audio: ~100MB per hour (waveform)
- Results: ~1MB per hour (text)

### Bottlenecks

1. **Whisper transcription**: 50-70% of time
2. **Diarization**: 20-30% of time
3. **Alignment**: 10-20% of time
4. **Preprocessing**: 5-10% of time (if enabled)

## Future Improvements

### Potential Enhancements

1. **Streaming Support**: Process audio in chunks
2. **Async API**: Non-blocking processing
3. **Model Caching**: Keep models in memory between calls
4. **Batch Processing**: Process multiple files efficiently
5. **Custom Diarizers**: Plugin architecture
6. **Real-time Processing**: Live audio support
7. **Speaker Identification**: Named speakers instead of IDs
8. **Language Detection**: Per-segment language detection

### Code Quality

1. **Type Hints**: Already implemented throughout
2. **Documentation**: Comprehensive docstrings
3. **Testing**: Add comprehensive test suite
4. **CI/CD**: Automated testing and deployment
5. **Profiling**: Identify optimization opportunities

## Conclusion

The refactored architecture provides:

✅ **Modularity**: Clear separation of concerns  
✅ **Flexibility**: Multiple interfaces (API + CLI)  
✅ **Maintainability**: Well-documented, typed code  
✅ **Performance**: Optimized memory management  
✅ **Extensibility**: Easy to add new features  
✅ **Usability**: Simple API for common tasks  

The design balances professional engineering practices with practical usability, making it suitable for both production use and further development.

