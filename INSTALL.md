# Installation via pip

## Direct Installation from GitHub

To install the package directly from the GitHub repository:

```bash
pip install git+https://github.com/MahmoudAshraf97/whisper-diarization.git
```

### Install a Specific Branch

```bash
pip install git+https://github.com/MahmoudAshraf97/whisper-diarization.git@main
```

### Install a Specific Version/Tag

```bash
pip install git+https://github.com/MahmoudAshraf97/whisper-diarization.git@v2.0.0
```

## Development Mode Installation

If you've cloned the repository and want to install in editable (development) mode:

```bash
# Clone the repository
git clone https://github.com/MahmoudAshraf97/whisper-diarization.git
cd whisper-diarization

# Install in editable mode
pip install -e .
```

## Usage

After installation, you can use the package in your Python projects:

```python
from whisper_diarization import SpeechProcessor

# Create the processor
processor = SpeechProcessor(
    model_name='medium.en',
    device='cuda',  # or 'cpu'
    batch_size=8
)

# Transcribe and diarize audio
result = processor.diarize('audio.mp3')

# Save as SRT
with open('output.srt', 'w') as f:
    f.write(result.to_srt())

# Save as plain text
with open('output.txt', 'w') as f:
    f.write(result.to_txt())

# Access segments individually
for segment in result.segments:
    print(f"{segment.speaker}: {segment.text}")
```

## Git Dependencies

The package depends on some libraries that are installed directly from GitHub:
- demucs (vocal separation)
- deepmultilingualpunctuation (punctuation restoration)
- ctc-forced-aligner (forced alignment)

These dependencies are installed automatically during package installation.

## Troubleshooting

### Error Installing Git Dependencies

If you encounter problems installing git dependencies, try installing them manually:

```bash
pip install git+https://github.com/MahmoudAshraf97/demucs.git
pip install git+https://github.com/oliverguhr/deepmultilingualpunctuation.git
pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
```

### CUDA Issues

If you have CUDA-related problems, make sure you have PyTorch installed with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

