"""
Setup script for whisper-diarization.

This script handles git-based dependencies that cannot be specified in pyproject.toml.
"""

from setuptools import setup

# Git-based dependencies
git_dependencies = [
    "demucs @ git+https://github.com/MahmoudAshraf97/demucs.git",
    "deepmultilingualpunctuation @ git+https://github.com/oliverguhr/deepmultilingualpunctuation.git",
    "ctc-forced-aligner @ git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git",
]

setup(
    dependency_links=[],
    install_requires=[
        # Standard dependencies from pyproject.toml
        "nltk",
        "faster-whisper>=1.1.0",
        "torch",
        "torchaudio",
        "numpy",
        "nemo-toolkit[asr]>=2.3.0",
        # Git-based dependencies
    ] + git_dependencies,
)

