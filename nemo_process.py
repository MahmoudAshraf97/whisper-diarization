
import argparse
import os
from helpers import *
import librosa
import torch
import soundfile
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)
args = parser.parse_args()

# convert audio to mono for NeMo combatibility
signal, sample_rate = librosa.load(args.audio, sr=None)
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
soundfile.write(os.path.join(temp_path, "mono_file.wav"), signal, sample_rate, "PCM_24")

# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)
msdd_model.diarize()
