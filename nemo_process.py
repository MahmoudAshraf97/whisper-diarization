
import argparse
import os
from helpers import *
import librosa
import soundfile
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
args = parser.parse_args()

# convert audio to mono for NeMo combatibility
signal, sample_rate = librosa.load(args.audio, sr=None)
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
if not os.path.exists(temp_path):
    os.mkdir(temp_path)
os.chdir(temp_path)
soundfile.write("mono_file.wav", signal, sample_rate, "PCM_24")

# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config()).to("cuda")
msdd_model.diarize()
