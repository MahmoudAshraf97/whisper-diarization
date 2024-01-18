import argparse
import os
from helpers import *
import torch
from pydub import AudioSegment
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
import requests
import urllib
from io import BytesIO

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
if isinstance(args.audio, str):
    if args.audio.startswith("http://") or args.audio.startswith("https://"):
        req = urllib.request.Request(args.audio, headers={'User-Agent': ''})
        audio = urllib.request.urlopen(req).read()
        if args.audio.endswith(".wav"):
            sound = AudioSegment.from_wav(BytesIO(audio)).set_channels(1)
        elif args.audio.endswith(".mp3"):
            sound = AudioSegment.from_mp3(BytesIO(audio)).set_channels(1)
    else:
        sound = AudioSegment.from_file(args.audio).set_channels(1)


# convert audio to mono for NeMo combatibility
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")

# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)
msdd_model.diarize()
