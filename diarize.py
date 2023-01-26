import argparse
import os
from helpers import *
from whisper import load_model
import whisperx
import torch
import librosa
import soundfile
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
args = parser.parse_args()
download_target = args.audio


# ROOT = os.getcwd()
# os.chdir(ROOT)
# Isolate vocals from the rest of the audio
os.system(
    f'python3 -m demucs.separate -n htdemucs_ft --two-stems=vocals "{download_target}" -o "temp_outputs"'
)
vocal_target = f"temp_outputs/htdemucs_ft/{download_target[:-4]}/vocals.wav"


# Large models result in considerably better and more aligned (words, timestamps) mapping.
model = load_model("medium.en")
whisper_results = model.transcribe(vocal_target, beam_size=None)

# clear gpu vram
del model
torch.cuda.empty_cache()

device = "cuda"
alignment_model, metadata = whisperx.load_align_model(language_code="en", device=device)
result_aligned = whisperx.align(
    whisper_results["segments"], alignment_model, metadata, vocal_target, device
)

# clear gpu vram
del alignment_model
torch.cuda.empty_cache()

# convert audio to mono for NeMo combatibility
signal, sample_rate = librosa.load(vocal_target, sr=None)
os.chdir("temp_outputs")
soundfile.write("mono_file.wav", signal, sample_rate, "PCM_24")

# Initialize NeMo MSDD diarization model
model = NeuralDiarizer(cfg=create_config())
model.diarize()

del model
torch.cuda.empty_cache()

# Reading timestamps <> Speaker Labels mapping

output_dir = "nemo_outputs"

speaker_ts = []
with open(f"{output_dir}/pred_rttms/mono_file.rttm", "r") as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(" ")
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

wsm = get_words_speaker_mapping(result_aligned["word_segments"], speaker_ts, "start")

# restoring punctuation in the transcript to help realign the sentences
punct_model = PunctuationModel(model="kredor/punctuate-all")

words_list = list(map(lambda x: x["word"], wsm))

labled_words = punct_model.predict(words_list)

ending_puncts = ".?!"
model_puncts = ".,;:!?"

# We don't want to punctuate U.S.A. with a period. Right?
is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

for word_dict, labeled_tuple in zip(wsm, labled_words):
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

os.chdir("..")  # back to parent dir

wsm = get_realigned_ws_mapping_with_punctuation(wsm)
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

with open(f"{download_target[:-4]}.txt", "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f)

with open(f"{download_target[:-4]}.srt", "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt)

cleanup("temp_outputs")
