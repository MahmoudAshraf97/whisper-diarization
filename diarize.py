import argparse
from urllib.parse import urlparse
import os
from helpers import *
from faster_whisper import WhisperModel
import whisperx
import torch
from pydub import AudioSegment
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
import logging
import requests
import urllib
from io import BytesIO

mtypes = {"cpu": "int8", "cuda": "float16"}

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
parser.add_argument(
    "--no-stem",
    action="store_false",
    dest="stemming",
    default=False,
    help="Disables source separation."
    "This helps with long files that don't contain a lot of music.",
)

parser.add_argument(
    "--suppress_numerals",
    action="store_true",
    dest="suppress_numerals",
    default=False,
    help="Suppresses Numerical Digits."
    "This helps the diarization accuracy but converts all digits into written text.",
)

parser.add_argument(
    "--whisper-model",
    dest="model_name",
    default="large-v3",
    help="name of the Whisper model to use",
)

parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=8,
    help="Batch size for batched inference, reduce if you run out of memory, set to 0 for non-batched inference",
)

parser.add_argument(
    "--language",
    type=str,
    default=None,
    choices=whisper_langs,
    help="Language spoken in the audio, specify None to perform language detection",
)

parser.add_argument(
    "--wav2vec2-model",
    dest="wav2vec2_model_name",
    default=None,
    help="name of the Wav2Vec2 model to use.",
)

parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)

args = parser.parse_args()

if args.stemming:
    # Isolate vocals from the rest of the audio

    return_code = os.system(
        f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o "temp_outputs"'
    )

    if return_code != 0:
        logging.warning(
            "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
        )
        vocal_target = args.audio
    else:
        vocal_target = os.path.join(
            "temp_outputs",
            "htdemucs",
            os.path.splitext(os.path.basename(args.audio))[0],
            "vocals.wav",
        )
else:
    vocal_target = args.audio


# Transcribe the audio file
if args.batch_size != 0:
    from transcription_helpers import transcribe_batched

    whisper_results, language = transcribe_batched(
        vocal_target,
        args.language,
        args.batch_size,
        args.model_name,
        mtypes[args.device],
        args.suppress_numerals,
        args.device,
    )
else:
    from transcription_helpers import transcribe

    whisper_results, language = transcribe(
        vocal_target,
        args.language,
        args.model_name,
        mtypes[args.device],
        args.suppress_numerals,
        args.device,
    )


wav2vec2_model_name = args.wav2vec2_model_name

if language in wav2vec2_langs or wav2vec2_model_name is not None:
    alignment_model, metadata = whisperx.load_align_model(
        language_code=language, device=args.device,
        model_name=wav2vec2_model_name
    )
    result_aligned = whisperx.align(
        whisper_results, alignment_model, metadata, vocal_target, args.device
    )
    word_timestamps = filter_missing_timestamps(
        result_aligned["word_segments"],
        initial_timestamp=whisper_results[0].get("start"),
        final_timestamp=whisper_results[-1].get("end"),
    )
    # clear gpu vram
    del alignment_model
    torch.cuda.empty_cache()
else:
    assert (
        args.batch_size == 0  # TODO: add a better check for word timestamps existence
    ), (
        f"Unsupported language: {language}, use --batch_size to 0"
        " to generate word timestamps using whisper directly and fix this error."
    )
    word_timestamps = []
    for segment in whisper_results:
        for word in segment["words"]:
            word_timestamps.append({"word": word[2], "start": word[0], "end": word[1]})

if isinstance(vocal_target, str):
    if vocal_target.startswith("http://") or vocal_target.startswith("https://"):
        req = urllib.request.Request(vocal_target, headers={'User-Agent': ''})
        audio = urllib.request.urlopen(req).read()
        if vocal_target.endswith(".wav"):
            sound = AudioSegment.from_wav(BytesIO(audio)).set_channels(1)
        elif vocal_target.endswith(".mp3"):
            sound = AudioSegment.from_mp3(BytesIO(audio)).set_channels(1)
    else:
        sound = AudioSegment.from_file(vocal_target).set_channels(1)

# convert audio to mono for NeMo combatibility
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")

# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)
msdd_model.diarize()

del msdd_model
torch.cuda.empty_cache()

# Reading timestamps <> Speaker Labels mapping


speaker_ts = []
with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(" ")
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

if language in punct_model_langs:
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

else:
    logging.warning(
        f"Punctuation restoration is not available for {language} language. Using the original punctuation."
    )

wsm = get_realigned_ws_mapping_with_punctuation(wsm)
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

if isinstance(args.audio, str):
    if args.audio.startswith("http://") or args.audio.startswith("https://"):
        parsed_url = urlparse(args.audio)
        args.audio = os.path.basename(parsed_url.path)

with open(f"{os.path.splitext(args.audio)[0]}.txt", "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f)

with open(f"{os.path.splitext(args.audio)[0]}.srt", "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt)

cleanup(temp_path)
