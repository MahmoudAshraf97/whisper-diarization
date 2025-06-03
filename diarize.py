import argparse
import logging
import os
import re

import faster_whisper
import torch
import torchaudio

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from helpers import (
    cleanup,
    create_config,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)

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
    default=True,
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
    default="medium.en",
    help="name of the Whisper model to use",
)

parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=8,
    help="Batch size for batched inference, reduce if you run out of memory, "
    "set to 0 for original whisper longform inference",
)

parser.add_argument(
    "--language",
    type=str,
    default=None,
    choices=whisper_langs,
    help="Language spoken in the audio, specify None to perform language detection",
)

parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)

parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for sampling")
parser.add_argument("--initial-prompt", type=str, default="", help="Initial prompt for context")
parser.add_argument("--hotwords", type=str, default="", help="Hotwords as a single string")
parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Penalty for repeated tokens")
parser.add_argument("--best-of", type=int, default=10, help="Number of candidates when sampling")
parser.add_argument("--beam-size", type=int, default=10, help="Beam size for beam search")
parser.add_argument("--patience", type=float, default=1.2, help="Beam search patience")
parser.add_argument("--no-repeat-ngram-size", type=int, default=3, help="Prevent repeating ngrams of this size")
parser.add_argument("--chunk-length", type=int, default=30, help="Length of audio chunks in seconds")
parser.add_argument("--length-penalty", type=float, default=1.0, help="Penalty for shorter/longer sequences")
parser.add_argument("--condition-on-previous-text", action="store_true", help="Condition decoding on previous text")
parser.add_argument("--multilingual", action="store_true", help="Enable multilingual mode")
parser.add_argument("--vad-filter", action="store_true", help="Enable voice activity detection filter")
parser.add_argument("--without-timestamps", action="store_true", help="Do not include timestamps in output")

args = parser.parse_args()
language = process_language_arg(args.language, args.model_name)

if args.stemming:
    return_code = os.system(
        f'python -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o temp_outputs --device "{args.device}"'
    )
    if return_code != 0:
        logging.warning(
            "Source splitting failed, using original audio file. "
            "Use --no-stem argument to disable it."
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

whisper_model = faster_whisper.WhisperModel(
    args.model_name, device=args.device, compute_type=mtypes[args.device]
)
whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
audio_waveform = faster_whisper.decode_audio(vocal_target)
suppress_tokens = (
    find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
    if args.suppress_numerals
    else [-1]
)

transcribe_kwargs = dict(
    audio=audio_waveform,
    language=language,
    suppress_tokens=suppress_tokens,
    log_progress=True,
    multilingual=args.multilingual,
    temperature=args.temperature,
    initial_prompt=args.initial_prompt,
    repetition_penalty=args.repetition_penalty,
    best_of=args.best_of,
    vad_filter=args.vad_filter,
    without_timestamps=args.without_timestamps,
    beam_size=args.beam_size,
    patience=args.patience,
    no_repeat_ngram_size=args.no_repeat_ngram_size,
    chunk_length=args.chunk_length,
    length_penalty=args.length_penalty,
    condition_on_previous_text=args.condition_on_previous_text,
    hotwords=args.hotwords,
)

if args.batch_size > 0:
    transcribe_kwargs["batch_size"] = args.batch_size
    transcript_segments, info = whisper_pipeline.transcribe(**transcribe_kwargs)
else:
    transcript_segments, info = whisper_model.transcribe(**transcribe_kwargs)

full_transcript = "".join(segment.text for segment in transcript_segments)

# clear gpu vram
del whisper_model, whisper_pipeline
torch.cuda.empty_cache()

# Forced Alignment
alignment_model, alignment_tokenizer = load_alignment_model(
    args.device,
    dtype=torch.float16 if args.device == "cuda" else torch.float32,
)

emissions, stride = generate_emissions(
    alignment_model,
    torch.from_numpy(audio_waveform)
    .to(alignment_model.dtype)
    .to(alignment_model.device),
    batch_size=args.batch_size,
)

del alignment_model
torch.cuda.empty_cache()

tokens_starred, text_starred = preprocess_text(
    full_transcript,
    romanize=True,
    language=langs_to_iso[info.language],
)

segments, scores, blank_token = get_alignments(
    emissions,
    tokens_starred,
    alignment_tokenizer,
)

spans = get_spans(tokens_starred, segments, blank_token)

word_timestamps = postprocess_results(text_starred, spans, stride, scores)


# convert audio to mono for NeMo combatibility
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
torchaudio.save(
    os.path.join(temp_path, "mono_file.wav"),
    torch.from_numpy(audio_waveform).unsqueeze(0).float(),
    16000,
    channels_first=True,
)


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

if info.language in punct_model_langs:
    # restoring punctuation in the transcript to help realign the sentences
    punct_model = PunctuationModel(model="kredor/punctuate-all")

    words_list = list(map(lambda x: x["word"], wsm))

    labled_words = punct_model.predict(words_list, chunk_size=230)

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
        f"Punctuation restoration is not available for {info.language} language."
        " Using the original punctuation."
    )

wsm = get_realigned_ws_mapping_with_punctuation(wsm)
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

with open(f"{os.path.splitext(args.audio)[0]}.txt", "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f)

with open(f"{os.path.splitext(args.audio)[0]}.srt", "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt)

cleanup(temp_path)
