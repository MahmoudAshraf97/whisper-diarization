import argparse
import logging
import multiprocessing as mp
import os
import re

import faster_whisper
import torch

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel

from diarization import MSDDDiarizer
from helpers import (
    cleanup,
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


def diarize_parallel(audio: torch.Tensor, device, queue: mp.Queue):
    model = MSDDDiarizer(device=device)
    result = model.diarize(audio)
    queue.put(result)


mp.set_start_method("spawn", force=True)

if __name__ == "__main__":
    mtypes = {"cpu": "int8", "cuda": "float16"}
    pid = os.getpid()
    temp_outputs_dir = f"temp_outputs_{pid}"
    temp_path = os.path.join(os.getcwd(), temp_outputs_dir)
    os.makedirs(temp_path, exist_ok=True)

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
        default="large-v2",
        help="name of the Whisper model to use",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        default=4,
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

    parser.add_argument(
        "--diarizer",
        default="msdd",
        choices=["msdd"],
        help="Choose the diarization model to use",
    )

    args = parser.parse_args()
    language = process_language_arg(args.language, args.model_name)

    if args.stemming:
        # Isolate vocals from the rest of the audio

        return_code = os.system(
            f'python -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o "{temp_outputs_dir}" --device "{args.device}"'
        )

        if return_code != 0:
            logging.warning(
                "Source splitting failed, using original audio file. "
                "Use --no-stem argument to disable it."
            )
            vocal_target = args.audio
        else:
            vocal_target = os.path.join(
                temp_outputs_dir,
                "htdemucs",
                os.path.splitext(os.path.basename(args.audio))[0],
                "vocals.wav",
            )
    else:
        vocal_target = args.audio

    audio_waveform = faster_whisper.decode_audio(vocal_target)

    logging.info("Starting Nemo process with vocal_target: ", vocal_target)
    results_queue = mp.Queue()
    nemo_process = mp.Process(
        target=diarize_parallel,
        args=(
            torch.from_numpy(audio_waveform).unsqueeze(0),
            args.device,
            results_queue,
        ),
    )
    nemo_process.start()
    # Transcribe the audio file

    whisper_model = faster_whisper.WhisperModel(
        args.model_name, device=args.device, compute_type=mtypes[args.device]
    )
    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)

    suppress_tokens = (
        find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
        if args.suppress_numerals
        else [-1]
    )

    if args.batch_size > 0:
        transcript_segments, info = whisper_pipeline.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            batch_size=args.batch_size,
        )
    else:
        transcript_segments, info = whisper_model.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            vad_filter=True,
        )

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

    nemo_process.join()
    if results_queue.empty():
        raise RuntimeError("Diarization process did not return any results.")

    speaker_ts = results_queue.get_nowait()

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

    with open(
        f"{os.path.splitext(args.audio)[0]}.srt", "w", encoding="utf-8-sig"
    ) as srt:
        write_srt(ssm, srt)

    cleanup(temp_path)
