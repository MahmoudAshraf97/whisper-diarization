import torch


def transcribe(
    audio_file: str,
    language: str,
    model_name: str,
    compute_dtype: str,
    suppress_numerals: bool,
    device: str,
):
    from faster_whisper import WhisperModel
    from helpers import find_numeral_symbol_tokens, wav2vec2_langs

    # Faster Whisper non-batched
    # Run on GPU with FP16
    whisper_model = WhisperModel(model_name, device=device, compute_type=compute_dtype)

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    if suppress_numerals:
        numeral_symbol_tokens = find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
    else:
        numeral_symbol_tokens = None

    if language is not None and language in wav2vec2_langs:
        word_timestamps = False
    else:
        word_timestamps = True

    segments, info = whisper_model.transcribe(
        audio_file,
        language=language,
        beam_size=5,
        word_timestamps=word_timestamps,  # TODO: disable this if the language is supported by wav2vec2
        suppress_tokens=numeral_symbol_tokens,
        vad_filter=True,
    )
    whisper_results = []
    for segment in segments:
        whisper_results.append(segment._asdict())
    # clear gpu vram
    del whisper_model
    torch.cuda.empty_cache()
    return whisper_results, info.language


def transcribe_batched(
    audio_file: str,
    language: str,
    batch_size: int,
    model_name: str,
    compute_dtype: str,
    suppress_numerals: bool,
    device: str,
):
    import whisperx

    # Faster Whisper batched
    whisper_model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_dtype,
        asr_options={"suppress_numerals": suppress_numerals},
    )
    audio = whisperx.load_audio(audio_file)
    result = whisper_model.transcribe(audio, language=language, batch_size=batch_size)
    del whisper_model
    torch.cuda.empty_cache()
    return result["segments"], result["language"]
