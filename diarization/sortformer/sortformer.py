from warnings import warn

import torch

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig


class SortformerDiarizer:
    def __init__(self, device):
        self.model: SortformerEncLabelModel = SortformerEncLabelModel.from_pretrained(
            "nvidia/diar_streaming_sortformer_4spk-v2", map_location=device
        )

        self.model.sortformer_modules.chunk_len = 340
        self.model.sortformer_modules.chunk_right_context = 40
        self.model.sortformer_modules.fifo_len = 40
        self.model.sortformer_modules.spkcache_update_period = 300
        self.model.sortformer_modules.spkcache_len = 188
        self.model.sortformer_modules._check_streaming_parameters()

        warn(
            "Sortformer supports maximum of 4 speakers only, "
            "please use MSDD if your audio has more than 4 speakers",
            Warning,
        )

        self.model.eval()

    def diarize(self, audio: torch.Tensor):
        with torch.inference_mode():
            processed_signal, processed_signal_length = self.model.process_signal(
                audio_signal=audio,
                audio_signal_length=torch.tensor([audio.shape[-1]]),
            )
            processed_signal = processed_signal[:, :, : processed_signal_length.max()]

            preds = self.model.forward_streaming(processed_signal, processed_signal_length)
            preds = preds.cpu()

        # TODO: make this tunable
        diarize_cfg = DiarizeConfig(
            postprocessing_params={
                "onset": 0.5,
                "offset": 0.5,
                "pad_onset": 0.0,
                "pad_offset": 0.0,
                "min_duration_on": 0.0,
                "min_duration_off": 0.0,
            }
        )

        audio_rttm_map_dict = {
            "audio": {
                "uniq_id": "audio",
                "audio_filepath": "tensor_audio",
                "offset": 0.0,
                "duration": None,
                "text": "-",
                "label": "infer",
            }
        }

        self.model._diarize_audio_rttm_map = audio_rttm_map_dict
        uniq_ids = list(self.model._diarize_audio_rttm_map.keys())
        processed_outputs = self.model._diarize_output_processing(preds, uniq_ids, diarize_cfg)
        self.model._diarize_audio_rttm_map = {}

        labels = []
        for label in processed_outputs[0]:
            start, end, speaker = label.split()
            start, end = float(start), float(end)
            start, end = int(start * 1000), int(end * 1000)
            labels.append((start, end, int(speaker.split("_")[1])))

        labels = sorted(labels, key=lambda x: x[0])

        return labels
