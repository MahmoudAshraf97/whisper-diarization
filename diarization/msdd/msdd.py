import json
import os
import tempfile

from typing import Union

import torch
import torchaudio

from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels
from omegaconf import OmegaConf


class MSDDDiarizer:
    def __init__(self, device: Union[str, torch.device]):
        self.model: NeuralDiarizer = NeuralDiarizer(cfg=create_config()).to(device)

    def diarize(self, audio: torch.Tensor):
        with tempfile.TemporaryDirectory() as temp_path:
            torchaudio.save(
                os.path.join(temp_path, "mono_file.wav"),
                audio,
                16000,
                channels_first=True,
            )

            manifest_path = os.path.join(temp_path, "manifest.json")
            meta = {
                "audio_filepath": os.path.join(temp_path, "mono_file.wav"),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "rttm_filepath": None,
                "uem_filepath": None,
            }

            with open(manifest_path, "w") as f:
                json.dump(meta, f)

            self.model._initialize_configs(
                manifest_path=manifest_path,
                max_speakers=8,
                num_speakers=None,
                tmpdir=temp_path,
                batch_size=24,
                num_workers=0,
                verbose=True,
            )
            self.model.clustering_embedding.clus_diar_model._diarizer_params.out_dir = (
                temp_path
            )
            self.model.clustering_embedding.clus_diar_model._diarizer_params.manifest_filepath = (
                manifest_path
            )
            self.model.msdd_model.cfg.test_ds.manifest_filepath = manifest_path
            self.model.diarize()

            pred_labels_clus = rttm_to_labels(
                os.path.join(temp_path, "pred_rttms", "mono_file.rttm")
            )

            labels = []
            for label in pred_labels_clus:
                start, end, speaker = label.split()
                start, end = float(start), float(end)
                start, end = int(start * 1000), int(end * 1000)
                labels.append((start, end, int(speaker.split("_")[1])))

            labels = sorted(labels, key=lambda x: x[0])

        return labels


def create_config():
    config = OmegaConf.load(
        os.path.join(os.path.dirname(__file__), "diar_infer_telephonic.yaml")
    )
    pretrained_vad = "vad_multilingual_marblenet"
    pretrained_speaker_model = "titanet_large"

    config.diarizer.out_dir = None
    config.diarizer.manifest_filepath = None
    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.oracle_vad = (
        False  # compute VAD provided with model_path to vad config
    )
    config.diarizer.clustering.parameters.oracle_num_speakers = False

    # Here, we use our in-house pretrained NeMo VAD model
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    config.diarizer.msdd_model.model_path = (
        "diar_msdd_telephonic"  # Telephonic speaker diarization model
    )

    return config
