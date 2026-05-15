from __future__ import annotations

from math import gcd
from typing import Any, Dict, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def resolve_device(device_name: str) -> str:
    """Resolve user-facing device name to a torch device string."""
    if device_name == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_name in {"cpu", "-1"}:
        return "cpu"
    if device_name.isdigit():
        return f"cuda:{device_name}"
    return device_name


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convert an audio array to mono float32 samples."""
    audio_array = np.asarray(audio)
    if audio_array.ndim == 2:
        if audio_array.shape[0] <= audio_array.shape[1]:
            audio_array = audio_array.mean(axis=0)
        else:
            audio_array = audio_array.mean(axis=1)
    return audio_array.astype(np.float32)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio to the target sampling rate when needed."""
    if source_rate == target_rate:
        return audio

    divisor = gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    return resample_poly(audio, up, down).astype(np.float32)


class AsrTranscriber:
    def __init__(
        self,
        model_name: str,
        device_name: str = "auto",
        language: str = "russian",
        task: str = "transcribe",
    ) -> None:
        """Load Whisper ASR model and processor once for reuse."""
        self.model_name = model_name
        self.device = resolve_device(device_name)
        self.torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.generate_kwargs: Dict[str, Any] = {}
        if language:
            self.generate_kwargs["language"] = language
        if task:
            self.generate_kwargs["task"] = task

        model_kwargs: Dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                dtype=self.torch_dtype,
                **model_kwargs,
            )
        except TypeError:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                **model_kwargs,
            )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        """Read an audio file from disk and return mono samples with sample rate."""
        audio, sampling_rate = sf.read(path)
        return to_mono_float32(audio), int(sampling_rate)

    def transcribe_path(self, path: str) -> str:
        """Transcribe an audio file path into text."""
        audio, sampling_rate = self._load_audio(path)
        target_sampling_rate = self.processor.feature_extractor.sampling_rate
        audio = resample_audio(audio, sampling_rate, target_sampling_rate)

        inputs = self.processor(
            audio,
            sampling_rate=target_sampling_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)

        generation_kwargs = dict(self.generate_kwargs)
        if "attention_mask" in inputs:
            generation_kwargs["attention_mask"] = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, **generation_kwargs)

        return str(self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]).strip()
