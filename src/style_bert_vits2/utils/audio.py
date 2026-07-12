from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray


def load_audio_for_pyannote(wav_path: str | Path) -> dict[str, Any]:
    """Decode audio to the in-memory mapping accepted by pyannote."""
    samples, sample_rate = sf.read(
        wav_path,
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(np.ascontiguousarray(samples.T))
    return {"waveform": waveform, "sample_rate": sample_rate}


def run_style_inference(
    inference: Callable[[dict[str, Any]], NDArray[Any]],
    wav_path: str | Path,
) -> NDArray[Any]:
    """Run pyannote inference without delegating path decoding to torchcodec."""
    return inference(load_audio_for_pyannote(wav_path))
