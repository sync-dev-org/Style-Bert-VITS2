# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest


torch = pytest.importorskip("torch", reason="requires the torch optional dependency")
wavfile = pytest.importorskip("scipy.io.wavfile")

from style_bert_vits2.utils.audio import run_style_inference


class RecordingInference:
    def __init__(self) -> None:
        self.audio: dict[str, Any] | None = None

    def __call__(self, audio: dict[str, Any]) -> np.ndarray:
        self.audio = audio
        return np.arange(256, dtype=np.float32)


def test_run_style_inference_decodes_pcm_without_passing_a_path(tmp_path: Path):
    wav_path = tmp_path / "stereo.wav"
    pcm = np.array(
        [
            [-32768, 16384],
            [0, -16384],
            [16384, 0],
        ],
        dtype=np.int16,
    )
    wavfile.write(wav_path, 44100, pcm)
    inference = RecordingInference()

    result = run_style_inference(inference, wav_path)

    np.testing.assert_array_equal(result, np.arange(256, dtype=np.float32))
    assert inference.audio is not None
    assert inference.audio["sample_rate"] == 44100
    waveform = inference.audio["waveform"]
    assert waveform.dtype == torch.float32
    assert waveform.shape == (2, 3)
    torch.testing.assert_close(
        waveform,
        torch.tensor(
            [
                [-1.0, 0.0, 0.5],
                [0.5, -0.5, 0.0],
            ]
        ),
    )
