from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("torch", reason="requires the torch optional dependency")

from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.tts_model import TTSModel


SAMPLING_RATE = 22050
LINE_WAVE_LENGTH = 1000
SPLIT_INTERVAL = 1.0


def _build_model(model_filename: str) -> TTSModel:
    hyper_parameters = HyperParameters()
    hyper_parameters.data.sampling_rate = SAMPLING_RATE
    return TTSModel(
        model_path=Path(model_filename),
        config_path=hyper_parameters,
        style_vec_path=np.zeros((1, 256), dtype=np.float32),
        device="cpu",
    )


def test_line_split_silence_length_follows_config_sampling_rate(monkeypatch):
    import style_bert_vits2.models.infer as infer_module

    model = _build_model("dummy.safetensors")
    model.net_g = object()  # 実モデルの遅延ロードを回避する

    line_wave = np.full(LINE_WAVE_LENGTH, 0.5, dtype=np.float32)
    monkeypatch.setattr(infer_module, "infer", lambda **kwargs: line_wave.copy())

    sampling_rate, audio = model.infer(
        "一行目\n二行目", line_split=True, split_interval=SPLIT_INTERVAL
    )

    assert sampling_rate == SAMPLING_RATE
    assert len(audio) == 2 * LINE_WAVE_LENGTH + int(SAMPLING_RATE * SPLIT_INTERVAL)


def test_line_split_silence_length_follows_config_sampling_rate_onnx(monkeypatch):
    import style_bert_vits2.models.infer_onnx as infer_onnx_module

    model = _build_model("dummy.onnx")
    model.onnx_session = object()  # 実モデルの遅延ロードを回避する

    line_wave = np.full(LINE_WAVE_LENGTH, 0.5, dtype=np.float32)
    monkeypatch.setattr(
        infer_onnx_module, "infer_onnx", lambda **kwargs: line_wave.copy()
    )

    sampling_rate, audio = model.infer(
        "一行目\n二行目", line_split=True, split_interval=SPLIT_INTERVAL
    )

    assert sampling_rate == SAMPLING_RATE
    assert len(audio) == 2 * LINE_WAVE_LENGTH + int(SAMPLING_RATE * SPLIT_INTERVAL)
