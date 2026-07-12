from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.tts_model import TTSModel


def _build_model() -> TTSModel:
    return TTSModel(
        model_path=Path("dummy.safetensors"),
        config_path=HyperParameters(),
        style_vec_path=np.zeros((1, 256), dtype=np.float32),
        device="cpu",
    )


@pytest.mark.parametrize("text", ["", "\n", "\n\n"])
@pytest.mark.parametrize("line_split", [True, False])
def test_infer_rejects_effectively_empty_text(text, line_split):
    from style_bert_vits2.tts_model import EmptyEffectiveTextError

    model = _build_model()

    with pytest.raises(EmptyEffectiveTextError):
        model.infer(text, line_split=line_split)


def test_convert_to_16_bit_wav_treats_all_zero_waveform_as_silence():
    data = np.zeros(100, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converted = TTSModel.convert_to_16_bit_wav(data)

    assert converted.dtype == np.int16
    assert np.array_equal(converted, np.zeros(100, dtype=np.int16))


def test_convert_to_16_bit_wav_normalizes_non_zero_float_waveform():
    data = np.array([0.0, 0.5], dtype=np.float32)

    converted = TTSModel.convert_to_16_bit_wav(data)

    assert converted.dtype == np.int16
    assert converted[0] == 0
    assert converted[1] == 32767
