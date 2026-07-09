from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pytest
from scipy.io import wavfile

from style_bert_vits2.constants import BASE_DIR, Languages
from style_bert_vits2.tts_model import TTSModelHolder, TTSModelInfo
from tests.conftest import (
    requires_cuda,
    requires_onnx_provider,
    requires_python_package,
)


@dataclass(frozen=True)
class SynthesisCase:
    inference_type: Literal["torch", "onnx"]
    device: str
    onnx_providers: Sequence[tuple[str, dict[str, Any]]]


CPU_ONNX_PROVIDER = (
    "CPUExecutionProvider",
    {"arena_extend_strategy": "kSameAsRequested"},
)
CUDA_ONNX_PROVIDER = (
    "CUDAExecutionProvider",
    {
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "DEFAULT",
    },
)
DIRECTML_ONNX_PROVIDER = ("DmlExecutionProvider", {"device_id": 0})
COREML_ONNX_PROVIDER = ("CoreMLExecutionProvider", {})

SYNTHESIS_CASES = [
    pytest.param(
        SynthesisCase("torch", "cpu", [CPU_ONNX_PROVIDER]),
        marks=[requires_python_package("torch", "PyTorch synthesis")],
        id="torch-cpu",
    ),
    pytest.param(
        SynthesisCase("torch", "cuda", [CPU_ONNX_PROVIDER]),
        marks=[
            pytest.mark.gpu,
            requires_python_package("torch", "CUDA synthesis"),
            requires_cuda(),
        ],
        id="torch-cuda",
    ),
    pytest.param(
        SynthesisCase("onnx", "cpu", [CPU_ONNX_PROVIDER]),
        marks=[requires_onnx_provider("CPUExecutionProvider")],
        id="onnx-cpu",
    ),
    pytest.param(
        SynthesisCase("onnx", "cpu", [CUDA_ONNX_PROVIDER]),
        marks=[
            pytest.mark.gpu,
            requires_onnx_provider("CUDAExecutionProvider"),
        ],
        id="onnx-cuda",
    ),
    pytest.param(
        SynthesisCase("onnx", "cpu", [DIRECTML_ONNX_PROVIDER]),
        marks=[
            pytest.mark.gpu,
            requires_onnx_provider("DmlExecutionProvider"),
        ],
        id="onnx-directml",
    ),
    pytest.param(
        SynthesisCase("onnx", "cpu", [COREML_ONNX_PROVIDER]),
        marks=[
            pytest.mark.gpu,
            requires_onnx_provider("CoreMLExecutionProvider"),
            pytest.mark.xfail(
                strict=False,
                reason=(
                    "CoreMLExecutionProvider has a known GatherND zero-element "
                    "dynamic-shape limitation in the SDP flow"
                ),
            ),
        ],
        id="onnx-coreml",
    ),
]
TARGET_MODEL_NAMES = ("koharune-ami", "amitaro")
AUDIO_SILENCE_THRESHOLD = 1
SAMPLE_TEXTS = [
    "こんにちは、初めまして。あなたの名前はなんていうの？",
    "桜の樹の下には屍体が埋まっている！これは信じていいことなんだよ。",
    "あなたがいなくなって、私は一人になっちゃって、泣いちゃいそうなほど悲しい。",
    "音声合成は、機械学習を活用して、テキストから人の声を再現する技術です。この技術は、言語の構造を解析し、それに基づいて音声を生成します。",
]


def _target_style_params() -> list[pytest.ParameterSet]:
    holder = TTSModelHolder(BASE_DIR / "model_assets", "cpu", [CPU_ONNX_PROVIDER])
    params = [
        pytest.param(model_info.name, style, id=f"{model_info.name}-{style}")
        for model_info in holder.models_info
        if model_info.name in TARGET_MODEL_NAMES
        for style in model_info.styles
    ]
    if params:
        return params

    return [
        pytest.param(
            "",
            "",
            marks=pytest.mark.skip(reason="音声合成モデルが見つかりませんでした。"),
            id="no-target-model",
        )
    ]


def _find_model_info(holder: TTSModelHolder, model_name: str) -> TTSModelInfo:
    for model_info in holder.models_info:
        if model_info.name == model_name:
            return model_info
    raise AssertionError(f"Target synthesis model {model_name!r} disappeared")


def _select_model_file(
    model_info: TTSModelInfo, inference_type: Literal["torch", "onnx"]
) -> str:
    suffix = ".safetensors" if inference_type == "torch" else ".onnx"
    for model_file in model_info.files:
        if model_file.endswith(suffix) and not model_file.startswith("."):
            return model_file
    pytest.skip(
        f'音声合成モデル "{model_info.name}" の {suffix} モデルファイルが見つかりませんでした。'
    )


def _assert_audio_matches_model(sample_rate: int, audio_data: np.ndarray, expected_sample_rate: int) -> None:
    assert sample_rate == expected_sample_rate
    assert audio_data.size > 0
    assert np.max(np.abs(audio_data.astype(np.int64))) > AUDIO_SILENCE_THRESHOLD


@pytest.mark.parametrize("model_name,style", _target_style_params())
@pytest.mark.parametrize("synthesis_case", SYNTHESIS_CASES)
def test_synthesize_generates_non_silent_audio(
    tmp_path, synthesis_case: SynthesisCase, model_name: str, style: str
):
    holder = TTSModelHolder(
        BASE_DIR / "model_assets",
        synthesis_case.device,
        synthesis_case.onnx_providers,
    )
    model_info = _find_model_info(holder, model_name)
    model_file = _select_model_file(model_info, synthesis_case.inference_type)
    model = holder.get_model(model_info.name, model_file)
    try:
        model.load()
        if synthesis_case.inference_type == "onnx":
            assert model.onnx_session is not None
            assert (
                model.onnx_session.get_providers()[0]
                == synthesis_case.onnx_providers[0][0]
            )

        expected_sample_rate = model.hyper_parameters.data.sampling_rate
        for index, text in enumerate(SAMPLE_TEXTS, start=1):
            sample_rate, audio_data = model.infer(
                text,
                language=Languages.JP,
                speaker_id=0,
                sdp_ratio=0.4,
                style=style,
                style_weight=2.0,
            )
            _assert_audio_matches_model(sample_rate, audio_data, expected_sample_rate)

            wav_file_path = tmp_path / model_info.name / style / f"{index:02d}.wav"
            wav_file_path.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(wav_file_path, sample_rate, audio_data)
            written_sample_rate, written_audio_data = wavfile.read(wav_file_path)

            assert written_sample_rate == sample_rate
            assert written_audio_data.shape == audio_data.shape
    finally:
        model.unload()
