from __future__ import annotations

import os
import wave
from io import BytesIO
from pathlib import Path

import pytest


pytest.importorskip("fastapi", reason="requires the fastapi optional dependency")
pytest.importorskip("httpx", reason="requires the httpx optional dependency")
pytest.importorskip("torch", reason="requires the torch optional dependency")

from fastapi.testclient import TestClient

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.server.app import create_app
from style_bert_vits2.tts_model import TTSModelHolder
from style_bert_vits2.utils import torch_device_to_onnx_providers
from tests.conftest import requires_cuda


@pytest.mark.gpu
@requires_cuda()
def test_openai_server_runs_real_cuda_inference():
    model_root_value = os.environ.get("SBV2_OPENAI_TEST_MODEL_DIR")
    if model_root_value is None:
        pytest.skip("set SBV2_OPENAI_TEST_MODEL_DIR to run real server inference")

    model_root = Path(model_root_value)
    language = Languages(os.environ.get("SBV2_OPENAI_TEST_LANGUAGE", "JP"))
    model_name = os.environ.get("SBV2_OPENAI_TEST_MODEL")
    text = os.environ.get(
        "SBV2_OPENAI_TEST_TEXT",
        "This is a GPU integration test."
        if language == Languages.EN
        else "これはGPU統合テストです。",
    )

    pyopenjtalk.initialize_worker()
    update_dict()
    bert_models.load_model(language, device_map="cuda")
    bert_models.load_tokenizer(language)

    holder = TTSModelHolder(
        model_root,
        "cuda",
        torch_device_to_onnx_providers("cuda"),
    )
    if model_name is None:
        if not holder.model_names:
            pytest.fail(f"No models found in {model_root}")
        model_name = holder.model_names[0]
    if model_name not in holder.model_names:
        pytest.fail(f"Model {model_name!r} not found in {model_root}")

    response = TestClient(create_app(holder, default_language=language)).post(
        "/v1/audio/speech",
        json={
            "input": text,
            "model": model_name,
            "language": language.value,
            "response_format": "wav",
        },
    )

    assert response.status_code == 200, response.text
    with wave.open(BytesIO(response.content), "rb") as wav_file:
        assert wav_file.getframerate() > 0
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() > 0
