# ruff: noqa: E402
from __future__ import annotations

import atexit
from pathlib import Path
from typing import Any

import pytest


pytest.importorskip("fastapi", reason="requires the fastapi optional dependency")
pytest.importorskip("httpx", reason="requires the httpx optional dependency")
pytest.importorskip("GPUtil", reason="requires the GPUtil optional dependency")
pytest.importorskip("psutil", reason="requires the psutil optional dependency")
pytest.importorskip("torch", reason="requires the torch optional dependency")

import numpy as np
from fastapi.testclient import TestClient

from style_bert_vits2.constants import BASE_DIR, DEFAULT_BERT_MODEL_PATHS, Languages
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder
from style_bert_vits2.utils import torch_device_to_onnx_providers


class EmptyModelHolder:
    root_dir = Path("model_assets")
    device = "cpu"
    model_files_dict: dict[str, list[Path]] = {}
    model_names: list[str] = []
    models_info: list[Any] = []

    def refresh(self) -> None:
        return None


@pytest.fixture(scope="module")
def japanese_g2p_environment():
    pyopenjtalk.initialize_worker()
    try:
        update_dict()
        tokenizer_path = DEFAULT_BERT_MODEL_PATHS[Languages.JP]
        tokenizer_source = (
            str(tokenizer_path)
            if tokenizer_path.exists()
            else "ku-nlp/deberta-v2-large-japanese-char-wwm"
        )
        bert_models.load_tokenizer(Languages.JP, tokenizer_source)
        yield
    finally:
        pyopenjtalk.terminate_worker()
        bert_models.unload_tokenizer(Languages.JP)
        try:
            atexit.unregister(pyopenjtalk.terminate_worker)
        except ValueError:
            pass


@pytest.fixture()
def client(japanese_g2p_environment):
    from server_fastapi import create_app

    app = create_app(
        model_holder=EmptyModelHolder(),
        loaded_models=[],
        language=Languages.JP,
        limit=100,
        allow_origins=[],
    )
    return TestClient(app)


def test_status_returns_runtime_payload(client):
    response = client.get("/status")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["devices"], list)
    assert "cpu" in payload["devices"]
    assert isinstance(payload["cpu_percent"], int | float)
    assert isinstance(payload["memory_total"], int)
    assert isinstance(payload["memory_available"], int)
    assert isinstance(payload["memory_used"], int)
    assert isinstance(payload["memory_percent"], int | float)
    assert isinstance(payload["gpu"], list)


def test_status_returns_empty_gpu_list_when_gpu_detection_fails(monkeypatch):
    import GPUtil

    from server_fastapi import create_app

    def raise_gpu_detection_error():
        raise ValueError("unexpected nvidia-smi output")

    monkeypatch.setattr(GPUtil, "getGPUs", raise_gpu_detection_error)
    app = create_app(
        model_holder=EmptyModelHolder(),
        loaded_models=[],
        language=Languages.JP,
        limit=100,
        allow_origins=[],
    )
    client = TestClient(app)

    response = client.get("/status")

    assert response.status_code == 200
    assert response.json()["gpu"] == []


def test_models_info_returns_empty_mapping_without_assets(client):
    response = client.get("/models/info")

    assert response.status_code == 200
    assert response.json() == {}


def test_g2p_returns_kana_tone_pairs_without_model_assets(client):
    response = client.post("/g2p", params={"text": "こんにちは"})

    assert response.status_code == 200
    payload = response.json()
    assert payload
    assert all(isinstance(item, list) for item in payload)
    assert all(len(item) == 2 for item in payload)
    assert all(isinstance(item[0], str) for item in payload)
    assert all(isinstance(item[1], int) for item in payload)


def _model_holder_with_assets():
    holder = TTSModelHolder(
        BASE_DIR / "model_assets",
        "cpu",
        torch_device_to_onnx_providers("cpu"),
    )
    if holder.models_info:
        return holder

    pytest.skip("model_assets contains no TTS model assets")


class SingleDummyModelHolder(EmptyModelHolder):
    model_names = ["dummy"]


def test_voice_returns_400_for_effectively_empty_text():
    from server_fastapi import create_app

    # 実効テキスト空の検証は実モデルファイルのロード前に行われるため、
    # モデルファイルが存在しない TTSModel でも API 層の変換を検証できる
    model = TTSModel(
        model_path=Path("dummy.safetensors"),
        config_path=HyperParameters(),
        style_vec_path=np.zeros((1, 256), dtype=np.float32),
        device="cpu",
    )
    app = create_app(
        model_holder=SingleDummyModelHolder(),
        loaded_models=[model],
        language=Languages.JP,
        limit=100,
        allow_origins=[],
    )
    client = TestClient(app)

    response = client.post("/voice", params={"text": "\n"})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail[0]["type"] == "invalid_params"
    assert detail[0]["loc"] == ["query", "text"]


def test_voice_returns_non_empty_wav_when_model_assets_exist():
    from server_fastapi import create_app, load_models

    holder = _model_holder_with_assets()
    loaded_models = load_models(holder)
    app = create_app(
        model_holder=holder,
        loaded_models=loaded_models,
        language=Languages.JP,
        limit=100,
        allow_origins=[],
    )
    client = TestClient(app)

    response = client.post("/voice", params={"text": "こんにちは", "model_id": 0})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert len(response.content) > 0
