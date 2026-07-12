# ruff: noqa: E402
from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest


pytest.importorskip("fastapi", reason="requires the fastapi optional dependency")
pytest.importorskip("httpx", reason="requires the httpx optional dependency")
pytest.importorskip("torch", reason="requires the torch optional dependency")
pytest.importorskip("uvicorn", reason="requires the uvicorn optional dependency")

from fastapi.testclient import TestClient

from style_bert_vits2 import tts_model as tts_model_module
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese import user_dict as user_dict_module


class FakeModel:
    def __init__(self, spk2id: dict[str, int]):
        self.spk2id = spk2id
        self.infer_calls: list[dict[str, Any]] = []

    def infer(self, **kwargs: Any) -> tuple[int, np.ndarray]:
        self.infer_calls.append(kwargs)
        return 44100, np.zeros(256, dtype=np.int16)


class FakeModelHolder:
    def __init__(self, *args: Any, **kwargs: Any):
        self.model_names = ["fake"]
        self.models_info: list[Any] = []
        self.model = FakeModel(spk2id={"zundamon": 0, "metan": 1})

    def get_model(self, model_name: str, model_path_str: str) -> FakeModel:
        return self.model


@pytest.fixture(scope="module")
def server_editor():
    """server_editor module を、外部境界 (worker process / model file) だけ stub して import する。"""
    mp = pytest.MonkeyPatch()
    mp.setattr(sys, "argv", ["server_editor.py", "--device", "cpu"])
    mp.setattr(pyopenjtalk, "initialize_worker", lambda *args, **kwargs: None)
    mp.setattr(user_dict_module, "update_dict", lambda *args, **kwargs: None)
    mp.setattr(bert_models, "load_model", lambda *args, **kwargs: None)
    mp.setattr(bert_models, "load_tokenizer", lambda *args, **kwargs: None)
    mp.setattr(tts_model_module, "TTSModelHolder", FakeModelHolder)
    try:
        module = importlib.import_module("server_editor")
        yield module
    finally:
        mp.undo()


@pytest.fixture()
def fake_holder(server_editor, monkeypatch):
    holder = FakeModelHolder()
    monkeypatch.setattr(server_editor, "model_holder", holder)
    return holder


@pytest.fixture()
def client(server_editor) -> TestClient:
    return TestClient(server_editor.app)


def _line(**overrides: Any) -> dict[str, Any]:
    line: dict[str, Any] = {
        "model": "fake",
        "modelFile": "fake.safetensors",
        "text": "テスト",
        "moraToneList": [
            {"mora": "テ", "tone": 0},
            {"mora": "ス", "tone": 1},
            {"mora": "ト", "tone": 1},
        ],
    }
    line.update(overrides)
    return line


def test_multi_synthesis_passes_each_line_resolved_speaker_id(client, fake_holder):
    response = client.post(
        "/api/multi_synthesis",
        json={"lines": [_line(speaker="metan"), _line(speaker="zundamon")]},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    speaker_ids = [call.get("speaker_id") for call in fake_holder.model.infer_calls]
    assert speaker_ids == [1, 0]


def test_multi_synthesis_defaults_missing_speaker_to_id_zero(client, fake_holder):
    response = client.post("/api/multi_synthesis", json={"lines": [_line()]})

    assert response.status_code == 200
    assert [call.get("speaker_id") for call in fake_holder.model.infer_calls] == [0]


def test_multi_synthesis_rejects_unknown_speaker_with_400(client, fake_holder):
    response = client.post(
        "/api/multi_synthesis",
        json={"lines": [_line(speaker="nobody")]},
    )

    assert response.status_code == 400
    assert "nobody" in response.json()["detail"]


def test_synthesis_passes_resolved_speaker_id(client, fake_holder):
    response = client.post("/api/synthesis", json=_line(speaker="metan"))

    assert response.status_code == 200
    assert [call.get("speaker_id") for call in fake_holder.model.infer_calls] == [1]


def test_synthesis_rejects_unknown_speaker_with_400(client, fake_holder):
    response = client.post("/api/synthesis", json=_line(speaker="nobody"))

    assert response.status_code == 400
    assert "nobody" in response.json()["detail"]


_WORD = {"surface": "テスト", "pronunciation": "テスト", "accent_type": 1}


class _UpdateDictSpy:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls += 1


@pytest.fixture()
def user_dict_file_guard(server_editor):
    """dict_data/user_dict.json を test 前後で退避・復元する。"""
    path = user_dict_module.user_dict_path
    original = path.read_bytes() if path.is_file() else None
    try:
        yield
    finally:
        if original is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(original)


@pytest.fixture()
def update_dict_spy(server_editor, monkeypatch, user_dict_file_guard):
    spy = _UpdateDictSpy()
    monkeypatch.setattr(user_dict_module, "update_dict", spy)
    monkeypatch.setattr(server_editor, "update_dict", spy, raising=False)
    return spy


def test_add_user_dict_word_updates_dict_once(client, update_dict_spy):
    response = client.post("/api/user_dict_word", json=_WORD)

    assert response.status_code == 201
    assert update_dict_spy.calls == 1


def test_update_user_dict_word_updates_dict_once(client, update_dict_spy):
    uuid = client.post("/api/user_dict_word", json=_WORD).json()["uuid"]
    update_dict_spy.calls = 0

    response = client.put(
        f"/api/user_dict_word/{uuid}", json={**_WORD, "accent_type": 2}
    )

    assert response.status_code == 200
    assert update_dict_spy.calls == 1


def test_delete_user_dict_word_updates_dict_once(client, update_dict_spy):
    uuid = client.post("/api/user_dict_word", json=_WORD).json()["uuid"]
    update_dict_spy.calls = 0

    response = client.delete(f"/api/user_dict_word/{uuid}")

    assert response.status_code == 200
    assert update_dict_spy.calls == 1
