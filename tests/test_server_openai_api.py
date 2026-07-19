from __future__ import annotations

import wave
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest


pytest.importorskip("fastapi", reason="requires the fastapi optional dependency")
pytest.importorskip("httpx", reason="requires the httpx optional dependency")

from fastapi.testclient import TestClient

from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from style_bert_vits2.server.app import create_app, split_sentences


class FakeModel:
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.spk2id = {"Alice": 0, "Bob": 1}
        self.id2spk = {0: "Alice", 1: "Bob"}
        self.style2id = {"Neutral": 0, "Happy": 1}
        self.infer_calls: list[dict[str, Any]] = []

    def infer(self, **kwargs: Any) -> tuple[int, np.ndarray]:
        self.infer_calls.append(kwargs)
        call_number = len(self.infer_calls)
        audio = np.array([call_number, -call_number], dtype=np.int16)
        return self.sample_rate, audio


class FakeModelHolder:
    def __init__(self):
        self.model_names = ["voice-a", "voice-b"]
        self.model_files_dict = {
            name: [Path(f"{name}.safetensors")] for name in self.model_names
        }
        self.models = {name: FakeModel() for name in self.model_names}
        self.get_model_calls: list[tuple[str, str]] = []

    def get_model(self, model_name: str, model_path_str: str) -> FakeModel:
        self.get_model_calls.append((model_name, model_path_str))
        return self.models[model_name]


@pytest.fixture()
def holder() -> FakeModelHolder:
    return FakeModelHolder()


@pytest.fixture()
def client(holder: FakeModelHolder) -> TestClient:
    return TestClient(create_app(holder, default_language=Languages.EN))


def test_health_reports_process_liveness(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_lists_served_model_identifiers(client: TestClient):
    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {"id": "voice-a", "object": "model"},
            {"id": "voice-b", "object": "model"},
        ],
    }


def test_buffered_speech_returns_pcm16_wav_and_maps_defaults(
    client: TestClient, holder: FakeModelHolder
):
    response = client.post("/v1/audio/speech", json={"input": "Hello."})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    with wave.open(BytesIO(response.content), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert (
            wav_file.readframes(wav_file.getnframes())
            == np.array([1, -1], dtype="<i2").tobytes()
        )

    call = holder.models["voice-a"].infer_calls[0]
    assert call == {
        "text": "Hello.",
        "language": Languages.EN,
        "speaker_id": 0,
        "sdp_ratio": DEFAULT_SDP_RATIO,
        "noise": DEFAULT_NOISE,
        "noise_w": DEFAULT_NOISEW,
        "length": 1.0,
        "line_split": False,
        "assist_text": None,
        "assist_text_weight": DEFAULT_ASSIST_TEXT_WEIGHT,
        "use_assist_text": False,
        "style": DEFAULT_STYLE,
        "style_weight": DEFAULT_STYLE_WEIGHT,
    }


def test_buffered_speech_maps_all_extension_fields_and_voice_wins(
    client: TestClient, holder: FakeModelHolder
):
    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "Hello.",
            "model": "voice-b",
            "voice": "Bob",
            "speaker_id": 0,
            "language": "JP",
            "speed": 2.0,
            "style": "Happy",
            "style_weight": 1.5,
            "sdp_ratio": 0.3,
            "noise": 0.4,
            "noise_w": 0.5,
            "assist_text": "Calm delivery.",
            "assist_text_weight": 0.8,
        },
    )

    assert response.status_code == 200
    call = holder.models["voice-b"].infer_calls[0]
    assert call == {
        "text": "Hello.",
        "language": Languages.JP,
        "speaker_id": 1,
        "sdp_ratio": 0.3,
        "noise": 0.4,
        "noise_w": 0.5,
        "length": 0.5,
        "line_split": False,
        "assist_text": "Calm delivery.",
        "assist_text_weight": 0.8,
        "use_assist_text": True,
        "style": "Happy",
        "style_weight": 1.5,
    }


def test_streaming_speech_returns_one_pcm_inference_per_sentence(
    client: TestClient, holder: FakeModelHolder
):
    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "First sentence. Second sentence!",
            "stream": True,
            "response_format": "pcm",
        },
    )

    assert response.status_code == 200
    assert response.headers["x-sample-rate"] == "24000"
    assert response.headers["x-channels"] == "1"
    assert response.headers["x-bit-depth"] == "16"
    assert response.content == np.array([1, -1, 2, -2], dtype="<i2").tobytes()
    assert [call["text"] for call in holder.models["voice-a"].infer_calls] == [
        "First sentence.",
        "Second sentence!",
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "First sentence. Second sentence! Last sentence?",
            ["First sentence.", "Second sentence!", "Last sentence?"],
        ),
        (
            "最初の文です。次の文です！最後です",
            ["最初の文です。", "次の文です！", "最後です"],
        ),
        ("One line\nAnother line", ["One line", "Another line"]),
    ],
)
def test_split_sentences_uses_punctuation_and_newlines(text, expected):
    assert split_sentences(text) == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"input": "Hello.", "stream": False, "response_format": "pcm"},
        {"input": "Hello.", "stream": True, "response_format": "wav"},
    ],
)
def test_speech_rejects_unsupported_stream_format_combinations(
    client: TestClient, payload
):
    response = client.post("/v1/audio/speech", json=payload)

    assert response.status_code == 400
    assert isinstance(response.json()["detail"], str)


def test_speech_rejects_unknown_fields_with_422(client: TestClient):
    response = client.post(
        "/v1/audio/speech",
        json={"input": "Hello.", "unsupported": True},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_speech_rejects_unknown_model_with_detail(client: TestClient):
    response = client.post(
        "/v1/audio/speech",
        json={"input": "Hello.", "model": "missing"},
    )

    assert response.status_code == 404
    assert "missing" in response.json()["detail"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("voice", "Missing"),
        ("speaker_id", 99),
        ("style", "Missing"),
    ],
)
def test_speech_rejects_unknown_model_options_with_detail(
    client: TestClient, field, value
):
    response = client.post(
        "/v1/audio/speech",
        json={"input": "Hello.", field: value},
    )

    assert response.status_code == 400
    assert str(value) in response.json()["detail"]


def test_unknown_path_and_method_use_fastapi_error_shapes(client: TestClient):
    not_found = client.get("/missing")
    method_not_allowed = client.get("/v1/audio/speech")

    assert not_found.status_code == 404
    assert not_found.json() == {"detail": "Not Found"}
    assert method_not_allowed.status_code == 405
    assert method_not_allowed.json() == {"detail": "Method Not Allowed"}
