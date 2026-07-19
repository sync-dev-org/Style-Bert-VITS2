from __future__ import annotations

import pytest
from pydantic import ValidationError

from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
)
from style_bert_vits2.server.schemas import SpeechRequest


def test_speech_request_uses_documented_defaults():
    request = SpeechRequest(input="Hello.")

    assert request.model is None
    assert request.voice is None
    assert request.response_format == "wav"
    assert request.speed == 1.0
    assert request.stream is False
    assert request.language is None
    assert request.speaker_id is None
    assert request.style == DEFAULT_STYLE
    assert request.style_weight == DEFAULT_STYLE_WEIGHT
    assert request.sdp_ratio == DEFAULT_SDP_RATIO
    assert request.noise == DEFAULT_NOISE
    assert request.noise_w == DEFAULT_NOISEW
    assert request.assist_text is None
    assert request.assist_text_weight == DEFAULT_ASSIST_TEXT_WEIGHT


def test_speech_request_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SpeechRequest(input="Hello.", unsupported=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("response_format", "mp3"),
        ("language", "FR"),
        ("speed", 0),
        ("speed", -1),
    ],
)
def test_speech_request_rejects_invalid_values(field, value):
    with pytest.raises(ValidationError):
        SpeechRequest(input="Hello.", **{field: value})
