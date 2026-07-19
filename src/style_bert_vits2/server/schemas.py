from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    input: str = Field(min_length=1)
    model: str | None = None
    voice: str | None = None
    response_format: Literal["wav", "pcm"] = "wav"
    speed: float = Field(default=1.0, gt=0)
    stream: bool = False

    language: Languages | None = None
    speaker_id: int | None = None
    style: str = DEFAULT_STYLE
    style_weight: float = DEFAULT_STYLE_WEIGHT
    sdp_ratio: float = DEFAULT_SDP_RATIO
    noise: float = DEFAULT_NOISE
    noise_w: float = DEFAULT_NOISEW
    assist_text: str | None = None
    assist_text_weight: float = DEFAULT_ASSIST_TEXT_WEIGHT

    @field_validator("language", mode="before")
    @classmethod
    def parse_language(cls, value: object) -> object:
        if isinstance(value, str):
            return Languages(value)
        return value


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
