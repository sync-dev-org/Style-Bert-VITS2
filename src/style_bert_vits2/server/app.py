from __future__ import annotations

import threading
import wave
from collections.abc import Iterator
from io import BytesIO
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response, StreamingResponse

from style_bert_vits2.constants import Languages
from style_bert_vits2.server.schemas import (
    HealthResponse,
    ModelListResponse,
    ModelObject,
    SpeechRequest,
)


_SENTENCE_ENDINGS = frozenset(".!?。！？")


def split_sentences(text: str) -> list[str]:
    """Split text at sentence punctuation and newlines while retaining punctuation."""
    sentences: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        sentence = "".join(buffer).strip()
        buffer.clear()
        if sentence:
            sentences.append(sentence)

    for index, character in enumerate(text):
        if character == "\n":
            flush()
            continue
        buffer.append(character)
        if character not in _SENTENCE_ENDINGS:
            continue
        next_character = text[index + 1] if index + 1 < len(text) else ""
        if next_character not in _SENTENCE_ENDINGS:
            flush()

    flush()
    return sentences


def _pcm16_bytes(audio: Any) -> bytes:
    pcm = np.asarray(audio).reshape(-1)
    if pcm.dtype != np.int16:
        pcm = pcm.astype(np.int16)
    return pcm.astype("<i2", copy=False).tobytes()


def _wav_bytes(sample_rate: int, pcm: bytes) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()


class _ModelRegistry:
    def __init__(self, model_holder: Any):
        self.model_holder = model_holder
        self.model_names = tuple(model_holder.model_names)
        if not self.model_names:
            raise ValueError("No models are available to serve")
        self.default_model = self.model_names[0]
        self._models: dict[str, Any] = {}
        self._model_locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    def get(self, requested_name: str | None) -> tuple[str, Any, threading.Lock]:
        model_name = requested_name or self.default_model
        if model_name not in self.model_names:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name!r} was not found",
            )

        with self._registry_lock:
            if model_name not in self._models:
                model_path = self.model_holder.model_files_dict[model_name][0]
                self._models[model_name] = self.model_holder.get_model(
                    model_name, str(model_path)
                )
                self._model_locks[model_name] = threading.Lock()
            return (
                model_name,
                self._models[model_name],
                self._model_locks[model_name],
            )


def _speaker_id(model: Any, request: SpeechRequest) -> int:
    if request.voice is not None:
        if request.voice not in model.spk2id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Voice {request.voice!r} was not found",
            )
        return int(model.spk2id[request.voice])

    speaker_id = 0 if request.speaker_id is None else request.speaker_id
    known_speaker_ids = set(getattr(model, "id2spk", {}).keys())
    if not known_speaker_ids:
        known_speaker_ids = set(model.spk2id.values())
    if speaker_id not in known_speaker_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Speaker ID {speaker_id!r} was not found",
        )
    return speaker_id


def _validate_style(model: Any, style_name: str) -> None:
    if style_name not in model.style2id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Style {style_name!r} was not found",
        )


def _infer(
    model: Any,
    model_lock: threading.Lock,
    request: SpeechRequest,
    *,
    text: str,
    language: Languages,
    speaker_id: int,
) -> tuple[int, bytes]:
    try:
        with model_lock:
            sample_rate, audio = model.infer(
                text=text,
                language=language,
                speaker_id=speaker_id,
                sdp_ratio=request.sdp_ratio,
                noise=request.noise,
                noise_w=request.noise_w,
                length=1.0 / request.speed,
                line_split=False,
                assist_text=request.assist_text,
                assist_text_weight=request.assist_text_weight,
                use_assist_text=bool(request.assist_text),
                style=request.style,
                style_weight=request.style_weight,
            )
    except (KeyError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    return int(sample_rate), _pcm16_bytes(audio)


def create_app(
    model_holder: Any,
    default_language: Languages | str = Languages.JP,
) -> FastAPI:
    registry = _ModelRegistry(model_holder)
    resolved_default_language = Languages(default_language)
    app = FastAPI()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/v1/models", response_model=ModelListResponse)
    def models() -> ModelListResponse:
        return ModelListResponse(
            data=[ModelObject(id=model_name) for model_name in registry.model_names]
        )

    @app.post("/v1/audio/speech")
    def speech(request: SpeechRequest):
        if request.stream and request.response_format != "pcm":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Streaming responses require response_format='pcm'",
            )
        if not request.stream and request.response_format != "wav":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Buffered responses require response_format='wav'",
            )

        _, model, model_lock = registry.get(request.model)
        speaker_id = _speaker_id(model, request)
        _validate_style(model, request.style)
        language = request.language or resolved_default_language

        if not request.stream:
            sample_rate, pcm = _infer(
                model,
                model_lock,
                request,
                text=request.input,
                language=language,
                speaker_id=speaker_id,
            )
            return Response(
                content=_wav_bytes(sample_rate, pcm), media_type="audio/wav"
            )

        sentences = split_sentences(request.input)
        if not sentences:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input contains no text to synthesize",
            )
        sample_rate, first_pcm = _infer(
            model,
            model_lock,
            request,
            text=sentences[0],
            language=language,
            speaker_id=speaker_id,
        )

        def iter_pcm() -> Iterator[bytes]:
            yield first_pcm
            for sentence in sentences[1:]:
                chunk_sample_rate, pcm = _infer(
                    model,
                    model_lock,
                    request,
                    text=sentence,
                    language=language,
                    speaker_id=speaker_id,
                )
                if chunk_sample_rate != sample_rate:
                    raise RuntimeError(
                        "The model changed sample rate during a streaming response"
                    )
                yield pcm

        return StreamingResponse(
            iter_pcm(),
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": str(sample_rate),
                "X-Channels": "1",
                "X-Bit-Depth": "16",
            },
        )

    return app
