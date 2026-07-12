# ruff: noqa: E402
from __future__ import annotations

import atexit
import json
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("gradio", reason="requires the gradio optional dependency")
pytest.importorskip("torch", reason="requires the torch optional dependency")

from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.nlp import InvalidToneError
from style_bert_vits2.tts_model import TTSModelHolder


CPU_ONNX_PROVIDER = (
    "CPUExecutionProvider",
    {"arena_extend_strategy": "kSameAsRequested"},
)


def _write_model_asset(
    root: Path,
    name: str,
    *,
    speakers: dict[str, int] | None = None,
) -> Path:
    model_dir = root / name
    model_dir.mkdir()
    (model_dir / "first.safetensors").write_bytes(b"placeholder")

    hps = HyperParameters()
    if speakers is not None:
        hps.data.spk2id = speakers
        hps.data.n_speakers = len(speakers)
    (model_dir / "config.json").write_text(
        json.dumps(hps.model_dump(mode="json")),
        encoding="utf-8",
    )
    np.save(model_dir / "style_vectors.npy", np.zeros((hps.data.num_styles, 256)))
    return model_dir


@pytest.fixture(scope="module")
def inference_module():
    # gradio_tabs.inference は import 時に pyopenjtalk worker を起動する
    from gradio_tabs import inference
    from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk

    yield inference
    pyopenjtalk.terminate_worker()
    try:
        atexit.unregister(pyopenjtalk.terminate_worker)
    except ValueError:
        pass


@pytest.mark.parametrize(
    "raised",
    [InvalidToneError("bad tone"), ValueError("bad value")],
    ids=["invalid_tone_error", "value_error"],
)
def test_tts_fn_error_paths_match_event_output_arity(
    inference_module, tmp_path, monkeypatch, raised
):
    model_dir = _write_model_asset(tmp_path, "voice-a", speakers={"Alice": 0})
    holder = TTSModelHolder(tmp_path, "cpu", [CPU_ONNX_PROVIDER])

    app = inference_module.create_inference_app(holder)
    block_fn = next(
        fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "tts_fn"
    )
    n_outputs = len(block_fn.outputs)

    model = holder.get_model("voice-a", str(model_dir / "first.safetensors"))

    def raising_infer(**kwargs):
        raise raised

    monkeypatch.setattr(model, "infer", raising_infer)

    result = block_fn.fn(
        "voice-a",
        str(model_dir / "first.safetensors"),
        "こんにちは",
        "JP",
        None,  # reference_audio_path
        0.2,  # sdp_ratio
        0.6,  # noise_scale
        0.8,  # noise_scale_w
        1.0,  # length_scale
        False,  # line_split
        0.5,  # split_interval
        "",  # assist_text
        0.7,  # assist_text_weight
        False,  # use_assist_text
        "Neutral",  # style
        1.0,  # style_weight
        "",  # kata_tone_json_str
        False,  # use_tone
        "Alice",  # speaker
        1.0,  # pitch_scale
        1.0,  # intonation_scale
        {},  # null_models
        False,  # force_reload_model
    )

    assert isinstance(result, tuple)
    assert len(result) == n_outputs
