# ruff: noqa: E402
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("gradio", reason="requires the gradio optional dependency")
pytest.importorskip("torch", reason="requires the torch optional dependency")

import gradio as gr

from gradio_tabs.merge import create_merge_app, merge_models_gr
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.tts_model import TTSModelHolder


CPU_ONNX_PROVIDER = (
    "CPUExecutionProvider",
    {"arena_extend_strategy": "kSameAsRequested"},
)


def _write_model_asset(root: Path, name: str) -> Path:
    model_dir = root / name
    model_dir.mkdir()
    (model_dir / "first.safetensors").write_bytes(b"placeholder")

    hps = HyperParameters()
    (model_dir / "config.json").write_text(
        json.dumps(hps.model_dump(mode="json")),
        encoding="utf-8",
    )
    np.save(model_dir / "style_vectors.npy", np.zeros((hps.data.num_styles, 256)))
    return model_dir


@pytest.fixture(scope="module")
def merge_app():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        _write_model_asset(root, "voice-a")
        holder = TTSModelHolder(root, "cpu", [CPU_ONNX_PROVIDER])
        yield create_merge_app(holder)


def test_merge_models_gr_empty_output_name_matches_event_output_arity(merge_app):
    block_fn = next(
        fn
        for fn in merge_app.fns.values()
        if getattr(fn.fn, "__name__", "") == "merge_models_gr"
    )
    n_outputs = len(block_fn.outputs)

    result = merge_models_gr(
        "model_assets/a/a.safetensors",
        "model_assets/b/b.safetensors",
        "model_assets/c/c.safetensors",
        1.0,  # model_a_coeff
        0.0,  # model_b_coeff
        0.0,  # model_c_coeff
        "usual",
        "",  # output_name
        0.5,  # voice_weight
        0.5,  # voice_pitch_weight
        0.5,  # speech_style_weight
        0.5,  # tempo_weight
        False,  # use_slerp_instead_of_lerp
    )

    assert isinstance(result, tuple)
    assert len(result) == n_outputs


def test_merge_app_has_no_leftover_placeholder_markdown(merge_app):
    markdown_values = [
        block.value
        for block in merge_app.blocks.values()
        if isinstance(block, gr.Markdown)
    ]
    assert not any("Hello world" in (value or "") for value in markdown_values)
