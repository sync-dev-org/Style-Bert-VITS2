from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.tts_model import TTSModelHolder


CPU_ONNX_PROVIDER = (
    "CPUExecutionProvider",
    {"arena_extend_strategy": "kSameAsRequested"},
)


def _write_model_asset(
    root: Path,
    name: str,
    *,
    model_files: tuple[str, ...] = ("first.safetensors",),
    speakers: dict[str, int] | None = None,
    styles: dict[str, int] | None = None,
) -> Path:
    model_dir = root / name
    model_dir.mkdir()
    for model_file in model_files:
        (model_dir / model_file).write_bytes(b"placeholder")

    hps = HyperParameters()
    if speakers is not None:
        hps.data.spk2id = speakers
        hps.data.n_speakers = len(speakers)
    if styles is not None:
        hps.data.style2id = styles
        hps.data.num_styles = len(styles)
    (model_dir / "config.json").write_text(
        json.dumps(hps.model_dump(mode="json")),
        encoding="utf-8",
    )
    np.save(model_dir / "style_vectors.npy", np.zeros((hps.data.num_styles, 256)))
    return model_dir


def _holder(root: Path, *, ignore_onnx: bool = False) -> TTSModelHolder:
    return TTSModelHolder(root, "cpu", [CPU_ONNX_PROVIDER], ignore_onnx=ignore_onnx)


def test_model_holder_lists_valid_model_assets(tmp_path):
    model_dir = _write_model_asset(
        tmp_path,
        "voice-a",
        model_files=("voice-a.safetensors", "voice-a.onnx"),
        speakers={"Alice": 0, "Bob": 1},
        styles={"Neutral": 0, "Happy": 1},
    )
    _write_model_asset(tmp_path, ".hidden")
    missing_config_dir = tmp_path / "missing-config"
    missing_config_dir.mkdir()
    (missing_config_dir / "model.safetensors").write_bytes(b"placeholder")

    holder = _holder(tmp_path)

    assert holder.model_names == ["voice-a"]
    assert set(holder.model_files_dict) == {"voice-a"}
    assert [path.name for path in holder.model_files_dict["voice-a"]] == [
        "voice-a.onnx",
        "voice-a.safetensors",
    ]
    assert holder.models_info[0].name == "voice-a"
    assert holder.models_info[0].files == [
        str(model_dir / "voice-a.onnx"),
        str(model_dir / "voice-a.safetensors"),
    ]
    assert holder.models_info[0].speakers == ["Alice", "Bob"]
    assert holder.models_info[0].styles == ["Neutral", "Happy"]


def test_model_holder_refresh_rebuilds_model_file_index(tmp_path):
    model_dir = _write_model_asset(tmp_path, "voice-a")
    holder = _holder(tmp_path)
    assert [path.name for path in holder.model_files_dict["voice-a"]] == [
        "first.safetensors"
    ]

    time.sleep(0.01)
    (model_dir / "second.safetensors").write_bytes(b"placeholder")
    holder.refresh()

    assert [path.name for path in holder.model_files_dict["voice-a"]] == [
        "second.safetensors",
        "first.safetensors",
    ]


def test_model_holder_can_exclude_onnx_files(tmp_path):
    _write_model_asset(
        tmp_path,
        "voice-a",
        model_files=("voice-a.safetensors", "voice-a.onnx"),
    )

    holder = _holder(tmp_path, ignore_onnx=True)

    assert [path.suffix for path in holder.model_files_dict["voice-a"]] == [
        ".safetensors"
    ]


def test_model_holder_rejects_unknown_model_and_model_file(tmp_path):
    model_dir = _write_model_asset(tmp_path, "voice-a")
    holder = _holder(tmp_path)

    with pytest.raises(ValueError, match="Model `missing` is not found"):
        holder.get_model("missing", str(model_dir / "first.safetensors"))

    with pytest.raises(ValueError, match="Model file `.*missing\\.safetensors` is not found"):
        holder.get_model("voice-a", str(model_dir / "missing.safetensors"))


def test_model_holder_instantiates_requested_model_without_loading_weights(tmp_path):
    model_dir = _write_model_asset(tmp_path, "voice-a")
    holder = _holder(tmp_path)

    model = holder.get_model("voice-a", str(model_dir / "first.safetensors"))

    assert model.model_path == model_dir / "first.safetensors"
    assert model.config_path == model_dir / "config.json"
    assert model.style_vec_path == model_dir / "style_vectors.npy"
    assert model.net_g is None
    assert model.onnx_session is None
