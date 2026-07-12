from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from default_style import save_neutral_vector, save_styles_by_dirs


def _write_config(path: Path) -> dict:
    config = {
        "data": {
            "num_styles": 1,
            "style2id": {"Neutral": 0},
        }
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return config


def _make_flat_wav_dir(tmp_path: Path) -> Path:
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    np.save(wav_dir / "a.wav.npy", np.zeros(256))
    np.save(wav_dir / "b.wav.npy", np.ones(256))
    return wav_dir


def _make_styled_wav_dir(tmp_path: Path) -> Path:
    wav_dir = tmp_path / "wavs"
    for style, value in [("style1", 0.0), ("style2", 1.0)]:
        style_dir = wav_dir / style
        style_dir.mkdir(parents=True)
        np.save(style_dir / "a.wav.npy", np.full(256, value))
    return wav_dir


def test_save_neutral_vector_keeps_vectors_when_config_is_missing(tmp_path):
    wav_dir = _make_flat_wav_dir(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing_vectors = np.full((1, 256), 7.0)
    np.save(output_dir / "style_vectors.npy", existing_vectors)

    with pytest.raises(FileNotFoundError):
        save_neutral_vector(
            wav_dir,
            output_dir,
            tmp_path / "missing-config.json",
            output_dir / "config.json",
        )

    np.testing.assert_array_equal(
        np.load(output_dir / "style_vectors.npy"), existing_vectors
    )


def test_save_neutral_vector_writes_vector_config_and_backups(tmp_path):
    wav_dir = _make_flat_wav_dir(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing_vectors = np.full((1, 256), 7.0)
    np.save(output_dir / "style_vectors.npy", existing_vectors)
    config_path = output_dir / "config.json"
    original_config = _write_config(config_path)

    save_neutral_vector(wav_dir, output_dir, config_path, config_path)

    saved = np.load(output_dir / "style_vectors.npy")
    assert saved.shape == (1, 256)
    np.testing.assert_array_equal(saved[0], np.full(256, 0.5))

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["data"]["num_styles"] == 1
    assert config["data"]["style2id"] == {"Neutral": 0}

    np.testing.assert_array_equal(
        np.load(output_dir / "style_vectors.npy.bak"), existing_vectors
    )
    backup_config = json.loads(
        (output_dir / "config.json.bak").read_text(encoding="utf-8")
    )
    assert backup_config == original_config


def test_save_styles_by_dirs_keeps_vectors_when_config_is_missing(tmp_path):
    wav_dir = _make_styled_wav_dir(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing_vectors = np.full((1, 256), 7.0)
    np.save(output_dir / "style_vectors.npy", existing_vectors)

    with pytest.raises(FileNotFoundError):
        save_styles_by_dirs(
            wav_dir,
            output_dir,
            tmp_path / "missing-config.json",
            output_dir / "config.json",
        )

    np.testing.assert_array_equal(
        np.load(output_dir / "style_vectors.npy"), existing_vectors
    )


def test_save_styles_by_dirs_writes_vectors_config_and_backups(tmp_path):
    wav_dir = _make_styled_wav_dir(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing_vectors = np.full((1, 256), 7.0)
    np.save(output_dir / "style_vectors.npy", existing_vectors)
    config_path = output_dir / "config.json"
    original_config = _write_config(config_path)

    save_styles_by_dirs(wav_dir, output_dir, config_path, config_path)

    saved = np.load(output_dir / "style_vectors.npy")
    assert saved.shape == (3, 256)
    np.testing.assert_array_equal(saved[0], np.full(256, 0.5))

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["data"]["num_styles"] == 3
    assert config["data"]["style2id"] == {"Neutral": 0, "style1": 1, "style2": 2}

    np.testing.assert_array_equal(
        np.load(output_dir / "style_vectors.npy.bak"), existing_vectors
    )
    backup_config = json.loads(
        (output_dir / "config.json.bak").read_text(encoding="utf-8")
    )
    assert backup_config == original_config
