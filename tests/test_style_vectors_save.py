# ruff: noqa: E402
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("gradio", reason="requires the gradio optional dependency")
pytest.importorskip("umap", reason="requires the umap-learn optional dependency")
pytest.importorskip(
    "matplotlib", reason="requires the matplotlib optional dependency"
)

from gradio_tabs import style_vectors as sv


def _write_config(path: Path, *, num_styles: int = 1) -> dict:
    config = {
        "data": {
            "num_styles": num_styles,
            "style2id": {"Neutral": 0},
        }
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return config


@pytest.fixture()
def clustering_env(monkeypatch, tmp_path):
    assets_root = tmp_path / "assets"
    assets_root.mkdir()
    monkeypatch.setattr(sv, "assets_root", assets_root)
    monkeypatch.setattr(sv, "mean", np.full(256, 0.5))
    monkeypatch.setattr(sv, "centroids", [np.zeros(256), np.ones(256)])

    model_dir = assets_root / "voice-a"
    model_dir.mkdir()
    existing_vectors = np.full((1, 256), 7.0)
    np.save(model_dir / "style_vectors.npy", existing_vectors)
    return model_dir, existing_vectors


def test_clustering_save_keeps_vectors_when_config_is_missing(clustering_env):
    model_dir, existing_vectors = clustering_env

    message = sv.save_style_vectors_from_clustering("voice-a", "Angry, Happy")

    assert "存在しません" in message
    np.testing.assert_array_equal(
        np.load(model_dir / "style_vectors.npy"), existing_vectors
    )
    assert not (model_dir / "style_vectors.npy.bak").exists()


def test_clustering_save_keeps_vectors_when_style_count_mismatches(clustering_env):
    model_dir, existing_vectors = clustering_env
    _write_config(model_dir / "config.json")

    message = sv.save_style_vectors_from_clustering("voice-a", "Angry")

    assert "スタイルの数が合いません" in message
    np.testing.assert_array_equal(
        np.load(model_dir / "style_vectors.npy"), existing_vectors
    )


def test_clustering_save_keeps_vectors_when_style_names_duplicate(clustering_env):
    model_dir, existing_vectors = clustering_env
    _write_config(model_dir / "config.json")

    message = sv.save_style_vectors_from_clustering("voice-a", "Angry, Angry")

    assert "重複" in message
    np.testing.assert_array_equal(
        np.load(model_dir / "style_vectors.npy"), existing_vectors
    )


def test_clustering_save_writes_vectors_config_and_backups(clustering_env):
    model_dir, existing_vectors = clustering_env
    original_config = _write_config(model_dir / "config.json")

    message = sv.save_style_vectors_from_clustering("voice-a", "Angry, Happy")

    assert "成功" in message
    saved = np.load(model_dir / "style_vectors.npy")
    assert saved.shape == (3, 256)
    np.testing.assert_array_equal(saved[0], np.full(256, 0.5))

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    assert config["data"]["num_styles"] == 3
    assert config["data"]["style2id"] == {"Neutral": 0, "Angry": 1, "Happy": 2}

    np.testing.assert_array_equal(
        np.load(model_dir / "style_vectors.npy.bak"), existing_vectors
    )
    backup_config = json.loads(
        (model_dir / "config.json.bak").read_text(encoding="utf-8")
    )
    assert backup_config == original_config


@pytest.fixture()
def files_env(monkeypatch, tmp_path):
    assets_root = tmp_path / "assets"
    assets_root.mkdir()
    dataset_root = tmp_path / "dataset"
    monkeypatch.setattr(sv, "assets_root", assets_root)
    monkeypatch.setattr(sv, "dataset_root", dataset_root)
    monkeypatch.setattr(sv, "x", np.full((2, 256), 0.25))

    wavs_dir = dataset_root / "voice-a" / "wavs"
    wavs_dir.mkdir(parents=True)
    (wavs_dir / "angry.wav").write_bytes(b"placeholder")
    np.save(wavs_dir / "angry.wav.npy", np.ones(256))

    model_dir = assets_root / "voice-a"
    model_dir.mkdir()
    existing_vectors = np.full((1, 256), 7.0)
    np.save(model_dir / "style_vectors.npy", existing_vectors)
    return model_dir, existing_vectors


def test_files_save_keeps_vectors_when_config_is_missing(files_env):
    model_dir, existing_vectors = files_env

    message = sv.save_style_vectors_from_files("voice-a", "angry.wav", "Angry")

    assert "存在しません" in message
    np.testing.assert_array_equal(
        np.load(model_dir / "style_vectors.npy"), existing_vectors
    )
    assert not (model_dir / "style_vectors.npy.bak").exists()


def test_files_save_writes_vectors_config_and_backups(files_env):
    model_dir, existing_vectors = files_env
    original_config = _write_config(model_dir / "config.json")

    message = sv.save_style_vectors_from_files("voice-a", "angry.wav", "Angry")

    assert "成功" in message
    saved = np.load(model_dir / "style_vectors.npy")
    assert saved.shape == (2, 256)
    np.testing.assert_array_equal(saved[1], np.ones(256))

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    assert config["data"]["num_styles"] == 2
    assert config["data"]["style2id"] == {"Neutral": 0, "Angry": 1}

    np.testing.assert_array_equal(
        np.load(model_dir / "style_vectors.npy.bak"), existing_vectors
    )
    backup_config = json.loads(
        (model_dir / "config.json.bak").read_text(encoding="utf-8")
    )
    assert backup_config == original_config
