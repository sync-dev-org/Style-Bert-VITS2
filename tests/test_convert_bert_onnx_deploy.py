# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path

import pytest


pytest.importorskip("onnx", reason="requires the onnx optional dependency")
pytest.importorskip("onnxsim", reason="requires the onnxsim optional dependency")
pytest.importorskip(
    "onnxconverter_common",
    reason="requires the onnxconverter-common optional dependency",
)
pytest.importorskip("torch", reason="requires the torch optional dependency")

from convert_bert_onnx import deploy_to_onnx_model_dir


def _write_converted_bert_dir(root: Path) -> Path:
    source_dir = root / "bert-pytorch"
    source_dir.mkdir()
    (source_dir / "model.onnx").write_bytes(b"fp32 model")
    (source_dir / "model_fp16.onnx").write_bytes(b"fp16 model")
    (source_dir / "model_temp.onnx").write_bytes(b"temp model")
    (source_dir / "pytorch_model.bin").write_bytes(b"torch weights")
    (source_dir / "config.json").write_text("{}", encoding="utf-8")
    (source_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (source_dir / "vocab.txt").write_text("[PAD]\n", encoding="utf-8")
    (source_dir / "spm.model").write_bytes(b"sentencepiece")
    return source_dir


def test_deploy_copies_onnx_models_and_tokenizer_files(tmp_path):
    source_dir = _write_converted_bert_dir(tmp_path)
    deploy_dir = tmp_path / "bert-onnx"

    deploy_to_onnx_model_dir(source_dir, deploy_dir)

    assert (deploy_dir / "model.onnx").read_bytes() == b"fp32 model"
    assert (deploy_dir / "model_fp16.onnx").read_bytes() == b"fp16 model"
    assert (deploy_dir / "config.json").exists()
    assert (deploy_dir / "tokenizer_config.json").exists()
    assert (deploy_dir / "vocab.txt").exists()
    assert (deploy_dir / "spm.model").exists()

    # 変換の一時ファイルと PyTorch 重みは配置しない
    assert not (deploy_dir / "model_temp.onnx").exists()
    assert not (deploy_dir / "pytorch_model.bin").exists()
    # 変換元に存在しない tokenizer ファイルは配置しない
    assert not (deploy_dir / "merges.txt").exists()


def test_deploy_overwrites_existing_deployed_models(tmp_path):
    source_dir = _write_converted_bert_dir(tmp_path)
    deploy_dir = tmp_path / "bert-onnx"
    deploy_dir.mkdir()
    (deploy_dir / "model_fp16.onnx").write_bytes(b"outdated model")

    deploy_to_onnx_model_dir(source_dir, deploy_dir)

    assert (deploy_dir / "model_fp16.onnx").read_bytes() == b"fp16 model"


def test_deploy_preserves_existing_tokenizer_files(tmp_path):
    # 既定 ONNX ディレクトリの tokenizer / 設定ファイルは git tracked な配布資産のため、
    # 配置は欠損補完のみ行い、既存ファイルを上書きしない
    source_dir = _write_converted_bert_dir(tmp_path)
    deploy_dir = tmp_path / "bert-onnx"
    deploy_dir.mkdir()
    (deploy_dir / "tokenizer_config.json").write_text(
        '{"distributed": true}', encoding="utf-8"
    )

    deploy_to_onnx_model_dir(source_dir, deploy_dir)

    assert (deploy_dir / "tokenizer_config.json").read_text(encoding="utf-8") == (
        '{"distributed": true}'
    )
    # 配置先に無い tokenizer ファイルは補完する
    assert (deploy_dir / "vocab.txt").exists()


def test_deploy_requires_fp16_model(tmp_path):
    source_dir = tmp_path / "bert-pytorch"
    source_dir.mkdir()
    (source_dir / "model.onnx").write_bytes(b"fp32 model")

    with pytest.raises(FileNotFoundError):
        deploy_to_onnx_model_dir(source_dir, tmp_path / "bert-onnx")
