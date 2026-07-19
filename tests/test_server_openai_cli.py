from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from style_bert_vits2.constants import Languages
from style_bert_vits2.server import app as app_module
from style_bert_vits2.server import cli


class FakeHolder:
    calls: list[tuple[Path, str, Any]] = []

    def __init__(self, model_dir: Path, device: str, providers: Any):
        self.calls.append((model_dir, device, providers))
        self.model_names = ["voice-a"]
        self.model_files_dict = {"voice-a": [model_dir / "voice-a.safetensors"]}


def test_main_initializes_runtime_and_starts_uvicorn_with_cli_values(
    tmp_path, monkeypatch
):
    calls: dict[str, Any] = {
        "worker": 0,
        "dict": 0,
        "bert_models": [],
        "bert_tokenizers": [],
    }
    fake_app = object()

    monkeypatch.setattr(
        "config.get_config",
        lambda: SimpleNamespace(
            assets_root=tmp_path / "configured-models",
            server_config=SimpleNamespace(language="JP"),
        ),
    )
    monkeypatch.setattr(
        "style_bert_vits2.nlp.japanese.pyopenjtalk_worker.initialize_worker",
        lambda: calls.__setitem__("worker", calls["worker"] + 1),
    )
    monkeypatch.setattr(
        "style_bert_vits2.nlp.japanese.user_dict.update_dict",
        lambda: calls.__setitem__("dict", calls["dict"] + 1),
    )
    monkeypatch.setattr(
        "style_bert_vits2.nlp.bert_models.load_model",
        lambda language, device_map: calls["bert_models"].append(
            (language, device_map)
        ),
    )
    monkeypatch.setattr(
        "style_bert_vits2.nlp.bert_models.load_tokenizer",
        lambda language: calls["bert_tokenizers"].append(language),
    )
    monkeypatch.setattr(
        "style_bert_vits2.tts_model.TTSModelHolder",
        FakeHolder,
    )
    monkeypatch.setattr(
        "style_bert_vits2.utils.torch_device_to_onnx_providers",
        lambda device: [f"{device}-provider"],
    )
    create_app_calls: list[tuple[Any, Languages]] = []
    monkeypatch.setattr(
        app_module,
        "create_app",
        lambda holder, default_language: (
            create_app_calls.append((holder, default_language)),
            fake_app,
        )[1],
    )
    uvicorn_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "uvicorn.run",
        lambda app, **kwargs: uvicorn_calls.append({"app": app, **kwargs}),
    )
    model_dir = tmp_path / "models"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "style_bert_vits2.server",
            "--host",
            "127.0.0.2",
            "--port",
            "18927",
            "--dir",
            str(model_dir),
            "--device",
            "cpu",
            "--language",
            "EN",
        ],
    )

    cli.main()

    assert calls == {
        "worker": 1,
        "dict": 1,
        "bert_models": [(Languages.JP, "cpu"), (Languages.EN, "cpu")],
        "bert_tokenizers": [Languages.JP, Languages.EN],
    }
    assert FakeHolder.calls[-1] == (model_dir, "cpu", ["cpu-provider"])
    assert create_app_calls[0][1] == Languages.EN
    assert uvicorn_calls == [
        {
            "app": fake_app,
            "host": "127.0.0.2",
            "port": 18927,
            "log_level": "warning",
        }
    ]


def test_main_can_disable_bert_preload(tmp_path, monkeypatch):
    model_loads: list[Any] = []
    tokenizer_loads: list[Any] = []

    monkeypatch.setattr(
        "config.get_config",
        lambda: SimpleNamespace(
            assets_root=tmp_path,
            server_config=SimpleNamespace(language="JP"),
        ),
    )
    monkeypatch.setattr(
        "style_bert_vits2.nlp.japanese.pyopenjtalk_worker.initialize_worker",
        lambda: None,
    )
    monkeypatch.setattr(
        "style_bert_vits2.nlp.japanese.user_dict.update_dict",
        lambda: None,
    )
    monkeypatch.setattr(
        "style_bert_vits2.nlp.bert_models.load_model",
        lambda *args, **kwargs: model_loads.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "style_bert_vits2.nlp.bert_models.load_tokenizer",
        lambda *args, **kwargs: tokenizer_loads.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "style_bert_vits2.tts_model.TTSModelHolder",
        FakeHolder,
    )
    monkeypatch.setattr(app_module, "create_app", lambda *args, **kwargs: object())
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["style_bert_vits2.server", "--device", "cpu", "--no-preload-bert"],
    )

    cli.main()

    assert model_loads == []
    assert tokenizer_loads == []
