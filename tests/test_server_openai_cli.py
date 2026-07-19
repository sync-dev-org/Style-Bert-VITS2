from __future__ import annotations

import builtins
import sys
from pathlib import Path
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
        "dict": [],
        "bert_models": [],
        "bert_tokenizers": [],
    }
    fake_app = object()
    dictionary_paths = cli._UserDictionaryPaths(
        default_dict_path=tmp_path / "default.csv",
        user_dict_path=tmp_path / "user_dict.json",
        compiled_dict_path=tmp_path / "user.dic",
    )
    monkeypatch.setattr(cli, "_resolve_user_dictionary_paths", lambda: dictionary_paths)
    monkeypatch.setattr(
        "style_bert_vits2.nlp.japanese.pyopenjtalk_worker.initialize_worker",
        lambda: calls.__setitem__("worker", calls["worker"] + 1),
    )
    monkeypatch.setattr(
        "style_bert_vits2.nlp.japanese.user_dict.update_dict",
        lambda **kwargs: calls["dict"].append(kwargs),
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
        "dict": [
            {
                "default_dict_path": dictionary_paths.default_dict_path,
                "user_dict_path": dictionary_paths.user_dict_path,
                "compiled_dict_path": dictionary_paths.compiled_dict_path,
            }
        ],
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


def test_main_uses_package_defaults_without_importing_repository_config(
    tmp_path, monkeypatch
):
    model_loads: list[Any] = []
    tokenizer_loads: list[Any] = []
    dictionary_updates: list[Any] = []
    warnings: list[str] = []

    monkeypatch.setattr(
        cli,
        "_resolve_user_dictionary_paths",
        lambda: cli._UserDictionaryPaths(
            default_dict_path=None,
            user_dict_path=tmp_path / "user_dict.json",
            compiled_dict_path=tmp_path / "user.dic",
        ),
    )
    monkeypatch.setattr(
        "style_bert_vits2.nlp.japanese.pyopenjtalk_worker.initialize_worker",
        lambda: None,
    )
    monkeypatch.setattr(
        "style_bert_vits2.nlp.japanese.user_dict.update_dict",
        lambda **kwargs: dictionary_updates.append(kwargs),
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
    monkeypatch.setattr(cli.logger, "warning", warnings.append)
    monkeypatch.setattr(
        sys,
        "argv",
        ["style_bert_vits2.server", "--device", "cpu", "--no-preload-bert"],
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "config":
            raise AssertionError("The package CLI must not import repository config.py")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    cli.main()

    assert model_loads == []
    assert tokenizer_loads == []
    assert dictionary_updates == []
    assert warnings == [
        "Default user dictionary not found; skipping dictionary update."
    ]
    assert FakeHolder.calls[-1][0] == Path("model_assets")


def test_parser_defaults_to_cwd_model_assets_and_japanese():
    args = cli._parser().parse_args([])

    assert args.dir == Path("model_assets")
    assert args.language == Languages.JP.value


def test_dictionary_paths_use_repository_directory_when_present(tmp_path):
    repository_dict_dir = tmp_path / "repository-dict"
    repository_dict_dir.mkdir()
    default_dict_path = repository_dict_dir / "default.csv"
    default_dict_path.touch()

    paths = cli._resolve_user_dictionary_paths(
        repository_dict_dir=repository_dict_dir,
        package_dir=tmp_path / "package",
        cache_dict_dir=tmp_path / "cache",
    )

    assert paths == cli._UserDictionaryPaths(
        default_dict_path=default_dict_path,
        user_dict_path=repository_dict_dir / "user_dict.json",
        compiled_dict_path=repository_dict_dir / "user.dic",
    )


def test_dictionary_paths_use_packaged_default_and_user_cache(tmp_path):
    package_dir = tmp_path / "package"
    packaged_dict_dir = package_dir / "dict_data"
    packaged_dict_dir.mkdir(parents=True)
    packaged_default_dict_path = packaged_dict_dir / "default.csv"
    packaged_default_dict_path.touch()
    cache_dict_dir = tmp_path / "cache"

    paths = cli._resolve_user_dictionary_paths(
        repository_dict_dir=tmp_path / "missing-repository-dict",
        package_dir=package_dir,
        cache_dict_dir=cache_dict_dir,
    )

    assert paths == cli._UserDictionaryPaths(
        default_dict_path=packaged_default_dict_path,
        user_dict_path=cache_dict_dir / "user_dict.json",
        compiled_dict_path=cache_dict_dir / "user.dic",
    )


def test_dictionary_paths_report_missing_default_and_use_user_cache(tmp_path):
    cache_dict_dir = tmp_path / "cache"

    paths = cli._resolve_user_dictionary_paths(
        repository_dict_dir=tmp_path / "missing-repository-dict",
        package_dir=tmp_path / "package",
        cache_dict_dir=cache_dict_dir,
    )

    assert paths == cli._UserDictionaryPaths(
        default_dict_path=None,
        user_dict_path=cache_dict_dir / "user_dict.json",
        compiled_dict_path=cache_dict_dir / "user.dic",
    )
