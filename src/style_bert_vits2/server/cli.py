from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from style_bert_vits2.constants import (
    DEFAULT_USER_DICT_DIR,
    PACKAGE_DIR,
    Languages,
)
from style_bert_vits2.logging import logger


@dataclass(frozen=True)
class _UserDictionaryPaths:
    default_dict_path: Path | None
    user_dict_path: Path
    compiled_dict_path: Path


def _resolve_user_dictionary_paths(
    repository_dict_dir: Path = DEFAULT_USER_DICT_DIR,
    package_dir: Path = PACKAGE_DIR,
    cache_dict_dir: Path | None = None,
) -> _UserDictionaryPaths:
    if cache_dict_dir is None:
        cache_dict_dir = Path.home() / ".cache" / "style-bert-vits2" / "dict"

    repository_default_dict_path = repository_dict_dir / "default.csv"
    packaged_default_dict_path = package_dir / "dict_data" / "default.csv"
    if repository_default_dict_path.is_file():
        default_dict_path = repository_default_dict_path
    elif packaged_default_dict_path.is_file():
        default_dict_path = packaged_default_dict_path
    else:
        default_dict_path = None

    writable_dict_dir = (
        repository_dict_dir if repository_dict_dir.is_dir() else cache_dict_dir
    )
    return _UserDictionaryPaths(
        default_dict_path=default_dict_path,
        user_dict_path=writable_dict_dir / "user_dict.json",
        compiled_dict_path=writable_dict_dir / "user.dic",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Style-Bert-VITS2 OpenAI-compatible TTS server."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1927)
    parser.add_argument("--dir", type=Path, default=Path("model_assets"))
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--language",
        choices=[language.value for language in Languages],
        default=Languages.JP.value,
    )
    parser.add_argument(
        "--no-preload-bert",
        action="store_true",
        help="Do not preload the Japanese and English BERT models.",
    )
    return parser


def main() -> None:
    import torch
    import uvicorn

    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
    from style_bert_vits2.nlp.japanese.user_dict import update_dict
    from style_bert_vits2.server.app import create_app
    from style_bert_vits2.tts_model import TTSModelHolder
    from style_bert_vits2.utils import torch_device_to_onnx_providers

    parser = _parser()
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    language = Languages(args.language)

    pyopenjtalk.initialize_worker()
    dictionary_paths = _resolve_user_dictionary_paths()
    if dictionary_paths.default_dict_path is None:
        logger.warning("Default user dictionary not found; skipping dictionary update.")
    else:
        dictionary_paths.user_dict_path.parent.mkdir(parents=True, exist_ok=True)
        update_dict(
            default_dict_path=dictionary_paths.default_dict_path,
            user_dict_path=dictionary_paths.user_dict_path,
            compiled_dict_path=dictionary_paths.compiled_dict_path,
        )

    if not args.no_preload_bert:
        for preload_language in (Languages.JP, Languages.EN):
            bert_models.load_model(preload_language, device_map=device)
            bert_models.load_tokenizer(preload_language)

    model_holder = TTSModelHolder(
        args.dir,
        device,
        torch_device_to_onnx_providers(device),
    )
    if not model_holder.model_names:
        parser.error(f"No models found in {args.dir}")

    app = create_app(model_holder, default_language=language)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
    )
