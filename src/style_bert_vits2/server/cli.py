from __future__ import annotations

import argparse
from pathlib import Path

from style_bert_vits2.constants import Languages


def _parser(default_dir: Path, default_language: Languages) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Style-Bert-VITS2 OpenAI-compatible TTS server."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1927)
    parser.add_argument("--dir", type=Path, default=default_dir)
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--language",
        choices=[language.value for language in Languages],
        default=default_language.value,
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

    from config import get_config
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
    from style_bert_vits2.nlp.japanese.user_dict import update_dict
    from style_bert_vits2.server.app import create_app
    from style_bert_vits2.tts_model import TTSModelHolder
    from style_bert_vits2.utils import torch_device_to_onnx_providers

    config = get_config()
    default_language = Languages(config.server_config.language)
    parser = _parser(config.assets_root, default_language)
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    language = Languages(args.language)

    pyopenjtalk.initialize_worker()
    update_dict()

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
