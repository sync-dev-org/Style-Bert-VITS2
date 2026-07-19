"""OpenAI-compatible HTTP server for Style-Bert-VITS2."""

from style_bert_vits2.server.app import create_app


__all__ = ["create_app"]
