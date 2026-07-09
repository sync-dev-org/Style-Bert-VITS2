from pathlib import Path

import pytest

from style_bert_vits2.constants import DEFAULT_BERT_MODEL_PATHS, Languages
from style_bert_vits2.nlp import bert_models


EN_BERT_MODEL_ID = "microsoft/deberta-v3-large"
ZH_BERT_MODEL_ID = "hfl/chinese-roberta-wwm-ext-large"


def _load_tokenizer_or_skip(language: Languages, source: str, cache_dir: Path):
    try:
        return bert_models.load_tokenizer(
            language,
            pretrained_model_name_or_path=source,
            cache_dir=str(cache_dir),
        )
    except (AssertionError, OSError, RuntimeError, ValueError) as e:
        pytest.skip(f"{language.name} BERT tokenizer is unavailable: {e}")


def _english_tokenizer_source() -> str:
    local_path = DEFAULT_BERT_MODEL_PATHS[Languages.EN]
    if (local_path / "spm.model").is_file():
        return str(local_path)
    return EN_BERT_MODEL_ID


def test_english_bert_tokenizer_loads_from_sentencepiece_path(tmp_path):
    tokenizer = _load_tokenizer_or_skip(
        Languages.EN,
        _english_tokenizer_source(),
        tmp_path / "hf-cache",
    )
    try:
        text = "Style-Bert-VITS2 tests English tokenization."

        assert Path(tokenizer.vocab_file).name == "spm.model"
        assert tokenizer.tokenize(text) == [
            "▁Style",
            "-",
            "Bert",
            "-",
            "V",
            "ITS",
            "2",
            "▁tests",
            "▁English",
            "▁token",
            "ization",
            ".",
        ]
        assert tokenizer(text, add_special_tokens=True)["input_ids"] == [
            1,
            6780,
            271,
            90294,
            271,
            1989,
            44681,
            445,
            2850,
            1342,
            10704,
            4820,
            260,
            2,
        ]
    finally:
        bert_models.unload_tokenizer(Languages.EN)


def test_chinese_bert_tokenizer_loads_and_tokenizes_known_text(tmp_path):
    local_path = DEFAULT_BERT_MODEL_PATHS[Languages.ZH]
    tokenizer = _load_tokenizer_or_skip(
        Languages.ZH,
        str(local_path) if local_path.exists() else ZH_BERT_MODEL_ID,
        tmp_path / "hf-cache",
    )
    try:
        text = "今天天气很好。"

        assert tokenizer.tokenize(text) == ["今", "天", "天", "气", "很", "好", "。"]
        assert tokenizer(text, add_special_tokens=True)["input_ids"] == [
            101,
            791,
            1921,
            1921,
            3698,
            2523,
            1962,
            511,
            102,
        ]
    finally:
        bert_models.unload_tokenizer(Languages.ZH)
