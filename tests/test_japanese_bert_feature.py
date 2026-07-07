import atexit

import pytest
import torch

from style_bert_vits2.constants import DEFAULT_BERT_MODEL_PATHS, Languages
from style_bert_vits2.nlp import bert_models, clean_text, extract_bert_feature
from style_bert_vits2.nlp.japanese import bert_feature as japanese_bert_feature
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.user_dict import update_dict


JP_BERT_MODEL_ID = "ku-nlp/deberta-v2-large-japanese-char-wwm"
MODEL_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")


def _japanese_bert_model_source() -> str:
    model_path = DEFAULT_BERT_MODEL_PATHS[Languages.JP]
    if any((model_path / name).is_file() for name in MODEL_WEIGHT_NAMES):
        return str(model_path)

    return JP_BERT_MODEL_ID


@pytest.fixture(scope="module")
def _loaded_japanese_bert():
    pyopenjtalk.initialize_worker()
    try:
        update_dict()
        model_source = _japanese_bert_model_source()
        try:
            model = bert_models.load_model(
                Languages.JP,
                pretrained_model_name_or_path=model_source,
                device_map="cpu",
            )
            bert_models.load_tokenizer(
                Languages.JP,
                pretrained_model_name_or_path=model_source,
            )
        except (OSError, RuntimeError, ValueError) as e:
            pytest.skip(f"JP BERT model is unavailable: {e}")

        yield model
    finally:
        bert_models.unload_all_models()
        pyopenjtalk.terminate_worker()
        try:
            atexit.unregister(pyopenjtalk.terminate_worker)
        except ValueError:
            pass


def test_japanese_bert_feature_cpu_smoke(_loaded_japanese_bert):
    text = "今日はいい天気です。"
    (
        normalized_text,
        phones,
        _tones,
        word2ph,
        sep_text,
        _sep_kata,
        _sep_kata_with_joshi,
    ) = clean_text(text, Languages.JP)

    feature = extract_bert_feature(
        normalized_text,
        word2ph,
        Languages.JP,
        "cpu",
        sep_text=sep_text,
    )

    assert feature.shape[0] == _loaded_japanese_bert.config.hidden_size
    assert feature.shape[1] == len(phones)
    assert feature.shape[1] == sum(word2ph)
    assert feature.dtype == torch.float32


def test_japanese_bert_model_loads_float32(_loaded_japanese_bert):
    assert next(_loaded_japanese_bert.parameters()).dtype == torch.float32


def test_japanese_bert_feature_uses_provided_sep_text(monkeypatch):
    hidden_size = 3
    word2ph = [1, 1, 1, 1]

    def fail_text_to_sep_kata(*_args, **_kwargs):
        raise AssertionError("text_to_sep_kata should not be called for main text")

    class FakeModel:
        def __call__(self, **_kwargs):
            hidden = torch.arange(4 * hidden_size, dtype=torch.float32).reshape(
                1, 4, hidden_size
            )
            return {"hidden_states": [hidden, hidden, hidden]}

    class FakeTokenizer:
        def __call__(self, text: str, return_tensors: str):
            assert text == "今日"
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor([[0, 1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            }

    monkeypatch.setattr(
        japanese_bert_feature, "text_to_sep_kata", fail_text_to_sep_kata
    )
    monkeypatch.setattr(
        japanese_bert_feature.bert_models,
        "load_model",
        lambda _language, device_map=None: FakeModel(),
    )
    monkeypatch.setattr(
        japanese_bert_feature.bert_models,
        "transfer_model",
        lambda _language, _device: None,
    )
    monkeypatch.setattr(
        japanese_bert_feature.bert_models,
        "load_tokenizer",
        lambda _language: FakeTokenizer(),
    )

    feature = japanese_bert_feature.extract_bert_feature(
        "再パースされない文字列",
        word2ph,
        "cpu",
        sep_text=["今", "日"],
    )

    assert feature.shape == (hidden_size, sum(word2ph))
