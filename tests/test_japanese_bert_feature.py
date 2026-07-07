import atexit

import pytest

from style_bert_vits2.constants import DEFAULT_BERT_MODEL_PATHS, Languages
from style_bert_vits2.nlp import bert_models, clean_text, extract_bert_feature
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
    normalized_text, phones, _tones, word2ph = clean_text(text, Languages.JP)

    feature = extract_bert_feature(
        normalized_text,
        word2ph,
        Languages.JP,
        "cpu",
    )

    assert feature.shape[0] == _loaded_japanese_bert.config.hidden_size
    assert feature.shape[1] == len(phones)
    assert feature.shape[1] == sum(word2ph)
