# ruff: noqa: E402
import pytest


torch = pytest.importorskip("torch", reason="requires the torch optional dependency")

from style_bert_vits2.constants import DEFAULT_BERT_MODEL_PATHS, Languages
from style_bert_vits2.nlp import bert_models, clean_text, extract_bert_feature


MODEL_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")
BERT_CASES = [
    pytest.param(
        (Languages.EN, "Hello, world!", "microsoft/deberta-v3-large"),
        id="english",
    ),
    pytest.param(
        (Languages.ZH, "你好，世界！", "hfl/chinese-roberta-wwm-ext-large"),
        id="chinese",
    ),
]


def _bert_model_source(language: Languages, fallback_model_id: str) -> str:
    model_path = DEFAULT_BERT_MODEL_PATHS[language]
    if any((model_path / name).is_file() for name in MODEL_WEIGHT_NAMES):
        return str(model_path)

    return fallback_model_id


@pytest.fixture(params=BERT_CASES)
def loaded_multilingual_bert(request):
    language, text, fallback_model_id = request.param
    model_source = _bert_model_source(language, fallback_model_id)
    try:
        model = bert_models.load_model(
            language,
            pretrained_model_name_or_path=model_source,
        )
        bert_models.load_tokenizer(
            language,
            pretrained_model_name_or_path=model_source,
        )
    except (OSError, RuntimeError, ValueError) as e:
        pytest.skip(f"{language.name} BERT model is unavailable: {e}")

    try:
        yield language, text, model
    finally:
        bert_models.unload_model(language)
        bert_models.unload_tokenizer(language)


def test_multilingual_bert_feature_cpu_contract(loaded_multilingual_bert):
    language, text, model = loaded_multilingual_bert
    normalized_text, phones, _tones, word2ph, _sep_text, _sep_kata, _sep_kata_with_joshi = (
        clean_text(text, language)
    )

    feature = extract_bert_feature(
        normalized_text,
        word2ph,
        language,
        "cpu",
    )

    assert feature.shape[0] == model.config.hidden_size
    assert feature.shape[1] == len(phones)
    assert feature.shape[1] == sum(word2ph)
    assert feature.dtype == torch.float32
