from __future__ import annotations

from types import SimpleNamespace

import pytest

from style_bert_vits2 import constants
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models


EXPECTED_BERT_MODEL_IDS = {
    Languages.JP: "ku-nlp/deberta-v2-large-japanese-char-wwm",
    Languages.EN: "microsoft/deberta-v3-large",
    Languages.ZH: "hfl/chinese-roberta-wwm-ext-large",
}


def test_default_bert_model_ids_cover_all_languages():
    assert constants.DEFAULT_BERT_MODEL_IDS == EXPECTED_BERT_MODEL_IDS


@pytest.mark.parametrize(
    ("loader_name", "backend_name", "unload_name"),
    [
        ("load_model", "AutoModelForMaskedLM", "unload_model"),
        ("load_tokenizer", "AutoTokenizer", "unload_tokenizer"),
    ],
)
@pytest.mark.parametrize("local_model_exists", [True, False])
def test_default_bert_loaders_prefer_local_path_then_hugging_face(
    tmp_path,
    monkeypatch,
    loader_name,
    backend_name,
    unload_name,
    local_model_exists,
):
    local_model_path = tmp_path / "bert-model"
    if local_model_exists:
        local_model_path.mkdir()
    calls: list[tuple[str, dict[str, object]]] = []
    loaded = object()

    def from_pretrained(model_name_or_path, **kwargs):
        calls.append((model_name_or_path, kwargs))
        return loaded

    monkeypatch.setitem(
        bert_models.DEFAULT_BERT_MODEL_PATHS,
        Languages.JP,
        local_model_path,
    )
    monkeypatch.setattr(
        bert_models,
        "DEFAULT_BERT_MODEL_IDS",
        EXPECTED_BERT_MODEL_IDS,
        raising=False,
    )
    monkeypatch.setattr(
        bert_models,
        backend_name,
        SimpleNamespace(from_pretrained=from_pretrained),
    )
    getattr(bert_models, unload_name)(Languages.JP)

    result = getattr(bert_models, loader_name)(Languages.JP)

    expected_source = (
        str(local_model_path)
        if local_model_exists
        else EXPECTED_BERT_MODEL_IDS[Languages.JP]
    )
    assert result is loaded
    assert calls[0][0] == expected_source

    getattr(bert_models, unload_name)(Languages.JP)
