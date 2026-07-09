import pytest

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models, clean_text


pytest.importorskip("g2p_en", reason="requires the g2p_en optional dependency")


def _clean_english_text(text: str):
    try:
        return clean_text(text, Languages.EN)
    except LookupError as e:
        pytest.skip(f"EN g2p data is unavailable: {e}")


def test_english_g2p_known_text_snapshot():
    normalized_text, phones, tones, word2ph, sep_text, sep_kata, sep_kata_with_joshi = (
        _clean_english_text("Hello, world!")
    )

    try:
        assert normalized_text == "Hello, world!"
        assert phones == [
            "_",
            "hh",
            "ah",
            "l",
            "ow",
            ",",
            "w",
            "er",
            "l",
            "d",
            "!",
            "_",
        ]
        assert tones == [0, 3, 1, 3, 2, 0, 3, 2, 3, 3, 0, 0]
        assert word2ph == [1, 4, 1, 4, 1, 1]
        assert sep_text is None
        assert sep_kata is None
        assert sep_kata_with_joshi is None
        assert len(phones) == len(tones) == sum(word2ph)
    finally:
        bert_models.unload_tokenizer(Languages.EN)
