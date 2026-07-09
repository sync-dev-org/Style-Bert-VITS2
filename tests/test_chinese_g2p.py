from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import clean_text


def test_chinese_g2p_known_text_snapshot():
    normalized_text, phones, tones, word2ph, sep_text, sep_kata, sep_kata_with_joshi = (
        clean_text("你好，世界！", Languages.ZH)
    )

    assert normalized_text == "你好,世界!"
    assert phones == [
        "_",
        "n",
        "i",
        "h",
        "ao",
        ",",
        "sh",
        "ir",
        "j",
        "ie",
        "!",
        "_",
    ]
    assert tones == [0, 2, 2, 3, 3, 0, 4, 4, 4, 4, 0, 0]
    assert word2ph == [1, 2, 2, 1, 2, 2, 1, 1]
    assert sep_text is None
    assert sep_kata is None
    assert sep_kata_with_joshi is None
    assert len(phones) == len(tones) == sum(word2ph)
