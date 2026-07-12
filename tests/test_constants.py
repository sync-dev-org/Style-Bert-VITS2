from style_bert_vits2.constants import DEFAULT_ASSIST_TEXT_WEIGHT


def test_default_assist_text_weight_is_confirmed_effective_value():
    # 実効既定値は 1.0 で確定 (docs/spec/core-inference.md)。
    # 過去に 0.7 の dead 代入が並存していたため、意図しない「復活」を検出する。
    assert DEFAULT_ASSIST_TEXT_WEIGHT == 1.0
