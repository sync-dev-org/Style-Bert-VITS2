import inspect
from typing import Any, Callable

from style_bert_vits2.nlp.japanese.normalizer import replace_punctuation
from style_bert_vits2.nlp.symbols import PUNCTUATIONS


RUN_FRONTEND_COMPAT_KWARGS: dict[str, Any] = {
    "run_marine": False,
    "use_vanilla": False,
    "use_sudachi_kanji_yomi": True,
    "predict_nani": True,
    "normalize_mode": "None",
    "use_read_as_pron": False,
    "revert_long_vowels": False,
    "revert_yotsugana": False,
}


def run_frontend(text: str) -> list[dict[str, Any]]:
    import pyopenjtalk

    kwargs = _supported_kwargs(pyopenjtalk.run_frontend, RUN_FRONTEND_COMPAT_KWARGS)
    features = pyopenjtalk.run_frontend(text, **kwargs)
    _normalize_punctuation_pronunciations(features)
    return features


def _supported_kwargs(
    func: Callable[..., Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {}

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return kwargs

    return {key: value for key, value in kwargs.items() if key in parameters}


def _normalize_punctuation_pronunciations(features: list[dict[str, Any]]) -> None:
    for feature in features:
        if feature.get("pos") != "記号" or feature.get("mora_size") != 0:
            continue

        normalized = replace_punctuation(str(feature.get("string", "")))
        if not normalized or not set(normalized).issubset(PUNCTUATIONS):
            continue

        pronunciation = "？" if set(normalized) == {"?"} else "、"
        feature["read"] = pronunciation
        feature["pron"] = pronunciation
