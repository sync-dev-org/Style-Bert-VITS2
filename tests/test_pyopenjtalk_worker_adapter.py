import sys
from types import ModuleType


def _fake_punctuation_feature() -> dict[str, object]:
    return {
        "string": "！",
        "pos": "記号",
        "pos_group1": "一般",
        "read": "！",
        "pron": "！",
        "acc": 0,
        "mora_size": 0,
        "chain_rule": "*",
        "chain_flag": 0,
    }


def test_run_frontend_uses_pyopenjtalk_plus_compatibility_kwargs(monkeypatch):
    from style_bert_vits2.nlp.japanese.pyopenjtalk_worker import adapter

    recorded: dict[str, object] = {}
    fake_pyopenjtalk = ModuleType("pyopenjtalk")

    def run_frontend(
        text: str,
        *,
        run_marine: bool = True,
        use_vanilla: bool = True,
        use_sudachi_kanji_yomi: bool = False,
        predict_nani: bool = False,
        normalize_mode: str = "NFKC",
        use_read_as_pron: bool = True,
        revert_long_vowels: bool = True,
        revert_yotsugana: bool = True,
    ) -> list[dict[str, object]]:
        recorded.update(
            {
                "text": text,
                "run_marine": run_marine,
                "use_vanilla": use_vanilla,
                "use_sudachi_kanji_yomi": use_sudachi_kanji_yomi,
                "predict_nani": predict_nani,
                "normalize_mode": normalize_mode,
                "use_read_as_pron": use_read_as_pron,
                "revert_long_vowels": revert_long_vowels,
                "revert_yotsugana": revert_yotsugana,
            }
        )
        return [_fake_punctuation_feature()]

    fake_pyopenjtalk.run_frontend = run_frontend  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyopenjtalk", fake_pyopenjtalk)

    features = adapter.run_frontend("えっ！")

    assert recorded == {
        "text": "えっ！",
        "run_marine": False,
        "use_vanilla": False,
        "use_sudachi_kanji_yomi": True,
        "predict_nani": True,
        "normalize_mode": "None",
        "use_read_as_pron": False,
        "revert_long_vowels": False,
        "revert_yotsugana": False,
    }
    assert features[0]["read"] == "、"
    assert features[0]["pron"] == "、"


def test_run_frontend_supports_legacy_pyopenjtalk_signature(monkeypatch):
    from style_bert_vits2.nlp.japanese.pyopenjtalk_worker import adapter

    recorded: dict[str, str] = {}
    fake_pyopenjtalk = ModuleType("pyopenjtalk")

    def run_frontend(text: str) -> list[dict[str, object]]:
        recorded["text"] = text
        return [_fake_punctuation_feature()]

    fake_pyopenjtalk.run_frontend = run_frontend  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyopenjtalk", fake_pyopenjtalk)

    features = adapter.run_frontend("えっ！")

    assert recorded == {"text": "えっ！"}
    assert features[0]["read"] == "、"
    assert features[0]["pron"] == "、"


def test_worker_server_dispatches_run_frontend_through_adapter():
    from style_bert_vits2.nlp.japanese.pyopenjtalk_worker import adapter
    from style_bert_vits2.nlp.japanese.pyopenjtalk_worker import worker_server

    assert worker_server.PYOPENJTALK_FUNC_DICT["run_frontend"] is adapter.run_frontend
