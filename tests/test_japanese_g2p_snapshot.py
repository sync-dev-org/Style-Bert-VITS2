import atexit
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from style_bert_vits2.constants import DEFAULT_BERT_MODEL_PATHS, Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.g2p import g2p, text_to_sep_kata
from style_bert_vits2.nlp.japanese.normalizer import normalize_text
from style_bert_vits2.nlp.japanese.user_dict import update_dict


SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "japanese_g2p_snapshot.json"
UPDATE_SNAPSHOT_ENV = "UPDATE_JAPANESE_G2P_SNAPSHOT"
JP_BERT_MODEL_ID = "ku-nlp/deberta-v2-large-japanese-char-wwm"

CASES = [
    {"id": "simple_weather", "text": "今日はいい天気です。"},
    {"id": "meeting_time", "text": "明日の会議は午後三時です。"},
    {"id": "particles_wa_e", "text": "私は駅へ向かいます。"},
    {"id": "object_location", "text": "本は机の上にあります。"},
    {"id": "date_kanji", "text": "八月三十一日に図書館へ行きます。"},
    {"id": "date_time_digits", "text": "2026年7月7日、13時45分に開始します。"},
    {"id": "price_digits", "text": "価格は1,200円です。"},
    {"id": "english_api", "text": "APIの仕様を確認します。"},
    {"id": "english_usb", "text": "USBケーブルを用意してください。"},
    {"id": "katakana_server", "text": "サーバーの状態を確認します。"},
    {"id": "katakana_coffee", "text": "コーヒーを一杯ください。"},
    {"id": "shopping", "text": "スーパーへ買い物に行きます。"},
    {"id": "question", "text": "これはテストではありませんか？"},
    {"id": "exclamation", "text": "えっ、本当にそうなんですか！"},
    {"id": "katakana_campaign", "text": "キャンペーンのページを開きます。"},
    {"id": "long_vowel_mail", "text": "メールを送信しました。"},
    {"id": "direction", "text": "地図を見ながら右へ曲がります。"},
    {"id": "folder", "text": "新しいフォルダーを作成します。"},
    {"id": "speech_rate", "text": "読み上げ速度を少し下げます。"},
    {"id": "colors", "text": "青い空と白い雲が見えます。"},
    {"id": "station", "text": "市内の駅で待ち合わせます。"},
    {"id": "weight", "text": "三百五十グラムの材料を混ぜます。"},
    {"id": "letters", "text": "AとBを比較します。"},
    {"id": "version_number", "text": "バージョン2.5を確認します。"},
    {"id": "prolonged_and_dash", "text": "長音記号ーとダッシュ―を区別します。"},
    {"id": "small_kana", "text": "小さな「ゃゅょ」を含む文章です。"},
    {"id": "entrance_exit", "text": "入口はこちらです。出口はあちらです。"},
    {"id": "file_name", "text": "ファイル名はsample.txtです。"},
    {"id": "morning_time", "text": "朝七時半に起きます。"},
    {"id": "rain", "text": "雨が降ったので、傘を持っていきます。"},
]


@pytest.fixture(scope="module", autouse=True)
def _japanese_g2p_environment():
    pyopenjtalk.initialize_worker()
    try:
        update_dict()
        tokenizer_path = DEFAULT_BERT_MODEL_PATHS[Languages.JP]
        tokenizer_source = (
            str(tokenizer_path) if tokenizer_path.exists() else JP_BERT_MODEL_ID
        )
        bert_models.load_tokenizer(Languages.JP, tokenizer_source)
        yield
    finally:
        pyopenjtalk.terminate_worker()
        bert_models.unload_tokenizer(Languages.JP)
        try:
            atexit.unregister(pyopenjtalk.terminate_worker)
        except ValueError:
            pass


def _build_snapshot() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "metadata": {
            "pyopenjtalk_dict": version("pyopenjtalk-dict"),
            "tokenizer": JP_BERT_MODEL_ID,
        },
        "cases": [_build_case_snapshot(case) for case in CASES],
    }


def _build_case_snapshot(case: dict[str, str]) -> dict[str, Any]:
    normalized_text = normalize_text(case["text"])
    sep_text, sep_kata = text_to_sep_kata(normalized_text)
    phones, tones, word2ph = g2p(normalized_text)

    return {
        "id": case["id"],
        "text": case["text"],
        "normalized_text": normalized_text,
        "sep_text": sep_text,
        "sep_kata": sep_kata,
        "phones": phones,
        "tones": tones,
        "word2ph": word2ph,
    }


def _write_snapshot(snapshot: dict[str, Any]) -> None:
    SNAPSHOT_PATH.parent.mkdir(exist_ok=True)
    SNAPSHOT_PATH.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_japanese_g2p_snapshot():
    actual = _build_snapshot()

    if os.environ.get(UPDATE_SNAPSHOT_ENV) == "1":
        _write_snapshot(actual)

    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert actual == expected
