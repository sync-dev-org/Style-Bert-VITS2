from __future__ import annotations

import threading
from pathlib import Path

from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese import user_dict as user_dict_module
from style_bert_vits2.nlp.japanese.user_dict import apply_word, read_dict


def test_concurrent_apply_word_keeps_every_word(tmp_path, monkeypatch):
    """並行 mutation でも user_dict.json の read-modify-write が更新を失わないこと。"""
    monkeypatch.setattr(user_dict_module, "update_dict", lambda *args, **kwargs: None)
    user_dict_path = tmp_path / "user_dict.json"
    compiled_dict_path = tmp_path / "user.dic"

    thread_count = 4
    words_per_thread = 25
    barrier = threading.Barrier(thread_count)
    errors: list[Exception] = []

    def add_words(thread_index: int) -> None:
        barrier.wait()
        try:
            for i in range(words_per_thread):
                apply_word(
                    surface=f"テスト{thread_index}ノ{i}",
                    pronunciation="テスト",
                    accent_type=1,
                    user_dict_path=user_dict_path,
                    compiled_dict_path=compiled_dict_path,
                )
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [
        threading.Thread(target=add_words, args=(n,)) for n in range(thread_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads), "mutation がタイムアウトした"

    assert errors == []
    words = read_dict(user_dict_path=user_dict_path)
    assert len(words) == thread_count * words_per_thread


def test_apply_word_completes_with_real_update_dict(tmp_path, monkeypatch):
    """mutation が lock 保持中に内部の update_dict を呼んでも deadlock しないこと。"""
    monkeypatch.setattr(
        pyopenjtalk,
        "mecab_dict_index",
        lambda csv_path, out_path: Path(out_path).write_bytes(b"compiled"),
    )
    monkeypatch.setattr(pyopenjtalk, "unset_user_dict", lambda: None)
    monkeypatch.setattr(
        pyopenjtalk, "update_global_jtalk_with_user_dict", lambda path: None
    )
    user_dict_path = tmp_path / "user_dict.json"
    compiled_dict_path = tmp_path / "user.dic"
    result: dict[str, str] = {}

    def add_word() -> None:
        result["uuid"] = apply_word(
            surface="テスト",
            pronunciation="テスト",
            accent_type=1,
            user_dict_path=user_dict_path,
            compiled_dict_path=compiled_dict_path,
        )

    thread = threading.Thread(target=add_word)
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive(), "apply_word が deadlock した"

    assert result["uuid"] in read_dict(user_dict_path=user_dict_path)
    assert compiled_dict_path.read_bytes() == b"compiled"
