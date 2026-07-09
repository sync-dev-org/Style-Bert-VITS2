# feature: テストカバレッジの拡張 (EN / ZH 言語経路と API 経路)

テスト構造の整備 ([test-structure-overhaul.md](test-structure-overhaul.md)) で確立した環境能力検出・skip 設計の上に、言語経路 (EN / ZH の g2p / BERT 特徴抽出)、server_fastapi の API 経路、TTSModelHolder 単体のカバレッジを足す。CI 整備は本 feature の対象外。

## 決定事項

- EN / ZH の g2p 経路をテストする。既知テキストに対する phones / tones / word2ph の期待値一致と、`len(phones) == len(tones) == sum(word2ph)` 等の整合契約を検証する。EN g2p の NLTK data 等の外部資産が取得不能な場合は skip する
- EN / ZH の BERT 特徴抽出 (`extract_bert_feature`) をテストする。検証は shape / dtype 契約 (hidden_size × sum(word2ph)、float32)。BERT weights はローカル資産 → HF repo id の順で解決し、取得不能なら skip する (`tests/test_japanese_bert_feature.py` の既存パターン踏襲)
- server_fastapi.py を app factory (`create_app(model_holder, ...)` 相当) へ機能等価リファクタし、FastAPI アプリを module import だけで構築可能にする。CLI 起動 (`python server_fastapi.py`) の挙動 (引数・設定解決・起動シーケンス) は変えない
- API テストは `fastapi.testclient.TestClient` で行い、モデル資産不要のエンドポイント (`/status`, `/models/info`, `/g2p`) を常時検証、合成 (`/voice`) はモデル資産がある場合のみ検証する。fastapi / httpx が環境に無い場合は importorskip で skip する (依存グラフへの追加はしない)
- TTSModelHolder を単体テストする。`tmp_path` 上に組んだ偽のモデル資産ディレクトリ構造で、モデル列挙 / refresh / モデルファイル絞り込み / 不正入力時の失敗挙動を検証する (実モデル資産に依存しない)
- 新規テストはすべて第一弾の conftest 能力検出・skip 設計に従い、「素の pytest = その環境で failed 0」を維持する

## 変更 scope

1. `tests/test_english_g2p.py` / `tests/test_chinese_g2p.py` 新設 (置き場・分割は実装判断、既存 `test_japanese_g2p_snapshot.py` との対称性を考慮)
2. EN / ZH の `extract_bert_feature` テスト新設 (既存 JP テストとの対称性を考慮)
3. `server_fastapi.py`: app factory 化の機能等価リファクタ (エンドポイント定義を `__main__` ブロックから factory 関数へ移す)
4. `tests/test_server_fastapi_api.py` 新設 (TestClient による API 経路検証)
5. `tests/test_tts_model_holder.py` 新設 (偽資産による単体検証)

## やらないこと

- EN / ZH の TTS 合成 e2e (EN / ZH 対応の通常モデル資産 + BERT weights の両方が必要で資産コスト過大。既存 JP 合成テストが合成経路自体は担保している)
- server_editor.py のリファクタ・テスト (server_fastapi で型を作った後の将来対象)
- CI (GitHub Actions) 整備 (別判断)
- 依存グラフ (pyproject.toml / uv.lock) の変更
- server_fastapi の機能変更・エンドポイント仕様変更

## 受入条件

1. EN g2p: 既知英文に対する phones / tones / word2ph が期待値と一致し、整合契約 (`len(phones) == len(tones) == sum(word2ph)`) を満たす
2. ZH g2p: 既知中文に対して同様の検証が通る
3. EN / ZH の `extract_bert_feature` が shape (hidden_size × sum(word2ph)) / dtype (float32) 契約を満たす (weights 取得不能環境では skip 理由が明示される)
4. `python server_fastapi.py` の CLI 起動挙動が不変 (引数解釈・ポート解決・モデル探索の経路が変わらない)
5. TestClient で `/status` / `/models/info` / `/g2p` がモデル資産なしでも検証され、応答の内容 (status code + ペイロードの契約) が assert される
6. `/voice` がモデル資産のある環境で audio/wav 応答 (非零長) を返すことが検証される (資産なしでは skip)
7. TTSModelHolder が偽資産構造でモデル列挙 / refresh / 絞り込み / 失敗挙動を検証される (実資産不要で常時実行)
8. darwin + モデル資産あり環境と、資産・optional 依存なし環境の双方で、素の `uv run pytest` が failed 0 を維持する
9. 既存テストの検証水準が維持される (第一弾完了時点で pass していたテストが引き続き pass する)
