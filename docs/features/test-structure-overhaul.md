# feature: テスト構造の整備 (環境依存の構造化と検出力強化)

tests/ 配下のテスト盤面を「素の `pytest` 実行 = その環境で意味のある全テストが green」に成立させ、合成テストの検証力と言語カバレッジの空白 (EN / ZH tokenizer 経路) を埋める。カバレッジ拡張の第二弾 (EN / ZH 合成 e2e、API 経路、TTSModelHolder 単体) と CI 整備は本 feature の対象外。

## 決定事項

- 環境依存テスト (GPU / EP 依存) は実行環境の能力検出 (onnxruntime available providers / `torch.cuda.is_available()` / platform) による skipif で自動 skip する。skip 理由に不足能力を明示する
- CoreML EP の JP-Extra 合成は EP 能力制約 (SDP flow 内 GatherND の zero-element 動的 shape 非対応) による既知の実行時失敗があるため、xfail (strict=False、理由明記) として扱う。制約が解消されれば xpass で観測できる
- 合成テストの assertion は音声の振る舞い (非零長 / 非無音 / sample rate) を検証する。ファイル存在確認のみの assertion は廃する
- 合成テストは推論形式 × EP × スタイルで parametrize し、失敗箇所が個別に報告される構造にする
- テスト出力は pytest の `tmp_path` 系 fixture 配下に書き、repo 内 (`tests/wavs/`) への書き込みを廃止する
- EN / ZH の BERT tokenizer ロード経路をテストする。BERT モデル本体 (weights) は不要で、tokenizer 資産のみで検証する。ローカル資産不在時は HF repo id へ fallback する (`test_japanese_g2p_snapshot.py` の既存パターンに合わせる)。ネットワーク・資産とも不能な場合は skip する
- pytest 設定 (marker 定義 / testpaths) は `pyproject.toml` の `[tool.pytest.ini_options]` に置く

## 変更 scope

1. `tests/conftest.py` 新設: 環境能力検出 (available providers / CUDA / platform) の helper と共通 fixture
2. `pyproject.toml`: `[tool.pytest.ini_options]` 節の追加 (marker 定義 / testpaths)
3. `tests/test_main.py` 改修: parametrize 化 / 能力検出 skipif / CoreML xfail / 音声 assertion 強化 / `tmp_path` 出力化
4. EN / ZH tokenizer ロードテスト新設 (置き場は実装判断: 既存 file への追加または新設 file)
5. `tests/wavs/` と `tests/.gitignore` の廃止 (出力の `tmp_path` 化に伴い不要化)

## やらないこと

- EN / ZH の合成 e2e テスト、g2p テスト (第二弾)
- server_fastapi / server_editor の API 経路テスト (第二弾)
- TTSModelHolder 単体テスト (第二弾)
- CI (GitHub Actions) 整備 (別判断)
- 既存テストの検証内容の削減・緩和

## 受入条件

1. CUDA 非搭載の darwin 環境で素の `uv run pytest` が failed 0 で完走する (環境非対応テストは skip、CoreML の既知制約は xfail として報告される)
2. GPU / EP 依存テストが対応 marker を持ち、skip 理由から不足している能力 (EP 名 / CUDA / platform) が読み取れる
3. 合成テストが、生成音声の非零長 / 非無音 (振幅の最大絶対値がしきい値超) / sample rate のモデル設定一致を assert する
4. 合成テストが推論形式 × EP × スタイルで個別の test item として報告される
5. テスト実行が repo 内にファイルを書き込まない (`git status` が実行前後で不変)
6. EN tokenizer が fork の依存グラフのみで構築でき (spm.model からの sentencepiece 経路)、既知テキストのトークン化結果が期待値と一致する
7. ZH tokenizer についても同様にロードとトークン化結果を検証する
8. 既存テストの検証水準が維持される (darwin + モデル資産ありの環境で従来 pass していたテストが引き続き pass する)
