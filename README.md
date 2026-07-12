# Style-Bert-VITS2

> [!IMPORTANT]
> 利用する前に、必ず[お願いとデフォルトモデルの利用規約][terms]を確認してください。

Bert-VITS2 を基に、感情や発話スタイルの強さを連続的に制御できるようにした音声合成・学習プロジェクトです。音声合成エディター、Gradio WebUI、FastAPI サーバー、学習・モデル変換ツール、Python ライブラリを提供します。

このリポジトリは [litagin02/Style-Bert-VITS2][upstream] の独立 fork です。上流への pull request や再合流を前提とせず、[sync-dev-org/Style-Bert-VITS2][fork] として独立してメンテナンスしています。元になった [fishaudio/Bert-VITS2][bert-vits2] と上流プロジェクトの作者・貢献者に感謝します。

## この fork の主な変更点

- **パッケージングと環境構築の刷新**: 依存定義を `pyproject.toml` に集約し、uv と `uv.lock` による再現可能な環境構築へ移行しました。ビルドバックエンドは hatchling、バージョンは hatch-vcs により `vX.Y.Z` tag から導出します。パッケージは `src/style_bert_vits2/` の src layout です。
- **現行 Python / PyTorch への対応**: Python 3.10–3.12 を対象とし、PyTorch を optional dependency として整理しました。CUDA 12.9 用 PyTorch index を統一し、ONNX Runtime family は CUDA 12 runtime と整合する 1.24 系に固定しています。
- **日本語処理の改善**: 記号、数値、電話番号、郵便番号、住所などの正規化と g2p を拡充しました。`pyopenjtalk` worker は起動ディレクトリに依存せず、socket request とユーザー辞書アクセスを直列化して並行実行時の競合を防ぎます。
- **学習パイプラインの安定化**: 現行 PyTorch API への移行、rank 0 のみの checkpoint 保存、WSL2 での `gloo` backend 選択、再開時の learning rate fallback、前処理失敗の伝播などを追加しました。
- **推論・変換の堅牢化**: 空に近い入力、無音波形、行分割時の sampling rate、複数話者指定などの境界条件を修正しました。ONNX export を現行 PyTorch で利用できる状態へ戻し、AIVM / AIVMX 連携は削除しています。
- **品質基盤の整備**: リポジトリ全体へ Ruff の lint / format / import sorting を適用し、言語処理、推論、API、学習、ONNX export、worker 並行性などの test を拡充しました。コードから逆算した 8 系統の[挙動仕様][spec]も整備しています。
- **Google Colab support の終了**: Colab notebook は削除済みで、今後の動作保証対象にも含めません。

確定済みの対応範囲は[要件正典][requirements]、開発の現在地は[ロードマップ][roadmap]を参照してください。

## 主な機能

- 日本語・英語・中国語の音声合成
- スタイルベクトルによる発話スタイルと強度の制御
- 音声合成エディターと Gradio WebUI
- 音声 dataset の作成、文字起こし、前処理、学習
- 学習済みモデルと style vector の merge
- safetensors model の ONNX 変換
- FastAPI 音声合成 server
- Python API による推論

既存の `model_assets/` 形式にある学習済み model の読み込み互換を維持しています。

## 動作環境

- Python 3.10 / 3.11 / 3.12
- Windows、WSL2、Linux
- 音声合成は CPU でも実行可能
- 学習には NVIDIA GPU が必要

依存管理と repository checkout の install 経路は uv を正とします。Windows / Linux の CUDA 対応 package は PyTorch の `cu129` index から解決されます。

## インストール

### Repository checkout

Git、Python 3.10 以上、[uv][uv]を用意し、repository を clone します。

```bash
git clone https://github.com/sync-dev-org/Style-Bert-VITS2.git
cd Style-Bert-VITS2
uv sync --no-dev --extra torch --group webui
uv run python initialize.py
```

`initialize.py` は必要な BERT model とデフォルト TTS model を取得し、path 設定を初期化します。デフォルト model が不要な場合は `--skip_default_models` を指定できます。

```bash
uv run python initialize.py --skip_default_models
```

上流 release の `sbv2.zip` と `Install-Style-Bert-VITS2*.bat` は、この fork の install 経路ではありません。

### Python library

PyPI 公開後は、推論用 library を次の配布名で install できます。

```bash
pip install style-bert-vits2-mk
```

配布名は `style-bert-vits2-mk` ですが、Python の import 名は従来どおり `style_bert_vits2` です。

具体的な使用例は [library.ipynb](library.ipynb) を参照してください。

```python
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel
```

## 起動方法

Repository checkout を初期化した後、目的に応じて次の command を実行します。

| 用途 | コマンド |
|---|---|
| 音声合成 editor | `uv run python server_editor.py --inbrowser` |
| Gradio WebUI | `uv run python app.py` |
| FastAPI server | `uv run python server_fastapi.py` |

FastAPI の endpoint 仕様は server 起動後の `/docs` で確認できます。CLI で dataset 作成、前処理、学習を行う場合は [CLI guide][cli]を参照し、環境構築には本 README の手順を使用してください。

WebUI には音声合成、dataset 作成、学習、style 作成、merge、ONNX 変換の各画面があります。

## モデル資産

推論には、model ごとに次の 3 種類の file が必要です。

```text
model_assets/
└── <model-name>/
    ├── config.json
    ├── <model-file>.safetensors
    └── style_vectors.npy
```

- `config.json`: model 構成
- `*.safetensors`: 学習済み weight
- `style_vectors.npy`: 利用可能な style と style vector

model を共有する場合は、この 3 種類を同じ directory 構造で共有してください。

## ドキュメント

- [お願いとデフォルトモデルの利用規約][terms]
- [よくある質問][faq]
- [CLI guide][cli]
- [要件正典][requirements]
- [ロードマップ][roadmap]
- [挙動仕様][spec]
- [更新履歴][changelog]

## 上流プロジェクトについて

上流の tutorial 動画、Hugging Face Space、release archive、version 履歴は上流版を対象とした情報です。この fork では動作や install 手順が異なるため、必要な場合は[上流 repository][upstream]の資料として参照してください。この fork の利用方法と対応範囲は本 README と `docs/` を正とします。

## 系譜と謝辞

この repository の系譜は次のとおりです。

1. [fishaudio/Bert-VITS2][bert-vits2]
2. [litagin02/Style-Bert-VITS2][upstream]
3. [sync-dev-org/Style-Bert-VITS2][fork]（本 fork）

Style-Bert-VITS2 は Bert-VITS2 v2.1 / Japanese-Extra を基に、style vector、各種 WebUI、safetensors 対応などを加えた project です。上流では Windows 向け環境構築の多くに [EasyBertVits2][easy-bert-vits2] の成果が使われています。元 project とすべての contributor に感謝します。

## ライセンス

本 repository は、Bert-VITS2 および上流 Style-Bert-VITS2 と同じ [GNU Affero General Public License v3.0][license] で公開しています。

`src/style_bert_vits2/nlp/japanese/user_dict/` は [GNU Lesser General Public License v3.0][lgpl-license] の対象です。由来と詳細は同 directory 内の README を参照してください。

[terms]: https://github.com/sync-dev-org/Style-Bert-VITS2/blob/master/docs/TERMS_OF_USE.md
[upstream]: https://github.com/litagin02/Style-Bert-VITS2
[fork]: https://github.com/sync-dev-org/Style-Bert-VITS2
[bert-vits2]: https://github.com/fishaudio/Bert-VITS2
[requirements]: https://github.com/sync-dev-org/Style-Bert-VITS2/blob/master/docs/REQUIREMENTS.md
[roadmap]: https://github.com/sync-dev-org/Style-Bert-VITS2/blob/master/docs/ROADMAP.md
[spec]: https://github.com/sync-dev-org/Style-Bert-VITS2/blob/master/docs/spec/README.md
[uv]: https://docs.astral.sh/uv/
[cli]: https://github.com/sync-dev-org/Style-Bert-VITS2/blob/master/docs/CLI.md
[faq]: https://github.com/sync-dev-org/Style-Bert-VITS2/blob/master/docs/FAQ.md
[changelog]: https://github.com/sync-dev-org/Style-Bert-VITS2/blob/master/docs/CHANGELOG.md
[easy-bert-vits2]: https://github.com/Zuntan03/EasyBertVits2
[license]: https://github.com/sync-dev-org/Style-Bert-VITS2/blob/master/LICENSE
[lgpl-license]: https://github.com/sync-dev-org/Style-Bert-VITS2/blob/master/LGPL_LICENSE
