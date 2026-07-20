# システム仕様 — 全体俯瞰

## 対象範囲

この文書は Style-Bert-VITS2 のコードベース全体を俯瞰し、主要な実行入口、ディレクトリ、設定、データフローと、詳細仕様への索引を示す。

ここでは各系統の接続関係とファイル境界だけを扱う。音声合成、言語処理、各サーバー、WebUI、学習、変換・マージの詳細な引数や内部アルゴリズムは、後掲の兄弟仕様を参照する。

## アーキテクチャ

システムは次の層から構成される。

1. ルートの Python スクリプトと `.bat` が、セットアップ、WebUI、サーバー、前処理、学習、変換の実行入口になる。
2. `gradio_tabs/` が、音声合成、データセット作成、学習、スタイル作成、マージ、ONNX 変換の画面と操作を提供する。
3. `src/style_bert_vits2/` が、言語処理、モデル構築、PyTorch / ONNX 推論、モデル資産管理を提供する。
4. `Data/` が学習用データと再開用 checkpoint、`model_assets/` が推論用モデル資産を保持する。
5. `bert/` と `slm/` が、言語特徴および学習補助モデルを保持する。

主要な依存方向は次のとおりである。

```text
.bat / Python entrypoints
        |
        +-- app.py / gradio_tabs/ ---------+
        +-- style_bert_vits2.server --------+--> style_bert_vits2.tts_model
        +-- server_editor.py --------------+             |
        |                                                +--> nlp/ --> bert/
        |                                                +--> models/ --> torch / ONNX Runtime
        |
        +-- dataset / preprocess / train
                    |
inputs/ --> Data/<model>/ --> model_assets/<model>/ --> inference
```

## `src/style_bert_vits2` の層構造

### 公開・統合層

- `tts_model.py`: `TTSModel` による単一モデルの遅延ロードと推論、`TTSModelHolder` によるモデルディレクトリの列挙・選択を担う。
- `constants.py`: 言語 ID、推論既定値、テーマ、辞書ディレクトリなどの共通定数を定義する。
- `voice.py`: 推論後の音声に対する高さ・音量・テンポ調整を担う。
- `logging.py`: 共通 logger を提供する。

### `nlp/` — テキスト・言語特徴層

- `nlp/__init__.py`: テキストの正規化、g2p、音素・tone・言語 ID 列への変換、BERT 特徴抽出を統合する。
- `nlp/japanese/`、`nlp/english/`、`nlp/chinese/`: 言語別の正規化、g2p、BERT 特徴抽出を実装する。
- `nlp/bert_models.py`: PyTorch BERT モデルと tokenizer のロード・解放を管理する。
- `nlp/onnx_bert_models.py`: ONNX BERT セッションと tokenizer のロード・解放を管理する。
- `nlp/japanese/user_dict/` と `pyopenjtalk_worker/`: 日本語ユーザー辞書と pyopenjtalk の別プロセス実行を管理する。

### `models/` — 音声モデル層

- `models.py` と `models_jp_extra.py`: 通常版と JP-Extra 版のモデル構造を定義する。
- `infer.py` と `infer_onnx.py`: PyTorch と ONNX Runtime の推論処理を実装する。
- `hyper_parameters.py`: `config.json` を Pydantic モデルとして読み込む。
- `training.py`: 学習 device、AMP、DataLoader worker の共通処理を提供する。
- `attentions.py`、`modules.py`、`transforms.py`、`commons.py`、`monotonic_alignment.py`: モデル内部の構成要素を提供する。
- `models/utils/`: checkpoint、safetensors、weight normalization の読み書き・互換処理を提供する。

### `utils/` — 実行補助層

- device 名から ONNX Runtime の ExecutionProvider 設定への変換
- 子 Python スクリプトの実行とログ収集
- stdout の安全なラップと文字列 Enum

## リポジトリ内の主要ディレクトリ

| パス | 役割 |
| --- | --- |
| `gradio_tabs/` | `app.py` に組み込む各 Gradio タブと、単独タブとして起動できる画面 |
| `configs/` | パス設定の既定値と、通常版 / JP-Extra 版のモデル別学習設定テンプレート |
| `model_assets/` | 推論時に列挙されるモデル単位の資産ディレクトリ |
| `bert/` | 日本語・中国語・英語の PyTorch / ONNX BERT tokenizer、設定、重み |
| `slm/wavlm-base-plus/` | JP-Extra 学習で参照する WavLM の設定と重み |
| `dict_data/` | 日本語ユーザー辞書の CSV。起動時や前処理時に pyopenjtalk へ反映される |
| `inputs/` | `slice.py` とデータセット作成 UI が既定で読む、分割前音声の置き場 |
| `Data/` | モデル別の raw 音声、前処理結果、学習設定、再開用 checkpoint の既定ルート |
| `src/style_bert_vits2/` | install されるコア Python package |
| `tests/` | 推論、NLP、サーバー設定、ONNX 変換、学習 device・互換性の自動テスト |

ルート直下の Python ファイルは実行入口と学習パイプラインの調整層である。`app.py` と `server_editor.py` は利用者向けサービスを起動し、OpenAI 互換 API サーバーは install される `style_bert_vits2.server` package が提供する。`slice.py` から `train_ms*.py` までのスクリプトがデータ準備・学習を行う。`convert_onnx.py`、`default_style.py`、`style_gen.py` などはモデル資産を生成・変換する。`data_utils.py`、`mel_processing.py`、`losses.py` は学習処理から利用される。

## Windows エントリポイント

ルートの `.bat` はリポジトリルートへ移動し、原則として `venv\Scripts\python` で対応する Python 入口を起動する。失敗時は終了コードを返し、画面を閉じる前に停止する。

| `.bat` | Python エントリ | 役割 |
| --- | --- | --- |
| `App.bat` | `app.py` | 全 Gradio タブをまとめた WebUI |
| `Editor.bat` | `server_editor.py --inbrowser` | 音声合成エディター |
| `Train.bat` | `python -m gradio_tabs.train` | 学習タブ単独 UI |
| `Dataset.bat` | `python -m gradio_tabs.dataset` | データセット作成タブ単独 UI |
| `Merge.bat` | `python -m gradio_tabs.merge` | モデルマージタブ単独 UI |
| `StyleVectors.bat` | `python -m gradio_tabs.style_vectors` | スタイルベクトル作成タブ単独 UI |
| `Inference.bat` | `python -m gradio_tabs.inference` | 音声合成タブ単独 UI |
| `ConvertONNX.bat` | `python -m gradio_tabs.convert_onnx` | ONNX 変換タブ単独 UI |
| `Initialize.bat` | `initialize.py` | 必要モデルのダウンロードとパス設定初期化 |

`app.py` は上記の音声合成、データセット作成、学習、スタイル作成、マージ、ONNX 変換の 6 タブを単一の Gradio アプリに組み込む。

## セットアップ経路

### `initialize.py`

`initialize.py` は存在しないファイルだけを Hugging Face Hub から取得し、次の配置を作る。

- `bert/bert_models.json` に列挙された 3 言語の PyTorch BERT 重みと ONNX BERT モデルを、各 `bert/<model>/` に配置する。
- `--skip_default_models` がなければ、4 つの JVNV モデルと `koharune-ami`、`amitaro` を `model_assets/<model>/` に配置する。各モデルは `config.json`、`style_vectors.npy`、推論モデルを含む。
- `--only_infer` がなければ、`slm/wavlm-base-plus/pytorch_model.bin`、通常版の `pretrained/`、JP-Extra 版の `pretrained_jp_extra/` を配置する。
- `configs/paths.yml` がなければ `configs/default_paths.yml` をコピーする。
- `--dataset_root` または `--assets_root` が指定された場合は、生成・既存の `configs/paths.yml` の対応値を更新する。

`--only_infer` は学習用の WavLM と事前学習 checkpoint を省略するが、BERT モデルと、明示的に除外されていない既定音声モデルは取得対象のままである。

### `scripts/` の Windows セットアップ

- `Setup-Python.bat`: Python 3.10.11 embeddable package を取得して `site` を有効化し、pip と virtualenv を導入して仮想環境を作る。
- `Install-Style-Bert-VITS2.bat`: Git がなければ PortableGit を用意し、コードを clone する。Python 環境、uv、CUDA 12.9 用 PyTorch、WebUI を含む project 依存を導入し、`initialize.py` の後に editor を起動する。
- `Install-Style-Bert-VITS2-CPU.bat`: 同様に環境を構築するが、PyTorch の CUDA index を使う明示インストールを行わず、`initialize.py --only_infer` の後に editor を起動する。
- `Update-Style-Bert-VITS2.bat`: `git pull` 後、既存仮想環境で uv と project 依存を更新する。

2 つの install script の clone 先は、script 内の `REPO_URL` に固定されている。

## 設定の全体像

設定は、全体パス・実行調整用 YAML と、モデル別 JSON に分かれる。

### パス設定

`configs/default_paths.yml` は次の既定値を持つ。

- `dataset_root: Data`
- `assets_root: model_assets`

実行時に使うのは `configs/paths.yml` である。存在しない場合、`initialize.py` または `config.get_path_config()` が既定ファイルから生成する。

### 実行調整用 `config.yml`

`default_config.yml` は、対象 `model_name`、resample、テキスト前処理、BERT / style 特徴生成、学習、現在サポート対象外の WebUI 設定を持つ。実行時に使うルートの `config.yml` がなければ、`config.Config` または学習 UI の初期化処理が `default_config.yml` から生成する。

`config.py` は `configs/paths.yml` と `config.yml` を読み、相対パスを対象 dataset directory に結合した設定オブジェクトを作る。CUDA が利用できない場合、BERT、style、WebUI の device は CPU に補正される。`config.yml` が必須キー不足で読めない場合は、既定ファイルで置き換えて再読込する。

### モデル別 `config.json`

`configs/config.json` と `configs/config_jp_extra.json` は、通常版 / JP-Extra 版の学習設定テンプレートである。前処理の初期化時に選択したテンプレートを `Data/<model_name>/config.json` へ展開し、モデル名、train / validation list、batch size、epoch、保存間隔、freeze 設定などを反映する。

学習開始時、同じモデル別 JSON がモデル構造とデータ条件の入力になる。推論用には style 情報を反映した `config.json` が `model_assets/<model_name>/` に保存される。

## データフロー

### 1. データセット作成

```text
inputs/ または任意の音声ディレクトリ
  --> slice.py
  --> Data/<model_name>/raw/**/*.wav
  --> transcribe.py
  --> Data/<model_name>/esd.list
```

`slice.py` は音声を VAD で分割し、設定された `dataset_root` 以下へ保存する。すでに適切な長さの音声がある場合は `raw/` へ直接配置できる。`transcribe.py` は `raw/` 以下の WAV を文字起こしし、既存の `esd.list` があれば `.bak` へ退避する。

### 2. 前処理

`preprocess_all.py` は `gradio_tabs.train.preprocess_all()` を CLI から呼び、次の順序で処理する。

1. 通常版または JP-Extra 版の設定テンプレートを展開し、対応する `pretrained*/` を `Data/<model_name>/models/` へコピーする。
2. `resample.py` が `raw/` を 44.1 kHz の `wavs/` へ変換する。指定により loudness normalize と無音 trim を行う。
3. `preprocess_text.py` が `esd.list` を正規化・g2p 処理し、cleaned list、`train.list`、`val.list` と更新済み `config.json` を作る。
4. `bert_gen.py` が各音声に対応する `*.bert.pt` を作る。
5. `style_gen.py` が各音声に対応する `*.wav.npy` の style 特徴を作る。

各段階は失敗すると後続処理へ進まない。日本語処理を使う入口では、`dict_data/` のユーザー辞書が pyopenjtalk に反映される。

### 3. 学習と資産出力

通常版は `train_ms.py`、JP-Extra 版は `train_ms_jp_extra.py` が学習する。

- 再開用の generator / discriminator checkpoint と TensorBoard log は `Data/<model_name>/models/` に保存する。
- style 特徴の平均またはサブディレクトリ別平均から `model_assets/<model_name>/style_vectors.npy` を作る。
- style 名と ID を含む推論用 `config.json` を `model_assets/<model_name>/` に作る。
- 保存時点の generator を推論用 `.safetensors` として `model_assets/<model_name>/` に出力する。

### 4. 推論

`TTSModelHolder` は `assets_root` 直下の非 hidden directory をモデル名として列挙し、`.pth`、`.pt`、`.safetensors`、必要に応じて `.onnx` をモデル候補とする。モデル directory に `config.json` がない場合、候補から除外する。

標準的なモデル資産は次の形式である。

```text
model_assets/
  <model_name>/
    config.json
    style_vectors.npy
    <任意のモデルファイル名>.safetensors
    <任意のモデルファイル名>.onnx    # 変換済みの場合
```

`TTSModel` はモデルファイルの拡張子が `.onnx` なら ONNX Runtime、それ以外なら PyTorch のロード経路を選ぶ。両経路とも同じ directory の `config.json` と `style_vectors.npy` を使い、テキスト処理、style 選択、話者選択を経て波形を返す。

`convert_onnx.py` は `.safetensors` と同じ directory に同名 stem の `.onnx` を生成する。変換時にも隣接する `config.json` と `style_vectors.npy` が必要である。

## 制約と境界

- `Data/` は学習を継続・再開するための作業資産、`model_assets/` は推論に必要な配布・利用資産であり、保存目的が異なる。
- `config.yml` とモデル別 `config.json` は別の設定である。前者はパイプライン実行とパス解決、後者はモデル構造・学習データ・推論メタデータを担う。
- `model_assets/<model_name>/` ではモデルファイル名を固定しない。同じ directory に複数世代や PyTorch / ONNX 形式を置ける。
- BERT と音声合成モデルは別の資産として配置される。`TTSModel` の PyTorch 推論経路は PyTorch BERT、ONNX 推論経路は ONNX BERT を使用する。
- `app.py`、OpenAI 互換 API サーバー、単独タブは同じ package と資産形式を共有するが、公開する操作・API と ONNX の列挙可否は各入口の設定に従う。

## 詳細仕様の索引

- [core-inference.md](core-inference.md): `TTSModel` / `TTSModelHolder`、モデルロード、PyTorch / ONNX 音声合成の仕様。
- [nlp.md](nlp.md): 正規化、g2p、BERT 特徴抽出、日本語辞書と pyopenjtalk worker の仕様。
- [openai-api-server.md](openai-api-server.md): package module で起動する OpenAI 互換音声合成 API、buffered WAV、文単位 PCM streaming の仕様。
- [editor-server.md](editor-server.md): `server_editor.py` の editor server、文書・音声生成操作の仕様。
- [webui.md](webui.md): `app.py` と `gradio_tabs/` の画面構成、単独タブ、操作フローの仕様。
- [training-pipeline.md](training-pipeline.md): データセット作成、前処理、特徴生成、通常版 / JP-Extra 学習の仕様。
- [conversion-merge.md](conversion-merge.md): ONNX 変換、モデルマージ、style vector 作成・更新の仕様。

## 関連文書

- [CLI.md](../CLI.md): セットアップ、データ作成、前処理、学習のコマンド利用方法。
- [REQUIREMENTS.md](../REQUIREMENTS.md): fork の確定要件、互換方針、対応環境、スコープ境界。
- [ROADMAP.md](../ROADMAP.md): 近代化計画の現在地。

## 関連テスト

全体境界に直接関係する主なテストは次のとおりである。各系統の詳細テストは兄弟仕様に記載する。

- `tests/test_tts_model_holder.py`: `model_assets/` の列挙、refresh、ONNX 除外、モデル選択。
- `tests/test_main.py`: 公開 package を通した音声合成の統合挙動。
- `tests/test_server_openai_api.py`: OpenAI 互換 endpoint、WAV / PCM、streaming header、validation。
- `tests/test_onnx_export_restoration.py`: 音声・BERT ONNX export 経路と変換 UI の整合性。
- `tests/test_training_device.py`: 学習 device と CPU / CUDA 切替。
- `tests/test_train_torch_modernization.py`: 学習処理と既存 checkpoint の互換性。
