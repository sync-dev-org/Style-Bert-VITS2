# Gradio WebUI 仕様

## 対象範囲

本書は `app.py` が提供する Gradio WebUI と、次のタブ実装について、利用者の操作から呼び出される処理および画面上の結果までを定義する。

- `gradio_tabs/inference.py`: 音声合成
- `gradio_tabs/dataset.py`: データセット作成
- `gradio_tabs/train.py`: 前処理、学習、Tensorboard
- `gradio_tabs/style_vectors.py`: スタイルベクトル作成
- `gradio_tabs/merge.py`: モデルおよびスタイルのマージ
- `gradio_tabs/convert_onnx.py`: ONNX 変換

音声合成、学習、変換・マージの下位処理は、それぞれ [core-inference.md](core-inference.md)、[training-pipeline.md](training-pipeline.md)、[conversion-merge.md](conversion-merge.md) が扱う。本書では下位アルゴリズムを再定義せず、UI の入力、呼び先、出力、エラー表示を扱う。

## 起動と共有状態

### 起動前処理

`app.py` は UI 構築前に `pyopenjtalk_worker.initialize_worker()` を呼び、`dict_data/` の辞書を `update_dict()` で適用する。

起動引数は次のとおりである。

| 引数 | 既定値 | 挙動 |
| --- | --- | --- |
| `--device` | `cuda` | モデル実行デバイス。`cuda` 指定時に CUDA が利用できなければ `cpu` に切り替える |
| `--host` | `127.0.0.1` | `server_name` として Gradio に渡す |
| `--port` | 未指定 | `server_port` として Gradio に渡す |
| `--no_autolaunch` | false | 指定時はブラウザーを自動で開かない |
| `--share` | false | Gradio の共有 URL を有効にする |

`get_path_config()` の `assets_root`、選択済みデバイス、デバイスに対応する ONNX Runtime provider から `TTSModelHolder` を作る。`ignore_onnx=True` のため、各モデルファイル選択肢は ONNX モデルを除外する。

### タブ構成

画面タイトルはバージョンを含む `Style-Bert-VITS2 WebUI` であり、タブは次の順に並ぶ。

1. 音声合成
2. データセット作成
3. 学習
4. スタイル作成
5. マージ
6. ONNX変換

音声合成タブと ONNX 変換タブは、起動時に作成した同じ `TTSModelHolder` を受け取る。マージタブは、その holder のルート、デバイス、provider から ONNX を無視する別の holder を作る。各タブの「更新」は、対応する holder にモデル一覧を再走査させる。

## 音声合成タブ

### 初期状態とモデル選択

`assets_root` に利用可能なモデルがない場合、配置先を示す `Error: モデルが見つかりませんでした。...` の Markdown のみを表示する。

モデルがある場合は先頭のモデルと先頭のモデルファイルを選択する。「モデル一覧」を変更すると `TTSModelHolder.update_model_files_for_gradio()` が「モデルファイル」の選択肢を更新する。「モデルファイル」の変更または「更新」は音声合成ボタンを無効にし、「ロード」を再度必要とする。

「ロード」は `TTSModelHolder.get_model_for_gradio()` を呼び、次を更新する。

- 選択モデルのスタイル一覧
- 話者一覧
- 有効化された「音声合成」ボタン

モデルをロードするまで音声合成ボタンは操作できない。BERT モデルのロードも「ロード」操作まで行われない。

### 合成入力

主な入力は次のとおりである。

- 読み上げテキストと言語 (`JP`、`EN`、`ZH`)
- 話者、音高、抑揚
- 改行単位の分割と、分割間に挟む無音秒数
- `SDP Ratio`、`Noise`、`Noise_W`、`Length`
- Assist text、Assist text の重み、利用有無
- スタイル指定方法、スタイル、スタイルの強さ
- 日本語アクセント JSON と利用有無
- ヌルモデル群と、声質・声の高さ・話し方・テンポごとの重み

スタイル指定方法が「プリセットから選ぶ」の場合はスタイル dropdown を表示し、「音声ファイルを入力」の場合は filepath 型の参照音声を表示する。

「改行で分けて生成」の変更は無音秒数 slider の表示を切り替える。「アクセント調整を使う」を有効にすると「改行で分けて生成」を false にする。Assist text の checkbox は、テキスト欄と重み slider を同時に表示または非表示にする。

### ヌルモデル

「ヌルモデルの数」に応じ、モデル名、モデルファイル、4 種の重みを持つ行を動的に描画する。「増やす」は件数を 1 増やし、「減らす」は 0 を下限として 1 減らす。

各行のモデルファイルまたは重みを変更すると、`NullModelParam` を `gr.State` の辞書へ保存し、次回合成でモデルを強制再ロードする state を true にする。ヌルモデルのモデルファイル変更中は合成ボタンを無効にする。

### 合成処理と結果

「音声合成」は次の順で処理する。

1. `TTSModelHolder.get_model()` で選択モデルを取得する。
2. 日本語アクセント指定があれば JSON を `(カナ, 0|1)` の列として解析し、音素単位の tone に変換する。
3. 選択話者名を `spk2id` で話者 ID に変換する。
4. 全入力、ヌルモデル群、強制再ロード state を `TTSModel.infer()` に渡す。
5. 情報欄、音声、アクセント欄を更新し、強制再ロード state を false に戻す。

成功時の情報欄は `Success, time: ... seconds.` を表示し、音声欄には `(sample_rate, waveform)` を渡す。日本語で tone を明示しなかった場合は、正規化した入力文からアクセント情報を生成し、アクセント欄へ JSON として返す。日本語以外ではアクセント欄を空にする。

アクセント指定は日本語かつ改行分割なしの場合だけ合成へ渡される。条件を満たさない指定や JSON の解析失敗は、警告文を情報欄へ追加して tone なしで合成を続ける。`InvalidToneError` と `ValueError` の分岐は `Error: ...`、音声なし、元のアクセント文字列を返すが、イベントが要求する 4 出力に対して返値が 3 件しかないため、実際の画面更新は Gradio の出力数検証に依存する。それ以外の例外は関数内で捕捉せず、Gradio のイベントエラーとなる。

## データセット作成タブ

モデル名はスライスと文字起こしで共有する。このタブを使わず、必要な音声と `esd.list` を所定位置へ直接用意することもできる。

### 音声のスライス

入力はモデル名、入力ディレクトリ、最短秒数、最長秒数、無音とみなす最短時間、ファイル名への時間範囲付与である。

「スライスを実行」は `slice.py` を次の引数で `run_script_with_log()` から呼ぶ。

```text
slice.py --model_name <name> --min_sec <value> --max_sec <value>
         --min_silence_dur_ms <value> [--time_suffix] [--input_dir <dir>]
```

モデル名が空なら処理せず、結果欄へ `Error: モデル名を入力してください。` を表示する。下位スクリプト成功時は完了文を、失敗時は `Error: ` と下位メッセージを表示する。生成音声の保存先は `Data/{モデル名}/raw` として案内される。

### 音声の文字起こし

入力はモデル名、Whisper 実装、モデルまたは Hugging Face repo ID、計算精度、言語、初期プロンプト、batch size、beam 数である。

Hugging Face Whisper の checkbox を切り替えると、標準 Whisper のモデルと計算精度、または Hugging Face の repo ID と batch size のどちらかを表示する。

「音声の文字起こし」は `transcribe.py` を呼ぶ。標準経路では `--model`、`--compute_type`、`--language`、`--initial_prompt`、`--num_beams` を渡す。Hugging Face 経路ではさらに `--use_hf_whisper`、`--batch_size` を渡し、repo ID が `openai/whisper` でなければ `--hf_repo_id` も渡す。画面の repo 選択肢はいずれもこの引数を渡す。特定の repo は初期プロンプトを空にして呼び出す。

モデル名が空なら結果欄へエラーを表示する。成功時は完了文を表示し、失敗時は下位メッセージと `esd.list` の確認を促す文を表示する。文字起こし結果の保存先は `Data/{モデル名}/esd.list` として案内される。

## 学習タブ

### 前提データと共通設定

学習対象は `Data/{モデル名}` に置く。前処理は `raw/` の音声と `esd.list` を入力とし、`wavs/`、`train.list`、`val.list`、`config.json`、特徴ファイルを生成する。

自動前処理の UI はモデル名のほか、JP-Extra、batch size、epochs、保存間隔、音量正規化、無音除去、読みエラー時の扱いを受け取る。詳細設定ではプロセス数、言語ごとの検証データ数、Tensorboard のログ間隔、言語別 BERT・style・decoder の凍結を受け取る。

### 自動前処理

「自動前処理を実行」は `preprocess_all()` を呼び、次の段階を順に実行する。失敗した段階で後続を実行せず、その段階のメッセージを状況欄へ表示する。

| 段階 | 呼び先 | 主な結果 |
| --- | --- | --- |
| 1. 初期設定 | `initialize()` | 設定テンプレートと事前学習モデルを対象 dataset へ用意し、`config.json` とルートの `config.yml` を更新する |
| 2. 音声前処理 | `resample.py` | `raw/` から `wavs/` へ 44100 Hz で変換し、指定時は正規化・無音除去する |
| 3. テキスト前処理 | `preprocess_text.py` | `esd.list` から `train.list` と `val.list` を作り、音声パスを補正する |
| 4. BERT 特徴生成 | `bert_gen.py` | dataset の `config.json` に基づき BERT 特徴ファイルを生成する |
| 5. スタイル特徴生成 | `style_gen.py` | 指定プロセス数でスタイル特徴ファイルを生成する |

モデル名が空の場合、自動前処理は `Error: モデル名を入力してください` を表示する。初期設定は対象の既存 `models/` を `models_backup/` へコピーしてから置き換える。選択した構造に対応する `pretrained/` または `pretrained_jp_extra/` がなければ Step 1 のエラーとなる。前処理ログは dataset 内の日時付きファイルにも保存する。

### 手動前処理

「手動前処理」は自動前処理と同じ 5 段階を個別ボタンとして公開し、各段階の状況欄へ `second_elem_of()` でメッセージだけを表示する。利用者が順序と前提を管理する。

手動の初期設定と学習開始は、空のモデル名を UI 用エラーへ変換せず `get_path()` の assertion に到達する。後続段階も前段の生成物がない場合は、明示エラーまたは Gradio のイベントエラーになる。

### 学習と Tensorboard

「学習を開始する」はモデル名、スタイルファイル生成の skip、JP-Extra、カスタム batch sampler の無効化を `train()` へ渡す。JP-Extra の選択に応じて `train_ms_jp_extra.py` または `train_ms.py` を呼び、必要に応じて `--skip_default_style` と `--not_use_custom_batch_sampler` を追加する。成功・失敗と下位メッセージは状況欄へ表示する。

自動前処理または手動 Step 1 の JP-Extra checkbox は、学習欄の JP-Extra checkbox に同じ値を反映する。非表示の高速化 checkbox が true の場合は `--speedup` も渡す。

「Tensorboardを開く」は最初の操作時だけ `python -m tensorboard.main --logdir Data/{モデル名}/models` を別プロセスで起動する。ボタンを「起動中…」に変え、10 秒以内に localhost の port 6006 が開けば実行済みとする。その後ブラウザーで `http://localhost:6006` を開き、ボタンを元に戻す。起動確認が失敗した場合は logger にエラーを出すが、専用の状況欄は更新しない。

## スタイル作成タブ

全方式でモデル名を共有する。保存先の `model_assets/{モデル名}/config.json` が必要であり、方式 1 と 2 は `Data/{モデル名}/wavs` の音声別 `.npy` も前提とする。

### 方法 0: サブフォルダ単位

モデル名と音声ディレクトリを入力し、「スタイルベクトルを作成」を押す。対象拡張子は `.wav`、`.flac`、`.mp3`、`.ogg`、`.opus`、`.m4a` である。

各音声に対応する `.npy` がなければ `style_gen.save_style_vector()` で生成し、`default_style.save_styles_by_dirs()` へ音声ディレクトリ、出力先、config 入出力先を渡す。既存の `config.json` と `style_vectors.npy` は `.bak` へコピーしてから更新する。

モデル名または音声ディレクトリが空の場合、結果欄へ入力を求める文を表示する。`config.json` がない場合はそのパスを含むメッセージを表示する。音声ごとの特徴生成例外は収集されるが、結果欄には個別に列挙されない。

### 方法 1: 自動分類

「スタイルベクトルを読み込む」は `Data/{モデル名}/wavs/**/*.npy` を読み、全体平均と元の 256 次元ベクトルを process-global state に保持する。この state は browser session ごとには分離されない。選択した UMAP または t-SNE で 2 次元化し、散布図を表示する。

分類方法は次の二系統である。

- 指定クラスタ数: KMeans または AgglomerativeClustering を、元ベクトルまたは次元削減後のベクトルへ適用する
- DBSCAN: 次元削減後のベクトルへ `eps` と `min_samples` を適用する

分類結果は散布図、選択可能なスタイル番号、DBSCAN の場合は検出スタイル数として表示する。DBSCAN のクラスタ数が 10 を超える場合または 0 の場合は、パラメータ変更を促す文を表示する。「代表音声を取得」は選択クラスタ内で平均距離が小さい音声を最大 10 件表示する。

「スタイルベクトルを保存」は全体平均を `Neutral`、各 centroid を入力されたスタイル名として `style_vectors.npy` に保存し、`config.json` の `num_styles` と `style2id` を更新する。既存ベクトルと config は `.bak` へコピーする。実装はベクトルを保存してから config の存在、スタイル数、重複名を検査するため、これらのエラーを保存結果欄へ表示した時点で `style_vectors.npy` は更新済みになり得る。

### 方法 2: 手動選択

音声ファイル名とスタイル名をそれぞれ comma 区切りで入力する。「スタイルベクトルを保存」は全体平均を `Neutral` とし、各音声に対応する `Data/{モデル名}/wavs/<音声>.npy` を追加して保存する。

事前の「スタイルベクトルを読み込む」が未実行、入力数の不一致、重複名、音声または config の不在は保存結果欄へ表示する。config の存在確認はベクトル保存後なので、config 不在時も `style_vectors.npy` は更新済みになり得る。その他の load、次元削減、分類中の例外は捕捉されず、Gradio のイベントエラーとなる。

## マージタブ

### 初期状態とマージ方法

利用可能なモデルがなければ、モデル配置先を含むエラー Markdown のみを表示する。モデルがある場合はモデル A と B、必要な方式では C を選ぶ。モデル名変更は対応する safetensors ファイル一覧を更新し、「更新」は A、B、C の全一覧を再走査する。

マージ方法の radio は、説明、モデル C、係数、要素別 slider、球面線形補間 checkbox の表示を切り替える。

| 方法 | 式 | UI で指定する重み |
| --- | --- | --- |
| 通常マージ | `(1 - weight) * A + weight * B` | 声質、声の高さ、話し方、テンポ。線形補間または球面線形補間 |
| 差分マージ | `A + weight * (B - C)` | 4 要素の重み |
| 加重和 | `a * A + b * B + c * C` | A、B、C の係数。全 parameter に適用 |
| ヌルモデルマージ | `A + weight * B` | 4 要素の重み |

要素別マージでは、parameter key の prefix により `dec` を声質、`flow` を声の高さ、`enc_p` を話し方、`sdp` と `dp` をテンポとして扱う。それ以外はモデル A の値を維持する。

### モデルファイルのマージ

「モデルファイルのマージ」は選択した方式で safetensors を生成し、`model_assets/{新しいモデル名}/{新しいモデル名}.safetensors` に保存する。同じディレクトリへ次も保存する。

- モデル A の config を基にモデル名と話者名を調整した `config.json`
- 適用方式と入力を記録した `recipe.json`
- 各入力モデルの先頭スタイルを同じ方式で混ぜた、`Neutral` 1 件の `style_vectors.npy`

新しいモデル名が空、または入力モデルと同じ名前の場合は情報欄へ `Error: ...` を表示する。成功時は保存先を含む文を表示し、簡易合成のスタイル選択を `Neutral` にする。出力ディレクトリや同名ファイルが既にある場合の上書き確認は行わない。

空のモデル名に対する `merge_models_gr()` の返値は文字列 1 件だけである一方、イベントは情報欄とスタイル欄の 2 出力を要求する。この場合の画面更新は Gradio の出力数検証に依存する。ファイル不在、互換性のない parameter、読み書き例外も捕捉せず、Gradio のイベントエラーとなる。

### 結果の簡易音声合成

テキスト、スタイル、スタイルの強さを入力し、「音声合成」を押す。`model_assets/{新しいモデル名}` の同名 safetensors、`config.json`、`style_vectors.npy` から `TTSModel` を都度作成し、`TTSModel.infer()` を呼ぶ。

成功時は情報欄へ完了文、音声欄へ推論結果を表示する。モデル名が空なら情報欄へエラーを表示し、音声を返さない。その他の例外は Gradio のイベントエラーとなる。

### スタイルのマージ

画面には「作るスタイルの数」、「各モデルのスタイルを取得」、方式に応じて動的に描画されるスタイル選択行、「スタイルを増やす」「スタイルを減らす」、「スタイルのマージ」がある。この領域には固定 Markdown `Hello world!` も表示される。

「各モデルのスタイルを取得」は A、B、C の `config.json` から `style2id` の key を state へ読む。各スタイル行は入力モデルのスタイルと出力名を持ち、先頭行は `Neutral` で固定される。入力スタイル変更時は、すべて `Neutral` なら `Neutral`、それ以外なら入力名を underscore で連結した名前を出力名へ設定する。

「スタイルのマージ」は画面上の組を方式別の style merge 関数へ渡す。通常、差分、ヌルモデル方式は「話し方」の重みを使い、加重和はモデル A、B、C の係数を使う。`style_vectors.npy`、`config.json`、`recipe.json` を更新し、成功時は情報欄と簡易合成のスタイル一覧を更新する。存在しないスタイル名は `ValueError`、その他のファイル・形式エラーとともに Gradio のイベントエラーとなる。

## ONNX 変換タブ

利用可能なモデルがなければ、配置先を含むエラー Markdown のみを表示する。モデル名変更は `TTSModelHolder.update_model_files_for_gradio()` でモデルファイル一覧を更新し、「更新」はモデル名とファイルを再走査する。

「ONNX形式に変換」は選択したモデルファイルを次の形で `run_script_with_log()` から呼ぶ。

```text
convert_onnx.py --model <選択したモデルファイル>
```

選択値が空なら情報欄へ `Error: モデル名を入力してください。` を表示する。成功時は完了文、失敗時は `Error: ` と下位メッセージを表示する。変換進捗は画面ではなくターミナルログで確認する。生成物は選択ファイルと同じ名前の `.onnx` として案内される。変換処理の詳細は [conversion-merge.md](conversion-merge.md) が扱う。

## タブ間の前提関係

代表的な利用順序と成果物の関係は次のとおりである。

1. データセット作成タブで `Data/{モデル名}/raw` と `esd.list` を作る、または利用者が同等の入力を配置する。
2. 学習タブで音声・テキスト・BERT・スタイルの前処理を行い、学習済みモデルを作る。
3. スタイル作成タブは、学習前処理が生成した音声別スタイル特徴と、モデル資産の config を利用してスタイルを再構成できる。
4. 音声合成タブは、モデル資産の config、モデルファイル、スタイルベクトルをロードして合成する。
5. マージタブは複数のモデル資産を入力として新しいモデル資産を作り、その場で簡易合成できる。
6. ONNX 変換タブは safetensors モデルを入力として ONNX モデルを作る。

データセット作成タブは必須ではないが、学習前処理が要求するディレクトリと `esd.list` は別の方法で用意する必要がある。「更新」を持つ各タブは、同一 WebUI プロセス中に追加されたモデルを一覧へ反映できる。

## エラーと進捗の表示規則

- dataset、train、ONNX 変換の下位 script は `run_script_with_log()` で実行され、結果欄または状況欄には成功・失敗メッセージを表示する。詳細な進捗はターミナルへ出る。
- 音声合成は既知の tone と値エラーを情報欄へ変換する。
- スタイル作成とマージは一部の入力エラーだけを結果欄へ変換し、ファイル I/O、数値計算、形式不整合の多くを捕捉しない。
- タブ構築時にモデルが 0 件なら、音声合成、マージ、ONNX 変換は各タブ内をエラー Markdown に置き換える。
- Gradio が捕捉する未処理例外の表示内容は、Gradio の実行時挙動に従う。

## 関連テスト

- `tests/test_onnx_export_restoration.py`: ONNX 変換 UI が AIVM / AivisSpeech 系の外部ツールへ誘導しないことをソース文字列で検証する。

`app.py` および各 `gradio_tabs/*.py` のイベント結線、入力値、表示結果を直接検証する専用テストは存在しない。
