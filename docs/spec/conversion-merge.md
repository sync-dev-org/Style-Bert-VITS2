# モデル変換・マージ・スタイルベクトル仕様

## 対象範囲

この文書は、次の処理の入出力と挙動を定義する。

- `convert_onnx.py` による音声合成モデルの ONNX 変換
- `convert_bert_onnx.py` による言語別 BERT モデルの ONNX 変換
- `gradio_tabs/convert_onnx.py` から呼び出す変換処理
- `gradio_tabs/merge.py` のモデル重みおよびスタイルベクトルのマージ
- `gradio_tabs/style_vectors.py` と `default_style.py` のスタイルベクトル生成

ONNX 音声合成モデルのロードと推論時のテンソル処理は
[core-inference.md](core-inference.md)、BERT ONNX モデルのロードと特徴量抽出は
[nlp.md](nlp.md)、各機能の WebUI 構成は [webui.md](webui.md) が扱う。

## モデル資産の共通契約

### 音声合成モデル一式

音声合成モデルは同じモデルディレクトリ内の設定、スタイルベクトル、モデルファイルを
組み合わせて使用する。ONNX 変換後も変換元を削除しないため、変換したディレクトリには
通常、次の資産が共存する。

以下の `<assets-root>` は path 設定の `assets_root` を表し、既定値は `model_assets` である。

```text
<assets-root>/<model-name>/
├── config.json
├── style_vectors.npy
├── <checkpoint>.safetensors
└── <checkpoint>.onnx
```

- `config.json` はモデル構造、話者、スタイル数、`style2id` などを定義する。
- `style_vectors.npy` は `config.json` のスタイル定義に対応するベクトル配列である。
- `.safetensors` は変換元およびモデルマージの入力である。
- `.onnx` は ONNX Runtime 推論で使用する音声合成モデルである。

`convert_onnx.py` が新規生成する資産は `.onnx` だけであり、`config.json` と
`style_vectors.npy` は変換元と同じものを再利用する。したがって、ONNX モデルを単体で
配置しても音声合成モデル一式にはならない。

ONNX Runtime による音声合成に必要なモデル側の最小構成は `config.json`、
`style_vectors.npy`、`.onnx` の 3 点であり、`.safetensors` は実行時には不要である。

ONNX 推論では音声合成モデルに加えて、対象言語の ONNX BERT モデルとトークナイザーが
必要になる。音声合成モデルと BERT モデルの双方を ONNX Runtime で実行する経路では、
PyTorch のモデル重みは実行時入力にならない。一方、ONNX への変換処理自体は PyTorch
モデルをロードしてエクスポートするため、PyTorch と ONNX 変換用依存を必要とする。

### スタイルベクトルの整合条件

`style_vectors.npy` は行ごとに 1 スタイルを表す。生成処理が扱う各ベクトルは 256 次元で、
配列の形状は `(num_styles, 256)` になる。

- 0 行目は平均スタイル `Neutral` として扱われる。
- `config.json` の `data.num_styles` は配列の行数と一致する必要がある。
- `data.style2id` の各値は対応する行番号である。
- 推論時は `style2id` の要素数と `num_styles`、および配列の行数が一致しなければ
  モデル初期化が失敗する。

## 音声合成モデルの ONNX 変換

### エントリポイントと対象列挙

```bash
python convert_onnx.py --model <model-or-directory> [--force-convert]
```

`--model` がディレクトリの場合は、その配下を再帰的に検索し、ファイル名が `.` で
始まらないすべての `.safetensors` を変換対象にする。ファイルの場合は指定されたパスを
そのまま 1 件の対象として扱う。

各変換元 `<name>.safetensors` に対し、同じディレクトリの次のパスを使用する。

| 用途 | パス |
|---|---|
| 変換元モデル | `<name>.safetensors` |
| モデル設定 | `config.json` |
| スタイルベクトル | `style_vectors.npy` |
| 一時出力 | `<name>_temp.onnx` |
| 最終出力 | `<name>.onnx` |

変換元、設定、スタイルベクトルが存在しない場合は assertion で停止する。最終出力がすでに
存在し、`--force-convert` がない場合は変換をスキップする。`--force-convert` がある場合は
最終出力を上書きする。

### エクスポート前処理

変換元は CPU 上の `TTSModel` としてロードする。エクスポート用入力は固定の日本語文から
音素、アクセント、言語 ID、BERT 特徴量を生成し、`Neutral` が `style2id` にあればその
スタイルを、なければスタイル ID 0 を使用する。話者 ID は 0、推論制御値は次の固定値である。

| 入力 | 値 |
|---|---:|
| `length_scale` | `1.0` |
| `sdp_ratio` | `0.0` |
| `noise_scale` | `0.667` |
| `noise_scale_w` | `0.8` |

ロードされたネットワークの型により JP-Extra と通常モデルを分岐し、`forward` を
推論用の `infer` 呼び出しへ置き換えてエクスポートする。

### ONNX 入出力

両アーキテクチャの共通入力は次のとおりである。

- `x_tst`: 音素 ID
- `x_tst_lengths`: 音素列長
- `sid`: 話者 ID
- `tones`: アクセント・音高 ID
- `language`: 言語 ID
- `style_vec`: スタイルベクトル
- `length_scale`, `sdp_ratio`, `noise_scale`, `noise_scale_w`: 推論制御値

JP-Extra は日本語 BERT 特徴量だけを `bert` という入力名で受ける。通常モデルは
`bert`、`ja_bert`、`en_bert` の 3 系統を受ける。出力名はいずれも `output` である。

batch 軸と音素列長に依存する軸は dynamic axes としてエクスポートする。BERT 特徴量は
第 2 軸を音素列長に対応させ、`style_vec` は batch 軸だけを可変にする。

### エクスポート、簡略化、後処理

- `torch.onnx.export` は TorchScript exporter (`dynamo=False`) を使用する。
- opset は 20 に固定する。
- 一時 ONNX を `onnxsim.simplify` に渡し、返されたモデルを最終出力へ保存する。
- 簡略化の `check` 戻り値は成否判定に使用しない。
- 保存に成功すると一時 ONNX を削除する。

この処理は生成した音声合成 ONNX を ONNX Runtime で実行して PyTorch 出力と比較する
検証を行わない。途中で例外が発生した場合は、一時ファイルが残ることがある。

### WebUI からの呼び出し

`gradio_tabs/convert_onnx.py` は選択されたモデルファイルを
`convert_onnx.py --model <path>` としてサブプロセス実行する。空のモデル名はエラーを返し、
サブプロセスが失敗した場合はログ由来のメッセージをエラーとして返す。
`--force-convert` は WebUI から指定しないため、既存の同名 ONNX はスキップされる。

## BERT モデルの ONNX 変換

### エントリポイントと出力先

```bash
python convert_bert_onnx.py --language <JP|EN|ZH>
```

`--language` の既定値は `JP` である。変換元は言語ごとの
`DEFAULT_BERT_MODEL_PATHS` が指す PyTorch BERT モデルディレクトリであり、出力も同じ
ディレクトリへ保存する。

| 用途 | ファイル名 |
|---|---|
| 一時 ONNX | `model_temp.onnx` |
| 簡略化済み FP32 | `model.onnx` |
| 簡略化済み FP16 | `model_fp16.onnx` |

既定の ONNX 推論コードは別途 `DEFAULT_ONNX_BERT_MODEL_PATHS` が指す言語別ディレクトリの
`model_fp16.onnx` と、そのディレクトリのトークナイザーをロードする。
`convert_bert_onnx.py` は変換結果やトークナイザーをその既定 ONNX ディレクトリへコピーしない。
したがって、変換結果を既定の ONNX 推論経路で使用するには、参照先として明示するか、
ONNX 推論側が参照するモデル資産へ別途配置する必要がある。

### モデルラッパーと入出力

変換用ラッパーは元の BERT に `output_hidden_states=True` を指定し、末尾から 3 番目の
hidden state を batch 0 について返す。ONNX の入力名は次の 3 つ、出力名は `output` である。

- `input_ids`
- `token_type_ids`
- `attention_mask`

tokenizer 出力に `token_type_ids` がない場合は、`input_ids` と同じ shape、dtype、device の
ゼロテンソルを補う。3 入力の batch 軸と sequence 軸を dynamic axes として扱う。
exporter は音声合成モデルと同じく `dynamo=False`、opset 20 である。

### FP32 と FP16 の生成

一時 ONNX を `onnxsim.simplify` で簡略化し、FP32 の `model.onnx` として保存する。
そのモデルを `onnxconverter_common.float16` で FP16 化し、`model_fp16.onnx` として保存する。

- `keep_io_types=True` によりモデルの浮動小数点入出力型は FP32 のまま維持する。
- shape inference は無効化する。
- FP16 の value info に接続する `Cast` の属性が FP32 を指している場合は FP16 に補正する。
- 補正は main graph と node attribute 内の subgraph を再帰的に処理する。

FP32、FP16 の両方の検証に成功した後、一時 ONNX を削除する。途中で失敗した場合、
それまでに書き出したファイルはロールバックしない。

### 出力検証

検証は `CPUExecutionProvider` の ONNX Runtime セッションを使用し、言語別の固定テキスト群で
PyTorch 出力と ONNX 出力の絶対差を比較する。各テキストの最大差の全体最大値と、各テキストの
平均差の全体最大値が、両方とも次の閾値未満であることを要求する。

| 精度 | 最大差 | 平均差 |
|---|---:|---:|
| FP32 | `1e-3` | `1e-4` |
| FP16 | `2.5e-1` | `2e-3` |

閾値以上なら `RuntimeError` で停止する。比較時には ONNX セッションが公開する入力名だけを
渡すため、簡略化で未使用入力が除去されても検証できる。

## モデル重みのマージ

### 入力と出力

マージ対象は `.safetensors` であり、ONNX モデルは候補から除外する。入力の全 tensor は
CPU にロードし、次の出力一式を `<assets-root>/<output-name>/` に保存する。

```text
<output-name>/
├── <output-name>.safetensors
├── config.json
├── style_vectors.npy
└── recipe.json
```

出力ディレクトリは必要に応じて作成する。`config.json` はモデル A を基にし、
`model_name` を出力名へ変更する。話者数が 1 の場合は `spk2id` を
`{<output-name>: 0}` に置き換える。`recipe.json` は方式、入力モデル、係数を記録する。

モデルファイルのマージ直後はスタイル数を 1 にし、`style2id` を `{Neutral: 0}` にする。
各入力の `style_vectors.npy` の 0 行目を `Neutral` とみなして同じ方式でマージし、出力には
その 1 ベクトルだけを保存する。必要な追加スタイルは後段のスタイルマージで生成する。

### 対象コンポーネント

通常マージ、差分マージ、ヌルモデル加算では、parameter key の prefix により係数を分ける。

| UI 上の要素 | key prefix |
|---|---|
| 声質 | `dec` |
| 声の高さ | `flow` |
| 話し方・抑揚 | `enc_p` |
| 速さ・リズム・テンポ | `sdp`, `dp` |

これらに一致しない tensor はモデル A の値を維持する。重み付き和だけは prefix に関係なく
モデル A の全 key を演算対象にする。

### マージ方式

tensor またはベクトルを `A`、`B`、`C` とすると、各方式は次の式を使用する。

| 方式 | 式 | 係数の適用 |
|---|---|---|
| 通常マージ | `(1 - w) A + w B` | 4 コンポーネントに個別の `w` |
| 差分マージ | `A + w (B - C)` | 4 コンポーネントに個別の `w` |
| 重み付き和 | `a A + b B + c C` | 全 tensor に共通の `a`, `b`, `c` |
| ヌルモデル加算 | `A + w B` | 4 コンポーネントに個別の `w` |

WebUI の 4 コンポーネント用 slider は 0 以上 1 以下、刻み 0.1 である。重み付き和の係数は
数値入力であり、和が 1 になることや値域をコードで強制しない。低水準関数を直接呼ぶ場合、
4 コンポーネントの重みにも値域検証はない。

通常マージでは tensor ごとに linear interpolation の代わりに spherical linear
interpolation を選択できる。slerp は tensor 全体を 1 ベクトルとして正規化内積を計算し、
内積の絶対値が `0.998` を超える場合は linear interpolation にフォールバックする。
この選択はモデル tensor にだけ適用され、`Neutral` を含むスタイルベクトルは常に線形演算する。

### 互換性と上書き

- 出力名が使用した入力モデル名と同じ場合、WebUI wrapper はエラーを返す。
- それ以外の既存出力名は拒否せず、同名の各出力ファイルを上書きする。
- 入力間の key 集合、tensor shape、話者数、モデル version、アーキテクチャを事前検査しない。
- key がない場合は `KeyError`、shape が非互換なら tensor 演算時のエラーになる。
- 通常モデルと JP-Extra モデルの混在を明示的に拒否する分岐はないが、WebUI は
  1.x 系と 2.x-JP-Extra のマージが失敗するものとして案内しており、互換な入力ではない。
- 話者数が異なるモデルも互換性を保証しない。

## スタイルベクトルのマージ

モデル重みのマージ後、任意の入力スタイルの組を選び、複数の出力スタイルを生成できる。
通常マージとヌルモデル加算は A/B、差分マージと重み付き和は A/B/C のスタイル名を受け取る。

- 通常マージ、差分マージ、ヌルモデル加算はモデルマージの「話し方」の重みを使用する。
- 重み付き和はモデル A/B/C と同じ係数を使用する。
- 各入力スタイル名は対応する `config.json` の `style2id` に存在する必要がある。
- 出力ベクトルは指定順に並び、出力スタイル名へ 0 始まりの ID を割り当てる。
- `config.json` はモデル A を基に `num_styles`、`style2id`、`model_name` を更新する。
- `recipe.json` の既存内容を維持し、`style_tuple_list` を追加または上書きする。

ベクトル次元の一致や出力スタイル名の一意性は検証しない。出力スタイル名が重複すると
`style2id` は後の ID だけを保持する一方、配列には全ベクトルが残るため、推論時の整合条件を
満たさなくなる。空のスタイル指定も低水準関数では拒否しない。

## スタイルベクトルの生成

### 音声ごとのベクトル

音声からの元ベクトルは `style_gen.save_style_vector` が生成し、音声ファイル名に `.npy` を
付加して保存する。例えば `sample.wav` の出力は `sample.wav.npy` である。ベクトルに NaN が
含まれる場合は保存せず、エラーにする。

### 方法 0: サブディレクトリ単位

`save_style_vectors_by_dirs` は指定ディレクトリを再帰検索し、拡張子が
`.wav`, `.flac`, `.mp3`, `.ogg`, `.opus`, `.m4a` のファイルを対象にする。既存の
`<audio-name>.<ext>.npy` は再生成せず、不足分を thread pool で生成する。個々の生成例外は
処理結果として回収するが、この関数の戻り値では失敗件数を報告しない。

その後 `default_style.save_styles_by_dirs` が次の規則で集約する。

1. 全 `.npy` の平均を 0 行目の `Neutral` にする。
2. 直下のサブディレクトリを名前順に処理する。
3. 各サブディレクトリ配下の全 `.npy` の平均を、そのディレクトリ名のスタイルにする。
4. `.npy` がないサブディレクトリはスタイルに追加しない。
5. `num_styles` と `style2id` を生成結果に合わせて更新する。

直下のサブディレクトリが 0 または 1 個の場合は、サブディレクトリ別スタイルを作らず、
全 `.npy` の平均である `Neutral` だけを保存する。集約対象の `.npy` が 1 件もなければ、
配列の連結時に失敗する。

### 方法 1: クラスタリング

`dataset_root/<model-name>/wavs` 配下の全 `.npy` をロードし、対応する音声パスを
`.npy` suffix を外して記録する。全ベクトルの平均が `Neutral` になる。

可視化は cosine metric の UMAP または t-SNE で 2 次元化する。クラスタリング方式は次のとおり。

- KMeans または AgglomerativeClustering を元の 256 次元ベクトルに適用する。
- `after reduction` の 2 方式は 2 次元化した座標に適用する。
- DBSCAN は 2 次元化した座標に適用し、label `-1` の noise をスタイルから除外する。

どの方式でも、保存する各 cluster centroid は分類に使った空間ではなく、cluster に属する
元の 256 次元ベクトルの算術平均である。出力は `[全体平均, centroid ...]` の順になる。
代表音声は元の 256 次元空間で cluster 内の全 pair の Euclidean distance を求め、他の点への
平均距離が小さい順に選ぶ。

DBSCAN の cluster 数が 10 を超える場合、または 0 の場合は WebUI で保存候補として扱わず、
パラメータ変更を促す。KMeans 系と Agglomerative 系の WebUI 上限も 10 cluster である。

### 方法 2: 代表音声の手動指定

事前にロードした全ベクトルの平均を `Neutral` とし、指定された各音声の
`<audio-file>.npy` をそのまま後続行へ追加する。音声ファイル名とスタイル名は comma 区切りで、
件数が一致しなければ保存しない。音声本体と対応 `.npy` の両方が必要である。

### 保存、設定更新、バックアップ

WebUI の各保存方式は `<assets-root>/<model-name>/style_vectors.npy` を書き、同じディレクトリの
`config.json` の `data.num_styles` と `data.style2id` を更新する。既存のベクトルと設定は
それぞれ `.bak` suffix のファイルへコピーしてから上書きする。既存の `.bak` がある場合も
同名で上書きする。`default_style.save_neutral_vector` と
`default_style.save_styles_by_dirs` を直接呼ぶ経路でも、既存の `style_vectors.npy` と
config 出力先を同様に `.bak` へコピーしてから上書きする。

クラスタリング方式は `config.json` の存在、スタイル名数、重複をベクトル保存より前に
検査する。手動指定方式も `config.json` と各音声の存在をベクトル保存より前に検査する。
`default_style` の両関数も書き込み前に config を読むため、config が読めない場合に
`style_vectors.npy` だけが更新されることはない。

入力されたスタイル名同士の重複は拒否するが、入力に `Neutral` が含まれることは低水準関数で
拒否しない。その場合は先頭に自動追加される `Neutral` と `style2id` の key が衝突し、
`num_styles` と mapping の要素数が不一致になる。WebUI は手動指定時に `Neutral` を指定しない
よう案内する。

## 関連テスト

- `tests/test_onnx_export_restoration.py`
  - すべての `torch.onnx.export` が `dynamo=False` と opset 20 を明示すること
  - BERT 入力で欠落した `token_type_ids` をゼロテンソルで補うこと
  - FP16 graph の `Cast` 属性補正と FP16 検証閾値
- `tests/test_style_vectors_save.py`: WebUI 保存経路の検証順序とバックアップ生成
- `tests/test_default_style_save.py`: `default_style` 直接呼び経路の検証順序とバックアップ生成
  - ONNX 変換経路が AIVM/AIVMX 生成へ分岐しないこと

モデルマージとスタイルベクトル生成の数式、ファイル出力、エラー条件を直接検証するテストは
この文書の対象コードに対応する既存テスト群にはない。
