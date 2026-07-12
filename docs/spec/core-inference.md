# コアライブラリ: モデル読込と音声推論

## 対象範囲

この文書は、`style_bert_vits2.tts_model` を入口とする音声合成モデルの検出、設定読込、遅延ロード、推論、音声後処理の挙動を定義する。対象は次の系統である。

- `TTSModel`、`TTSModelHolder`、`TTSModelInfo`、`NullModelParam`
- PyTorch 推論の `models/infer.py` と ONNX 推論の `models/infer_onnx.py`
- `HyperParameters` とモデル checkpoint / safetensors の読込
- 通常モデルと JP-Extra モデルの推論時アーキテクチャ
- 推論を支える `commons.py`、`attentions.py`、`modules.py`、`transforms.py`、`monotonic_alignment.py`
- 推論後のピッチ・抑揚調整と 16-bit PCM 変換

テキストの正規化、g2p、BERT 特徴量生成の内部仕様は [nlp.md](nlp.md)、学習時のモデル動作は [training-pipeline.md](training-pipeline.md)、ONNX への変換処理は [conversion-merge.md](conversion-merge.md) が扱う。

## 公開エントリポイント

| エントリポイント | 役割 |
|---|---|
| `TTSModel(...)` | 1 個のモデルファイル、設定、スタイルベクトルを束ねる。設定とスタイルベクトルは生成時に読み、重みは遅延ロードする |
| `TTSModel.load()` | PyTorch の `SynthesizerTrn` または ONNX Runtime の `InferenceSession` を生成する |
| `TTSModel.unload()` | 保持中の PyTorch モデルまたは ONNX session を破棄する |
| `TTSModel.infer(...)` | テキストから音声を生成し、`(sampling_rate, int16_audio)` を返す |
| `TTSModel.get_style_vector(...)` | 保存済み style ID と重みから推論用 style vector を作る |
| `TTSModel.get_style_vector_from_audio(...)` | 参照音声から style vector を生成する |
| `TTSModelHolder(...)` | model asset のディレクトリを走査し、選択された 1 モデルをキャッシュする |
| `TTSModelHolder.refresh()` | model asset の索引を作り直す |
| `TTSModelHolder.get_model(...)` | 索引内のモデルを `TTSModel` として取得する |

## model asset の構成

### ディレクトリ形式

`TTSModelHolder` は root 直下の各ディレクトリを 1 モデルとして扱う。

```text
model_assets/
└── <model-name>/
    ├── config.json
    ├── style_vectors.npy
    ├── <arbitrary-name>.safetensors
    └── <arbitrary-name>.onnx
```

- 重みファイル名は任意である。
- holder が候補にする拡張子は `.pth`、`.pt`、`.safetensors`、`.onnx` である。
- `ignore_onnx=True` の holder は `.onnx` を候補から除外する。
- `config.json` がないディレクトリ、または候補重みがないディレクトリは warning を出して除外する。
- 名前が `.` で始まるディレクトリと重みファイルは除外する。
- モデルディレクトリは名前順、各ディレクトリ内の重みは更新時刻の降順で並ぶ。
- `style_vectors.npy` の存在は `refresh()` では検査せず、`TTSModel` 生成時の `numpy.load()` で読み込む。

### `config.json`

`HyperParameters.load_from_json()` は UTF-8 の JSON 全体を Pydantic model として検証する。トップレベルは次の構造を持つ。

| セクション | 推論で使う主要フィールド |
|---|---|
| top level | `model_name`、`version` |
| `train` | `segment_size`。モデル構築時に `segment_size / hop_length` を渡す |
| `data` | `sampling_rate`、`filter_length`、`hop_length`、`add_blank`、`n_speakers`、`spk2id`、`num_styles`、`style2id` |
| `model` | channel 数、Transformer 層数、resblock、upsample、speaker condition、flow、discriminator 関連設定 |

主要な `data` フィールドの意味は次のとおりである。

- `sampling_rate`: `infer()` の返却 sample rate。標準値は 44100。
- `filter_length`: spectrogram channel 数の算出に使い、`filter_length // 2 + 1` がモデルへ渡る。
- `hop_length`: 学習 segment を frame 数に変換する分母。
- `add_blank`: true の場合、phone、tone、language の各要素間と両端に ID `0` を挿入する。
- `n_speakers`: speaker embedding の要素数。
- `spk2id`: speaker 名から整数 ID への対応。`TTSModel.id2spk` はこれを反転した辞書である。
- `num_styles`: style 数。`style2id` と `style_vectors` の第 1 次元の検証基準になる。
- `style2id`: style 名から整数 ID への対応。

主要な `model` フィールドは、`inter_channels`、`hidden_channels`、`filter_channels`、`n_heads`、`n_layers`、`kernel_size`、`p_dropout`、`resblock`、`resblock_kernel_sizes`、`resblock_dilation_sizes`、`upsample_rates`、`upsample_initial_channel`、`upsample_kernel_sizes`、`gin_channels` である。`slm` は WavLM 系 discriminator の設定を保持するが、音声生成の `infer()` には渡されない。

JSON で省略されたフィールドには `hyper_parameters.py` の既定値が使われる。モデル種別の実際の選択条件は `data.use_jp_extra` ではなく、`version.endswith("JP-Extra")` である。

### `style_vectors.npy`

`TTSModel` は NumPy 配列をそのまま受け取るか、指定 path を `numpy.load()` して保持する。

- 第 1 次元は `num_styles` と一致しなければならず、不一致は `ValueError` になる。
- モデルの style projection は 256 次元を入力とする。
- index `0` は style 補間の基準となる mean vector として扱う。
- 第 2 次元が 256 であることは `TTSModel` の生成時には検査しない。

### PyTorch 重み

`get_net_g()` は設定からモデル構造を生成し、device へ移動して eval mode にした後、拡張子で loader を選ぶ。

| 拡張子 | loader | 内容 |
|---|---|---|
| `.pth` / `.pt` | `load_checkpoint(..., skip_optimizer=True)` | `model`、`iteration`、`learning_rate` などを持つ checkpoint 辞書から model state を読む |
| `.safetensors` | `load_safetensors(..., for_infer=True)` | tensor key を直接読み、任意の `iteration` を認識する |

どちらの loader も `.weight_g` / `.weight_v` key を parametrization 形式の key に移す。safetensors は `strict=False` で読み、推論用読込では `enc_q` の missing key を許容する。それ以外の missing / unexpected key はログへ出す。未対応拡張子は `ValueError` になる。

### ONNX 重み

model path の suffix が厳密に `.onnx` の場合だけ ONNX モデルとして扱う。ONNX session は指定された `onnx_providers` をそのまま `InferenceSession` へ渡す。

- provider list は 1 件以上必要であり、空なら load 時に assertion error になる。
- 先頭指定が `DmlExecutionProvider` の場合は graph optimization を `ORT_ENABLE_ALL` にする。
- それ以外の先頭指定では、変換済み model の再最適化を避けるため `ORT_DISABLE_ALL` にする。
- session と ONNX Runtime の log severity は error 相当に設定する。
- 実際に選択された provider は `onnx_session.get_providers()[0]` で決まる。

## `TTSModel` の生成とライフサイクル

### 生成時に行うこと

```python
TTSModel(
    model_path,
    config_path,
    style_vec_path,
    device="cpu",
    onnx_providers=[("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})],
)
```

- `config_path` は path または構築済み `HyperParameters` を受け取る。
- `style_vec_path` は path または構築済み `numpy.ndarray` を受け取る。
- object / array を直接渡した場合、互換用の `config_path` / `style_vec_path` 属性には空の `Path` が入る。
- config と style vector はこの時点で読み込む。
- PyTorch の `net_g` と ONNX の `onnx_session` は `None` のままであり、重みはロードしない。
- `style2id` の件数と `num_styles`、style vector の件数と `num_styles` を検証する。

### 遅延ロードと再利用

`infer()` は選択された backend のモデルが未ロードなら `load()` を呼ぶ。以後の呼び出しは同じ `net_g` または `onnx_session` を再利用する。

`force_reload_model=True` は該当 backend の保持属性を `None` にしてから、同じ `infer()` 内でロードし直す。この経路は `unload()` を呼ばず、明示的な garbage collection や CUDA cache clear も行わない。

`unload()` は次を行う。

- `net_g` があれば削除して `None` にする。
- PyTorch が認識する CUDA が利用可能なら CUDA cache を空にする。
- `onnx_session` があれば削除して `None` にする。
- Python の garbage collection を実行する。

### null model の重み加算

`NullModelParam` は `name`、`path` と、0.0 以上 1.0 以下に制約された `weight`、`pitch`、`style`、`tempo` を持つ。PyTorch モデルのロード時だけ、対象モデルへ null model の parameter を加算する。

| parameter | 加算対象 |
|---|---|
| `weight` | decoder (`dec`) |
| `pitch` | flow (`flow`) |
| `style` | text encoder (`enc_p`) |
| `tempo` | stochastic / deterministic duration predictor (`sdp` と `dp`) |

加算は対象 parameter に `null_parameter * weight` を足す。正規化や、複数モデル間の重み総和の制約は行わない。

null model の指定はロード契機でだけ反映される。すでに `net_g` がロード済みの `infer()` に `null_model_params` を渡しても、その呼び出しでは再ロードされないため重みは変わらない。一度加算したモデルは、指定を省略した次の推論でもロード済みのまま再利用される。反映または解除を確実に行うには `force_reload_model=True` を併用する。ONNX 推論は `null_model_params` を無視する。

## `TTSModelHolder`

### 索引

生成時に `refresh()` を呼び、次の属性を構築する。

- `model_files_dict`: model 名から候補重み path の list への対応。
- `model_names`: 有効な model 名の list。
- `models_info`: `name`、文字列化した `files`、`styles`、`speakers` を持つ `TTSModelInfo` の list。
- `current_model`: 最後に選択した `TTSModel`。refresh 時は `None` に戻る。

`styles` と `speakers` は config の `style2id` と `spk2id` の key 順を保つ。`refresh()` は既存 `current_model` に対して `unload()` を呼ばない。

### モデル取得とキャッシュ

`get_model(model_name, model_path_str)` は、model 名と path の両方が現在の索引に含まれることを検証する。未知の model 名または path は `ValueError` になる。

holder が保持するキャッシュは 1 個だけである。

- `current_model` がなく、または model path が異なる場合、新しい `TTSModel` を作る。
- model path が同じ場合、既存 instance を返す。
- instance の取得だけでは重みをロードしない。
- config と style vector は選択した model path と同じ model ディレクトリの固定名を使う。

`get_model_for_gradio()`、`update_model_files_for_gradio()`、`update_model_names_for_gradio()` は同じ索引とキャッシュを使い、Gradio component update を返す UI 用 adapter である。

## `TTSModel.infer()`

### 引数

| 引数 | 既定値 | 挙動 |
|---|---:|---|
| `text` | 必須 | 合成対象の文字列 |
| `language` | `Languages.JP` | `JP`、`EN`、`ZH` の言語指定 |
| `speaker_id` | `0` | speaker embedding の整数 ID |
| `reference_audio_path` | `None` | 指定時は保存済み style 名でなく参照音声から style vector を作る。空文字は `None` と同じ |
| `sdp_ratio` | `0.2` | duration の SDP 比率。計算式は `SDP * ratio + DP * (1 - ratio)` |
| `noise` | `0.6` | latent sampling の noise scale |
| `noise_w` | `0.8` | SDP の noise scale |
| `length` | `1.0` | duration 倍率。大きいほど出力が長い |
| `line_split` | `True` | 改行単位で別々に推論する |
| `split_interval` | `0.5` | 分割した音声間へ挿入する無音秒数 |
| `assist_text` | `None` | BERT 特徴量生成時の補助テキスト |
| `assist_text_weight` | `1.0` | 補助テキストの実効既定 weight |
| `use_assist_text` | `False` | false の場合は `assist_text` を無効化する |
| `style` | `"Neutral"` | `style2id` で引く style 名 |
| `style_weight` | `1.0` | mean vector から選択 style / 参照音声 vector への変化倍率 |
| `given_phone` | `None` | 指定済み phone 列。`given_tone` と組で使う |
| `given_tone` | `None` | 指定済み tone 列。`given_phone` と組で使う |
| `pitch_scale` | `1.0` | WORLD 後処理での平均 pitch 倍率 |
| `intonation_scale` | `1.0` | WORLD 後処理での平均からの偏差倍率 |
| `null_model_params` | `None` | PyTorch load 時の null model 加算指定 |
| `force_reload_model` | `False` | backend model をロードし直す |

これらの数値引数は `TTSModel.infer()` 自体では範囲を制限しない。`speaker_id` と `style` の有効範囲も事前検証せず、辞書 lookup または model 実行時の例外として現れる。

### style vector

保存済み style を使う場合、次の式で vector を作る。

```text
mean = style_vectors[0]
selected = style_vectors[style2id[style]]
result = mean + (selected - mean) * style_weight
```

`style_weight` は clamp しないため、0 は mean、1 は保存値、1 より大きい値は外挿になる。未知の style 名は `KeyError` になる。

参照音声を指定した場合、`pyannote.audio` を遅延 import し、`pyannote/wespeaker-voxceleb-resnet34-LM` を `window="whole"` で初期化して `device` へ移す。この推論 object は `TTSModel` 内で再利用する。得た x-vector にも mean からの同じ weight 式を適用する。package がない場合は `ImportError` を送出する。参照音声指定時、`style` 引数は使わない。

### 共通のテキスト入力生成

PyTorch と ONNX の両経路は、テキスト処理層から normalized text、phone、tone、word-to-phone 対応を取得し、phone / tone / language ID と 1024 次元 BERT 特徴量を作る。詳細な変換規則は [nlp.md](nlp.md) を参照する。

- `data.add_blank=True` なら phone、tone、language ID に blank ID `0` を intersperse し、word-to-phone 長も対応して増やす。
- BERT 特徴量末尾の長さと phone 長が一致しなければ assertion error になる。
- 通常モデルでは指定言語の BERT slot だけに特徴量を置き、残る 2 言語 slot を 0 で埋める。
- JP-Extra モデルでは Japanese BERT slot だけを model へ渡す。
- JP-Extra model に `JP` 以外の language を指定すると、推論開始前に `ValueError` になる。

### 改行分割

`line_split=True` の場合は `text.split("\n")` の空文字要素を除き、各行を独立して推論する。

- 行間には `int(data.sampling_rate * split_interval)` 個の 0 sample を挿入する (`data.sampling_rate` は config 由来)。
- 分割経路は `given_phone` と `given_tone` を下位推論へ渡さない。
- 空文字または空行だけの入力は推論結果 list が空になり、`numpy.concatenate()` が `ValueError` を送出する。
- `line_split=False` の場合だけ `given_phone` / `given_tone` を下位推論へ渡す。

### PyTorch と ONNX の分岐

| 項目 | PyTorch | ONNX |
|---|---|---|
| 選択条件 | model suffix が `.onnx` 以外 | model suffix が `.onnx` |
| model object | `SynthesizerTrn` / `SynthesizerTrnJPExtra` | `onnxruntime.InferenceSession` |
| 入力表現 | device 上の `torch.Tensor` | NumPy array から作る `OrtValue` |
| 実行 | `torch.no_grad()` 内の `net_g.infer()` | I/O Binding と `run_with_iobinding()` |
| BERT | PyTorch BERT 特徴量経路 | ONNX BERT 特徴量経路 |
| null model | load 時に対応 | 無視 |
| 実行後 | CUDA 利用可能時に cache clear | provider 別 memory arena shrink option を設定可能 |

ONNX 推論は session の input 名を宣言順に取得し、JP-Extra / 通常モデルごとの固定順 input tensor と `zip()` して bind する。出力は最初の output だけを bind し、`[0, 0]` の波形を返す。

実際の先頭 provider が CUDA なら `cuda`、DirectML なら `dml`、それ以外なら `cpu` に input/output を bind する。GPU の `device_id` は指定 provider option から取得し、未指定時と CPU は 0 を使う。CPU memory arena が有効な場合、CPU または CUDA の推論後に arena 縮小 option を付ける。DirectML ではこの縮小 option を付けない。

### 音響モデルの推論フロー

通常モデルと JP-Extra モデルは、入力 BERT slot の数を除き同じ生成フローを持つ。

1. speaker ID から global conditioning embedding を得る。
2. text encoder が phone、tone、language、BERT、style、speaker condition を加算し、Transformer encoder から prior の平均と log scale を出す。
3. SDP と DP の log duration を `sdp_ratio` で混合する。
4. `exp(log_duration) * length_scale` を切り上げ、phone-to-frame path を作る。
5. prior を path に沿って frame 長へ展開し、`noise` を使って latent を sample する。
6. flow を逆向きに通し、generator が upsample と residual block を経て `tanh` 波形を生成する。

通常モデルの text encoder は Chinese / Japanese / English の 3 BERT projection を持つ。JP-Extra は 1 個の BERT projection を持ち、Japanese BERT 特徴量だけを受け取る。`get_net_g()` は `version` suffix に応じて対応する class を構築する。

学習用 `forward()` で使う posterior encoder、monotonic alignment、discriminator 群は、`TTSModel.infer()` から呼ぶ生成経路には入らない。推論では duration から `commons.generate_path()` を作り、flow は reverse 方向に実行する。

### 内部 module の役割

| module | 推論に関係する役割 |
|---|---|
| `attentions.py` | text encoder と Transformer coupling flow の multi-head attention、FFN、mask 処理 |
| `modules.py` | WaveNet block、residual block、affine / convolution / Transformer coupling layer |
| `transforms.py` | stochastic duration predictor の convolution flow が使う rational quadratic spline |
| `commons.py` | blank 挿入、sequence mask、duration からの path 生成、segment 操作 |
| `monotonic_alignment.py` | 学習時の maximum path を NumPy / Numba で計算する。公開推論経路では使わない |
| `models/utils/weight_norm.py` | weight norm の適用・除去と `.weight_g` / `.weight_v` state dict key の移行 |

### 音声後処理と戻り値

`pitch_scale` と `intonation_scale` が両方 1.0 の場合、WORLD 後処理を省略する。それ以外の場合は `pyworld` で F0、spectral envelope、aperiodicity を抽出し、有声 frame の F0 を次の式で更新して再合成する。

```text
new_f0 = pitch_scale * voiced_f0_mean
       + intonation_scale * (old_f0 - voiced_f0_mean)
```

有声 F0 が 1 件もない波形では平均計算の除算が成立しない。

最後に `convert_to_16_bit_wav()` が波形を `numpy.int16` にする。

- floating point は最大絶対値で正規化し、32767 倍する。
- `int32` は 65536 で割る。
- `uint16` は 32768 を引く。
- `uint8` は 257 倍して 32768 を引く。
- `int8` は 256 倍する。
- `int16` はそのまま返す。
- その他の dtype は `ValueError` になる。

全 sample が 0 の floating-point 波形では最大絶対値も 0 になるため、正規化時に 0 除算が発生する。返却値は `(hyper_parameters.data.sampling_rate, int16 の一次元 NumPy 配列)` である。

## ログ

model load / unload、推論開始・完了、null model 加算、checkpoint key の不一致は共通の Loguru logger へ出力する。logger は既定 handler を除去し、safe stdout wrapper へ時刻、level、source file、line、message を出す。load と infer の完了ログには経過秒を含む。

## 例外と境界挙動

- config path、style vector path、model path が読めない場合、各 JSON / NumPy / PyTorch / ONNX loader の例外を伝播する。
- `num_styles` と `style2id` 件数、または style vector 件数が一致しない場合は `TTSModel` 生成時に `ValueError` になる。
- 未知の model 名・model path は `TTSModelHolder.get_model()` が `ValueError` にする。
- 未知の style 名は style lookup の `KeyError` になる。
- JP-Extra model への非 JP 言語指定は `ValueError` になる。
- PyTorch model の未対応拡張子は load 時に `ValueError` になる。
- ONNX provider list が空なら load 時に assertion error になる。
- `use_spk_conditioned_encoder=False` または `gin_channels=0` では、text encoder への speaker conditioning を無効化 (`enc_gin_channels=0`) して `SynthesizerTrn` を構築する。
- phone / tone の組、内容、長さに関する検証はテキスト処理層が担う。改行分割時は両指定を使わない。
- 参照音声 style の利用には `pyannote.audio` と、その model を初期化できる環境が必要である。

## 関連テスト

- `tests/test_tts_model_holder.py`
  - 有効な model asset の列挙
  - hidden / config 不足 directory の除外
  - 更新時刻順と `refresh()` の再索引
  - `ignore_onnx` の除外
  - 未知 model / path の拒否
  - `get_model()` が重みを遅延ロードすること
- `tests/test_main.py`
  - PyTorch CPU / CUDA と ONNX CPU / CUDA / DirectML / CoreML の合成経路
  - 返却 sample rate、非空・非無音の波形、WAV round trip
  - ONNX の実 provider 選択
- `tests/test_server_fastapi_api.py`
  - API 層から `TTSModelHolder` を利用する際の adapter 境界

`tests/test_main.py` の合成テストは model asset、対応 package、provider、GPU の有無に応じて skip される。CoreML case は SDP flow の動的 shape 制約を理由に non-strict xfail が付く。
