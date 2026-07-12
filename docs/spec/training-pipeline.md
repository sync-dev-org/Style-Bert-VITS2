# 前処理・学習パイプライン仕様

## 対象範囲

この文書は、学習用データセットを作成し、Style-Bert-VITS2 または JP-Extra モデルを学習して推論用資産を出力するまでの挙動を定義する。

対象は次の系統である。

- 音声の分割、文字起こし、リサンプリング
- テキスト正規化、学習・検証リストの作成
- BERT 特徴量とスタイルベクトルの生成
- 通常版と JP-Extra 版の学習
- 学習再開用チェックポイントと推論用モデル資産の保存
- 学習済み候補に対する SpeechMOS 評価

学習画面とその操作は [webui.md](webui.md)、推論時のモデル構造・読み込み・音声生成は [core-inference.md](core-inference.md)、テキスト正規化と BERT 特徴抽出そのものは [nlp.md](nlp.md) が扱う。

## パスと設定の正典

### グローバルパス

`config.py` は `configs/paths.yml` から次のルートを読む。ファイルがなければ `configs/default_paths.yml` をコピーして作成する。

| キー | 既定値 | 用途 |
|---|---|---|
| `dataset_root` | `Data` | 学習データセットのルート |
| `assets_root` | `model_assets` | 推論用モデル資産のルート |

モデル名を `<model_name>` とすると、通常のデータセットパスは `{dataset_root}/<model_name>`、推論用出力先は `{assets_root}/<model_name>` である。

### `config.yml`

リポジトリルートの `config.yml` は、前処理スクリプト群が共通で参照する YAML 設定である。存在しない場合は `default_config.yml` から生成される。主な契約は次のとおりである。

- `model_name`: 対象モデル名
- `dataset_path`: 指定時は `{dataset_root}/{model_name}` より優先されるデータセットパス
- `resample`: 入力 `raw`、出力 `wavs`、サンプリングレート 44100 Hz
- `preprocess_text`: `esd.list`、`train.list`、`val.list`、`config.json` の相対パス
- `bert_gen` / `style_gen`: `config.json`、device、並列数
- `train_ms`: DDP 環境変数、チェックポイントディレクトリ `models`、spec cache、保持世代数

`config.py` は `config.yml` が存在しない場合に生成する。既存ファイルを設定モデルとして読めない場合は `default_config.yml` で置き換える。

`gen_yaml.py --model_name ... --dataset_path ...` も `config.yml` を生成または読み込み、`model_name` と `dataset_path` の 2 項目だけを更新する。

### データセット内の `config.json`

`{dataset_path}/config.json` はモデルのハイパーパラメータとデータリストの位置を保持する。通常版は `configs/config.json`、JP-Extra 版は `configs/config_jp_extra.json` をテンプレートとする。

前処理の初期化は、モデル名、`train.list` / `val.list`、batch size、epochs、保存間隔、ログ間隔、freeze 設定、`data.use_jp_extra` を書き込む。`preprocess_text.py` はさらに `data.spk2id` と `data.n_speakers` を実データから更新する。

### ディレクトリ構造

標準の構成は次のとおりである。

```text
{dataset_root}/<model_name>/
├── raw/                         # 分割済みまたは持ち込み音声
├── wavs/                        # 44.1 kHz の学習音声
│   ├── <audio>.wav
│   ├── <audio>.bert.pt          # BERT 特徴量
│   ├── <audio>.wav.npy          # 発話単位の 256 次元 style vector
│   └── <audio>.spec.pt          # 学習時に任意生成される spectrogram cache
├── esd.list                     # 文字起こし
├── esd.list.cleaned             # 正規化済み全文
├── train.list
├── val.list
├── config.json
├── text_error.log               # テキスト処理失敗時
├── preprocess_<timestamp>.log
├── train_<timestamp>.log
└── models/                      # 再開用 checkpoint と TensorBoard log

{assets_root}/<model_name>/
├── config.json                  # 推論用 style 設定を含む config
├── style_vectors.npy
└── <model_name>_e<epoch>_s<step>.safetensors
```

`wavs/` 以下のサブディレクトリは維持される。学習開始時、直下のサブディレクトリが 2 個以上あれば各ディレクトリ名を style 名として平均ベクトルを作り、全発話平均を `Neutral` として先頭に置く。サブディレクトリが 0 または 1 個なら `Neutral` だけを作る。

## エントリポイント

| エントリポイント | 入力 | 主な出力 |
|---|---|---|
| `slice.py` | 任意の音声ディレクトリ | `raw/**/*.wav` |
| `transcribe.py` | `raw/**/*.wav` | `esd.list` |
| `preprocess_all.py` | `raw/` と既存の `esd.list` | `wavs/`、各 list、特徴量、`config.json` |
| `resample.py` | 任意の入力ディレクトリ | 相対構造を保った `.wav` |
| `preprocess_text.py` | 4 列の文字起こし | 7 列の cleaned/train/val list |
| `bert_gen.py` | train/val list | `<wav stem>.bert.pt` |
| `style_gen.py` | train/val list | `<wav path>.npy` |
| `train_ms.py` | 通常版 `config.json` と特徴量 | 通常版 checkpoint / model asset |
| `train_ms_jp_extra.py` | JP-Extra `config.json` と特徴量 | JP-Extra checkpoint / model asset |
| `speech_mos.py` | `model_assets` 内の safetensors | `mos_results/*.csv` と `*.png` |

`preprocess_all.py` は `slice.py` と `transcribe.py` を呼ばない。呼び出し時点で `{dataset_path}/raw` と `{dataset_path}/esd.list` が必要である。

## 音声準備

### 分割: `slice.py`

`slice.py --model_name <model_name>` は入力ディレクトリを再帰走査し、`.wav`、`.flac`、`.mp3`、`.ogg`、`.opus`、`.m4a` を対象とする。Silero VAD は 16 kHz で発話区間を検出し、元音声のサンプリングレートで切り出して `{dataset_root}/<model_name>/raw` に保存する。

- 既定の対象長は 2 秒以上 12 秒以下、分割点となる無音は 700 ms 以上である。
- 検出区間の前後には最大 200 ms の余白を加える。
- 出力名は既定で `<stem>-<index>.wav`、`--time_suffix` 指定時は `<stem>-<start_ms>-<end_ms>.wav` となる。
- 入力からの相対サブディレクトリを保つ。
- 出力 `raw/` が存在する場合は、処理開始前にディレクトリ全体を削除する。
- ファイル単位の失敗は全入力を処理した後にまとめて `RuntimeError` とする。

### 文字起こし: `transcribe.py`

`transcribe.py` は `raw/**/*.wav` を path 順に処理し、次の 4 列を `esd.list` に追記する。

```text
<raw からの相対 wav path>|<speaker name>|<JP|EN|ZH>|<text>
```

既存の `esd.list` は `esd.list.bak` に移動する。既存の `.bak` は先に削除される。対象 WAV が 0 件なら終了 status 1 で停止する。

既定経路は faster-whisper であり、`device` と `compute_type` をモデル生成へ渡す。指定した `compute_type` が拒否された場合は `compute_type` を省略して再生成する。`--use_hf_whisper` は Hugging Face pipeline を使い、CUDA 指定時に `float16`、それ以外で `float32` を使う。この経路は batch size を受け取る。

### リサンプリング: `resample.py`

`resample.py` は入力以下の全ファイルを読み込み対象として試行し、librosa で読める音声を mono の目標サンプリングレートへ変換する。拡張子は `.wav` に変換し、相対ディレクトリ構造を維持する。

- `preprocess_all.py` からは `raw/` → `wavs/`、44100 Hz で呼ばれる。
- `--normalize` は BS.1770 loudness を -23 LUFS へ正規化する。0.4 秒未満で測定不能な音声は正規化だけを省略する。
- `--trim` は先頭と末尾の無音を `top_db=30` で除去する。
- 読み込めないファイルは warning を記録してスキップする。
- 入力が空なら `ValueError` とする。
- 出力ディレクトリは削除しないため、入力から消えた既存出力は残る。

## 一括前処理

`preprocess_all.py` は `gradio_tabs.train.preprocess_all()` を呼び、各段が失敗した時点で後続を実行せず終了する。

### 1. 初期化

初期化は通常版または JP-Extra 版の JSON テンプレートを `{dataset_path}/config.json` に書き、リポジトリルートの `config.yml` を対象モデルへ向ける。

`{dataset_path}/models` が存在する場合は内容を `{dataset_path}/models_backup` へコピーしてから削除する。その後、通常版は `pretrained/`、JP-Extra 版は `pretrained_jp_extra/` を新しい `models/` としてコピーする。対応する pretrained ディレクトリがなければ初期化は失敗する。

このため、一括前処理は既存の再開用 `models/` をそのまま保持する操作ではない。再開用 checkpoint を使う学習では、既存の `models/` を維持した状態で学習エントリポイントを呼ぶ。

### 2. テキスト処理

`preprocess_text.py` は 4 列の `esd.list` を読み、各行を次の 7 列へ変換する。

```text
<wav path>|<speaker>|<language>|<normalized text>|<phones>|<tones>|<word2ph>
```

`preprocess_all.py` は `--correct_path` を指定するため、入力の相対 WAV path は `{esd.list の親}/wavs/<relative path>` へ変換される。結果は既定で `esd.list.cleaned` に全件を書き、存在する音声だけを `train.list` または `val.list` に振り分ける。

- 同じ音声 path の重複は 2 件目以降を除外する。
- 存在しない音声は除外する。
- speaker の初出順に `spk2id` を 0 から割り当てる。
- `val_per_lang` という引数名だが、抽出数は language 単位ではなく speaker 単位である。
- `val_per_lang=0` では全件を train に入れる。
- speaker の発話数より `val_per_lang` が大きい場合、random sample は失敗する。
- validation 全体が `max_val_total` を超えた分は train の末尾へ戻す。
- 読み・正規化エラーは `text_error.log` に記録する。`skip` は失敗行を除外して完了し、`raise` は全行処理後に失敗する。`use` は未知文字の読みエラーを許容するが、その他の例外があれば全行処理後に失敗する。

### 3. BERT 特徴量

`bert_gen.py` は `config.json` が指す train/validation list を連結し、各行の language に対応する BERT 特徴量を `<wav path>` の `.wav` を `.bert.pt` に置換した path へ保存する。

既存 cache を `weights_only=True` で読み、最終次元が blank 挿入後の phone 長と一致すれば再利用する。読み込みまたは shape 検証に失敗すると再生成する。実装上の executor は 1 worker に固定されている。`config.yml` の BERT device は CUDA が利用不能なら CPU へ置き換えられる。

### 4. スタイル特徴量

`style_gen.py` は pyannote の speaker embedding モデルを `config.yml` の style device へ配置し、train/validation の各 WAV から 256 次元ベクトルを生成して `<wav path>.npy` に保存する。

ベクトルに NaN が含まれる行は、対応する train/validation list から削除する。NaN 以外の抽出例外は処理全体を失敗させる。device は CUDA が利用不能なら CPU へ置き換えられる。

## 学習データの読み込み

`data_utils.TextAudioSpeakerLoader` は 7 列の list を読み、WAV、speaker ID、phone/tone/language、BERT cache、style vector を結合する。

- WAV のサンプリングレートが `config.json` と異なる場合は失敗する。
- spectrogram は `<wav stem>.spec.pt`、mel posterior encoder 使用時は `.mel.pt` を読む。cache がなければ `mel_processing.py` で計算し、`config.yml` の `train_ms.spec_cache` が真なら保存する。
- BERT cache の最終次元は phone 長と一致する必要がある。
- 通常版は ZH、JP、EN の 3 系統を 1024 次元テンソルとして構成し、対象 language 以外を 0 にする。
- JP-Extra 版は Japanese BERT 1 系統だけを batch に含める。
- style vector は常に `<wav path>.npy` から読む。

既定の `DistributedBucketSampler` は spectrogram 推定長を境界 `[32, 300, 400, 500, 600, 700, 800, 900, 1000]` で bucket 化する。最小境界以下または最大境界超の sample は除外される。`--not_use_custom_batch_sampler` では `DistributedLengthGroupedSampler` を使い、この bucket 境界による除外を行わない。

## 学習エントリポイントの分岐

`gradio_tabs.train.train()` は `use_jp_extra` が偽なら `train_ms.py`、真なら `train_ms_jp_extra.py` を選ぶ。両者は同じ CLI と保存規則を共有するが、モデル入力と補助損失が異なる。

| 項目 | `train_ms.py` | `train_ms_jp_extra.py` |
|---|---|---|
| テンプレート | `configs/config.json` | `configs/config_jp_extra.json` |
| BERT 入力 | ZH / JP / EN | JP のみ |
| 既定 `gin_channels` | 256 | 512 |
| duration discriminator | 既定で使用 | 既定で不使用 |
| WavLM discriminator | 学習 loop では不使用 | 既定で使用 |
| 追加 checkpoint | `DUR_<step>.pth` | `WD_<step>.pth`。設定時は `DUR_<step>.pth` も使用 |
| freeze | ZH/JP/EN BERT、style、decoder | JP BERT、style、decoder |

両者は generator、multi-period discriminator、任意の duration discriminator を学習する。generator loss は adversarial、feature matching、mel、duration、KL を合成する。JP-Extra で WavLM を有効にした場合は、`slm/wavlm-base-plus` を frozen feature extractor として使い、16 kHz へリサンプルした実音声・生成音声の hidden states に対する feature loss と WavLM discriminator loss を追加する。`slm/` 配下の設定が指すモデルを `transformers.AutoModel.from_pretrained()` で読み込めることが必要である。

モデル内部の encoder、flow、decoder、各 discriminator の構造は [core-inference.md](core-inference.md) の分界とする。

## device、precision、分散学習

### device 選択

`LOCAL_RANK` を `prepare_training_device()` に渡し、次の規則で device と DDP backend を決める。

- CUDA 利用可能: `cuda:<LOCAL_RANK>`。非 Windows かつ NCCL 利用可能なら `nccl`、それ以外は `gloo`。
- CUDA 利用不能: `cpu` と `gloo`。DDP の `device_ids` は指定しない。
- MPS を選択する経路はなく、CUDA 利用不能時は CPU となる。
- CPU では DataLoader worker を 0、pin memory を無効にする。CUDA では学習 DataLoader worker を 1、pin memory を有効にする。

### precision

両学習スクリプトは `train.bf16_run` が真のときだけ、device type に対応する `torch.amp.autocast` と `GradScaler` を有効にし、autocast dtype を `torch.bfloat16` とする。既定値は偽である。

`train.fp16_run` はハイパーパラメータとして受理されるが、学習 loop は参照しない。したがって fp16 autocast を選ぶ設定経路はない。推論用 safetensors も `is_half=False` の既定で保存され、学習時の generator parameter dtype を維持する。

CUDA では TF32 matmul を許可し、float32 matmul precision は `medium` である。両スクリプトは flash SDP と memory-efficient SDP を有効にし、通常版は math SDP も明示的に有効にする。

### DDP

学習スクリプトは単一 device の場合も `dist.init_process_group(init_method="env://")` を呼び、generator と discriminator を DDP で包む。`config.yml` の既定環境は `WORLD_SIZE=1`、`RANK=0`、`LOCAL_RANK=0` である。環境変数が既に存在する場合は上書きしないため、1 process / rank で `torchrun` 等から起動すれば multi-GPU の rank と world size を利用する。

- train sampler は rank と world size を受け取り、各 rank に batch を分配する。
- validation、TensorBoard、定期 checkpoint は rank 0 だけが実行する。
- 各 epoch の学習後に全 rank が scheduler を進める。
- 最終 epoch の checkpoint / safetensors 保存と Hugging Face upload も rank 0 だけが実行する。

## checkpoint、学習再開、成果物

### 再開用 checkpoint

再開用ファイルは `{dataset_path}/models` に保存する。

- `G_<global_step>.pth`: generator
- `D_<global_step>.pth`: multi-period discriminator
- `DUR_<global_step>.pth`: duration discriminator が有効な場合
- `WD_<global_step>.pth`: JP-Extra の WavLM discriminator が有効な場合

各 `.pth` は `model` state dict、optimizer state、`learning_rate`、`iteration` を保持する。`iteration` には保存時の epoch が入り、global step はファイル名から復元する。

`eval_interval` の倍数 step で rank 0 が評価と保存を行う。`--speedup` は TensorBoard と評価を省略するが、この定期保存は継続する。最終 epoch 終了時にも同形式を保存する。

`train_ms.keep_ckpts > 0` の場合、定期保存後に prefix ごとに指定数を残して古い `.pth` を削除する。`*_0.pth` は削除対象外である。最終 epoch の保存後にはこの cleanup を呼ばない。

### 再開判定

`models/` に `G_*.pth` が 1 件以上あれば再開と判定し、数字が最大の各 prefix の checkpoint を選ぶ。

- generator / discriminator と、有効な補助 discriminator の state を読み込む。
- `train.skip_optimizer=false` では optimizer state も復元する。真では model weight だけを復元する。
- learning-rate scheduler は checkpoint の epoch から再構成する。
- global step は最新 `G_<step>.pth` のファイル名から得る。
- 通常版は必要な `D_*.pth` / `DUR_*.pth` が欠けていると load が失敗する。
- JP-Extra 版は generator / discriminator の load 失敗を捕捉し、epoch 1、step 0 へ戻す。補助 discriminator は個別に load を試行する。

`G_*.pth` がなければ初期学習として、`G_0.safetensors`、`D_0.safetensors` と、有効な `DUR_0.safetensors` / `WD_0.safetensors` を `models/` から読み込む。読み込みのいずれかが失敗しても warning を記録し、epoch 1、step 0 から学習を開始する。

### 推論用資産

学習開始時、`--skip_default_style` がなければ `wavs/**/*.npy` の平均から `{assets_root}/<model_name>/style_vectors.npy` を作り、style metadata を反映した `config.json` を同じディレクトリへ書く。

定期保存時と最終 epoch に generator を次の名前で保存する。

```text
{assets_root}/<model_name>/<model_name>_e<epoch>_s<global_step>.safetensors
```

この safetensors は推論に不要な `enc_q` key を除外し、`iteration` tensor に epoch を格納する。`--assets_root` CLI 引数は parser に存在するが `config.out_dir` を更新しないため、実際の出力先は `configs/paths.yml` から構築された `{assets_root}/<model_name>` である。

`--repo_id` 指定時は、学習データと推論用資産を対応する Hugging Face repository path へ非同期 upload する。開始時には config の upload を先に試し、失敗時は学習を開始しない。

## SpeechMOS 評価

`speech_mos.py --model_name <model_name>` は `{assets_root}/<model_name>/*.safetensors` を列挙し、名前が `_s<step>.safetensors` で終わるモデルだけを評価する。

各モデルで固定の日本語テキストを推論し、SpeechMOS predictor の score と平均を計算する。結果は平均降順でログ表示し、次を出力する。

```text
mos_results/mos_<model_name>.csv
mos_results/mos_<model_name>.png
```

CSV は model file、step、各テキストの score、mean を持つ。PNG は step ごとの score と mean の折れ線を持ち、保存後に画面表示も行う。評価 device は `--device` で選択し、既定値は `cuda` である。

## 制約とエッジケース

- `preprocess_all.py` の再実行は既存 `models/` を退避後に作り直すため、単純な学習再開操作ではない。
- `slice.py` は既存 `raw/` を削除する一方、`resample.py` は既存 `wavs/` を掃除しない。
- train/validation list が参照する WAV、BERT cache、style vector のいずれかが欠けると学習データの読み込みは成立しない。
- style vector の NaN は前処理時に list から除外されるが、BERT load failure は data loader で warning 後も処理が続くため、有効な cache を前処理段で生成しておく必要がある。
- custom bucket sampler は境界外の長さを除外する。境界外も含める場合は `--not_use_custom_batch_sampler` を使う。
- `val.list` が空でも、rank 0 かつ `--speedup` なしでは evaluation loader を構築する。
- `speech_mos.py` は外部 SpeechMOS model を `torch.hub` から読み、推論用資産が既に揃っていることを前提とする。

## 関連テスト

- `tests/test_training_device.py`
  - CUDA / CPU の device、backend、DDP device IDs、pin memory を検証する。
  - device-aware な tensor/module 移動、DataLoader worker、AMP helper を検証する。
- `tests/test_train_torch_modernization.py`
  - spectrogram / mel の参照値互換を検証する。
  - legacy weight-norm checkpoint の読み込みを検証する。
  - 学習モデル構築時の torch deprecation warning と、学習スクリプトの deprecated SDP 呼び出し不在を検証する。
  - 全 `torch.load` 呼び出しが `weights_only` を明示することを検証する。
