# OpenAI 互換音声合成サーバー仕様

## 対象範囲

この文書は `style_bert_vits2.server` が提供する OpenAI 互換の音声合成 API
サーバーを対象とする。モデルの探索と推論そのものは
[core-inference.md](core-inference.md) を参照する。

このサーバーは本 fork が提供する唯一の API サーバーであり、OpenAI 互換の
`/v1/audio/speech` endpoint を公開する。

## セットアップとモデル配置

リポジトリの仮想環境へ server と PyTorch の optional dependency を導入する。

```console
uv sync --extra server --extra torch
```

モデルは `model_assets/` 直下へ次の構成で配置する。directory 名が API の model
identifier になる。

```text
model_assets/
└── <model-identifier>/
    ├── config.json
    ├── style_vectors.npy
    └── <model-file>.safetensors
```

`.pth`、`.pt`、`.onnx` も `TTSModelHolder` が対応する候補である。同じ model
directory に複数の候補がある場合は、更新時刻が最も新しいファイルを使う。

## エントリポイント

サーバーは package module または install 時に作られる console script から起動する。

```console
uv run python -m style_bert_vits2.server
sbv2-server
```

どちらも `style_bert_vits2.server.cli:main` を呼ぶ。console script はリポジトリルートの
`config.py` に依存せず、wheel を install した環境から実行できる。

### CLI 引数

| 引数 | 型・既定値 | 挙動 |
|---|---|---|
| `--host` | `str`、`127.0.0.1` | Uvicorn の bind address |
| `--port` | `int`、`1927` | Uvicorn の待受 port |
| `--dir` | path、`model_assets` | `TTSModelHolder` が探索する model root。相対 path は起動時の current working directory を基準にする |
| `--device` | `str`、自動 | 指定値を推論 device にする。省略時は CUDA が利用可能なら `cuda`、それ以外は `cpu` |
| `--language` | `JP` / `EN` / `ZH`、`JP` | request で `language` を省略した場合の既定言語 |
| `--no-preload-bert` | flag、既定 `False` | 日本語・英語 BERT の起動時 preload を無効にする |

### 起動シーケンス

1. CLI 引数を解析し、device を解決する。
2. `pyopenjtalk_worker` を起動する。
3. 既定辞書と書込先を実行環境に応じて解決し、ユーザー辞書を適用する。
4. `--no-preload-bert` がなければ、日本語・英語それぞれの PyTorch BERT model と
   tokenizer を preload する。
5. `TTSModelHolder` で model root を探索する。利用可能な model が 0 件なら起動を
   中止する。
6. `create_app()` で FastAPI application を作り、Uvicorn を起動する。

音声合成 model の重みは起動時にはロードしない。各 model への最初の合成 request
で `TTSModel.infer()` が遅延ロードし、以後の request で再利用する。

### ユーザー辞書 path

リポジトリ checkout では `dict_data/` が存在するため、既定 CSV、ユーザー JSON、
compiled dictionary を従来どおり同 directory から読み書きする。

wheel install 環境では、既定 CSV は package 内の
`style_bert_vits2/dict_data/default.csv` を使う。ユーザー JSON と compiled dictionary
は site-packages へ書き込まず、`~/.cache/style-bert-vits2/dict/` に保存する。

リポジトリ側と package 側のどちらにも既定 CSV がなければ warning を記録し、
辞書更新だけを skip してサーバー起動を続ける。

## Endpoint 一覧

| Method | Path | 成功応答 | 用途 |
|---|---|---|---|
| `GET` | `/health` | `200 application/json` | server process の生存確認 |
| `GET` | `/v1/models` | `200 application/json` | serve 中の model identifier 一覧 |
| `POST` | `/v1/audio/speech` | `200 audio/wav` または raw PCM stream | 音声合成 |

FastAPI の既定 endpoint `/openapi.json`、`/docs`、`/redoc` も公開される。

## `GET /health`

model のロード状態や GPU の状態は検査せず、process が request を処理できることだけを
示す。

```json
{"status": "ok"}
```

## `GET /v1/models`

`TTSModelHolder.model_names` の順序で model identifier を返す。

```json
{
  "object": "list",
  "data": [
    {"id": "model-a", "object": "model"},
    {"id": "model-b", "object": "model"}
  ]
}
```

model directory は名前順に探索されるため、request の `model` を省略した場合はこの
list の先頭 model が server 既定になる。

## `POST /v1/audio/speech`

request は JSON body で受け取る。不明 field は HTTP 422 で拒否し、文字列、数値、
真偽値の暗黙的な型変換は行わない。

### Request field

| field | 型 | 必須 | 既定値 | 挙動 |
|---|---|---:|---|---|
| `input` | `str` | yes | なし | 合成対象 text。空文字は 422 |
| `model` | `str` | no | server 既定 | `model_assets/` 内の directory 名 |
| `voice` | `str` | no | `None` | `config.json` の `spk2id` key。指定時は `speaker_id` より優先 |
| `response_format` | `"wav"` / `"pcm"` | no | `"wav"` | buffered / streaming の出力形式 |
| `speed` | 正の `float` | no | `1.0` | `TTSModel.infer(length=1 / speed)` へ写像する |
| `stream` | `bool` | no | `false` | true の場合は文単位 streaming |
| `language` | `"JP"` / `"EN"` / `"ZH"` | no | 起動時の既定言語 | テキスト処理言語 |
| `speaker_id` | `int` | no | `0` | `voice` がない場合の speaker ID |
| `style` | `str` | no | `"Neutral"` | model の `style2id` key |
| `style_weight` | `float` | no | `1.0` | style vector の適用強度 |
| `sdp_ratio` | `float` | no | `0.2` | SDP / DP の混合比 |
| `noise` | `float` | no | `0.6` | latent sampling の noise scale |
| `noise_w` | `float` | no | `0.8` | SDP の noise scale |
| `assist_text` | `str` | no | `None` | BERT 特徴量生成の補助 text |
| `assist_text_weight` | `float` | no | `1.0` | 補助 text の適用強度 |

### model、speaker、style の選択

1. `model` がなければ model list の先頭を使う。
2. 未知の `model` は HTTP 404 を返す。
3. `voice` があれば `spk2id` から speaker ID を解決する。
4. `voice` がなければ `speaker_id` を使い、省略時は `0` とする。
5. 未知の `voice`、`speaker_id`、`style` は HTTP 400 を返す。

model object は model ごとに cache する。同じ model に対する `infer()` は lock で直列化
し、単一 model object への同時推論を行わない。

### Buffered WAV

`stream: false` と `response_format: "wav"` の組だけを受容する。`input` 全体を 1 回の
`TTSModel.infer(line_split=False)` へ渡し、返された signed int16 mono PCM を WAV
container に格納する。

- Status: `200`
- Content-Type: `audio/wav`
- Channel: mono
- Sample width: 16 bit
- Sample rate: model の `config.json` に依存し、WAV header が自己記述する

### Streaming PCM

`stream: true` と `response_format: "pcm"` の組だけを受容する。WAV header や終了
sentinel は付けず、HTTP body の EOF で終了する。

- Body: raw signed int16 little-endian mono PCM
- `X-Sample-Rate`: model が返した sample rate
- `X-Channels`: `1`
- `X-Bit-Depth`: `16`

text は `. ! ? 。 ！ ？` と改行で文単位に分割する。最初の文を合成して sample rate
と最初の PCM chunk を確定した後に response を開始し、残りの文を順番に合成して
yield する。したがって first-byte latency は少なくとも最初の 1 文の推論時間になり、
後続 chunk の latency は各文の長さに依存する。

HTTP transport が application の yield 単位をさらに分割または結合する場合があるため、
client は HTTP chunk 境界を音声 frame や文の境界として解釈してはならない。

### 出力形式の組合せ

| `stream` | `response_format` | 結果 |
|---:|---|---|
| `false` | `"wav"` | buffered PCM16 WAV |
| `false` | `"pcm"` | HTTP 400 |
| `true` | `"wav"` | HTTP 400 |
| `true` | `"pcm"` | raw PCM streaming |

## エラー応答

エラー本文は FastAPI の `{"detail": ...}` 形式を使う。

| 条件 | Status |
|---|---:|
| JSON schema 違反、不明 field、欠落 field | 422 |
| streaming / format の不正な組合せ | 400 |
| 未知の voice、speaker ID、style、推論入力エラー | 400 |
| 未知の model | 404 |
| 未知 path | 404 |
| method 不一致 | 405 |

streaming response の開始後に後続文の推論が失敗した場合、すでに HTTP status と header
を送信済みのため JSON error には変換できず、body の途中終了として現れる。

## 契約確認手順

以下は既定 port `1927` で起動した server に対する例である。

### model list

```console
curl -sS http://127.0.0.1:1927/v1/models
```

### buffered WAV

```console
curl -sS \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello from Style-Bert-VITS2.","language":"EN","response_format":"wav"}' \
  http://127.0.0.1:1927/v1/audio/speech \
  --output speech.wav
```

WAV header は標準 library で確認できる。

```console
python -c "import wave; f=wave.open('speech.wav'); print(f.getframerate(), f.getnchannels(), f.getsampwidth())"
```

### streaming PCM

```console
curl -N -D stream.headers \
  -H 'Content-Type: application/json' \
  -d '{"input":"First sentence. Second sentence.","language":"EN","stream":true,"response_format":"pcm"}' \
  http://127.0.0.1:1927/v1/audio/speech \
  --output speech.pcm
```

`stream.headers` の `X-Sample-Rate`、`X-Channels`、`X-Bit-Depth` と、`speech.pcm` の
byte 数が 2 の倍数であることを確認する。

### validation error

```console
curl -sS \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello.","unsupported":true}' \
  http://127.0.0.1:1927/v1/audio/speech
```

## 関連テスト

- `tests/test_server_openai_schemas.py`: field の既定値、strict validation、不明 field。
- `tests/test_server_openai_api.py`: ASGI 経由の endpoint、WAV / PCM、文分割、header、
  model / speaker / style 選択、エラー形。
- `tests/test_server_openai_cli.py`: config 非依存の runtime 初期化、辞書 path 解決、
  BERT preload、CLI 値と Uvicorn 起動。
- `tests/test_server_openai_gpu.py`: 明示された実 model による CUDA 推論と WAV 契約。
