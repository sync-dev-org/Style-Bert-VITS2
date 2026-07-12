# FastAPI サーバー仕様

## 対象範囲

この文書は `server_fastapi.py` が提供する汎用音声合成 API サーバーと、
`config.py` の `Server_config` による設定解決を対象とする。

音声合成ライブラリ内部の推論処理は [core-inference.md](core-inference.md)、
音声合成エディター向けの `server_editor.py` は
[editor-server.md](editor-server.md) を参照する。

## エントリポイント

サーバーは次のコマンドで起動する。

```console
python server_fastapi.py [--cpu] [--dir MODEL_DIR] [--port PORT] [--preload_onnx_bert]
```

モジュールを import しただけではサーバーは起動しない。アプリケーションを組み込む場合は
`create_app()` で `FastAPI` インスタンスを生成できる。

### CLI 引数

| 引数 | 型・既定値 | 挙動 |
|---|---|---|
| `--cpu` | flag、既定 `False` | 指定時は推論 device を `cpu` に固定する |
| `--dir`, `-d` | `str`、既定 `config.assets_root` | `TTSModelHolder` が探索するモデルルート |
| `--port` | `int`、既定 `None` | 指定時は `server.port` より優先して待受 port にする |
| `--preload_onnx_bert` | flag、既定 `False` | 日本語 ONNX BERT model と tokenizer も起動時に事前ロードする |

`--cpu` を指定しない場合、CUDA が利用可能なら `cuda`、それ以外なら `cpu` を選ぶ。
`server.device` の値はこの device 選択には使われない。

## 起動シーケンス

`run_server()` は次の順で初期化する。

1. `get_config()` で path 設定と `config.yml` を読む。
2. `pyopenjtalk_worker` を起動し、`dict_data/` のユーザー辞書を適用する。
3. CLI 引数を解析し、device と port を決定する。
4. 日本語の PyTorch BERT model と tokenizer を事前ロードする。
5. `--preload_onnx_bert` 指定時は、日本語の ONNX BERT model と tokenizer もロードする。
6. モデルルートと device から `TTSModelHolder` を構築する。
7. 利用可能なモデルが 0 件なら、終了 code 1 でプロセスを終了する。
8. `load_models()` で各モデルの `TTSModel` オブジェクトを構築する。
9. `create_app()` で API を構成する。
10. Uvicorn を `host="0.0.0.0"`、解決済み port、`log_level="warning"` で起動する。

起動ログに表示する URL は `127.0.0.1` だが、実際の bind address は全 interface を表す
`0.0.0.0` である。Uvicorn の worker 数、接続数、keep-alive などは明示指定せず、
Uvicorn の既定値を使う。

## 設定

### `server` 節

`config.yml` の `server` mapping は `Server_config.from_dict()` へ展開される。

| 項目 | 型 | `Server_config` の既定値 | `default_config.yml` | サーバーでの用途 |
|---|---|---:|---:|---|
| `port` | `int` | `5100` | `5100` | CLI で port を指定しない場合の待受 port |
| `device` | `str` | `"cuda"` | `"cuda"` | 設定オブジェクトには保持されるが、`server_fastapi.py` の device 選択では参照しない |
| `limit` | `int` | `100` | `100` | `/voice` の `text` 最大文字数 |
| `language` | `str` | `"JP"` | `"JP"` | `/voice` の `language` 既定値 |
| `origins` | `list[str]` | `["*"]` | `["*"]` | CORS 許可 origin |

`config.yml` が存在しない場合は `default_config.yml` をコピーしてから読む。
`server` 節の形が不正で `TypeError` または `KeyError` になった場合も、`config.yml` 全体を
`default_config.yml` で置き換えて再読込する。

`limit` が 1 未満なら文字数上限を無効にし、`create_app()` には `None` を渡す。
`create_app()` 自体も、直接渡された `limit` が 1 未満なら `None` として扱う。

### CORS

`allow_origins` が truthy の場合だけ `CORSMiddleware` を追加し、次の値を設定する。

- `allow_origins`: 設定値
- `allow_credentials=True`
- `allow_methods=["*"]`
- `allow_headers=["*"]`

空 list または `None` なら CORS middleware を追加しない。`origins` キー自体を省略した場合は
`Server_config` の既定値 `["*"]` が使われるため、CORS は無効にならない。

## モデル探索とロード

### `TTSModelHolder` による探索

`TTSModelHolder` はモデルルート直下の directory を名前順に走査する。`.` で始まる
directory は除外する。各 directory では `.pth`、`.pt`、`.safetensors`、`.onnx` の
モデルファイルを更新日時の新しい順に並べる。モデルファイルがない directory、または
`config.json` がない directory は対象外となる。

対象 directory の並び順が API の `model_id` になる。directory 名は `model_name` になる。
各 directory の `config.json` から style 名と speaker 名を読み、`models_info` に保持する。

### API 用モデル状態

`load_models()` は各 `model_name` について、更新日時が最も新しいモデルファイル
`model_files_dict[model_name][0]` を選び、次のファイルから `TTSModel` を構築する。

- model: 選択した `.pth`、`.pt`、`.safetensors`、または `.onnx`
- config: `<model root>/<model_name>/config.json`
- style vector: `<model root>/<model_name>/style_vectors.npy`

この段階では config と style vector を読み込むが、PyTorch model weight または ONNX session は
ロードしない。`/voice` がその `TTSModel` で初めて `infer()` を呼んだ時点で遅延ロードする。
ロード後の model/session は同じ `TTSModel` に保持され、以後の推論で再利用する。

`load_models()` は `TTSModelHolder.get_model()` を経由せず、holder から `root_dir` と `device` を
使って `TTSModel` を直接構築する。このとき `TTSModelHolder.onnx_providers` は渡さないため、
選択されたファイルが ONNX model の場合は `TTSModel` の既定 provider
`CPUExecutionProvider` を使う。

`create_app()` は受け取った `loaded_models` をアプリ内状態として保持する。
`POST /models/refresh` は `TTSModelHolder.refresh()` でファイル一覧を再探索し、
`load_models()` の結果でアプリ内状態を丸ごと置き換える。これにより既存の `TTSModel`
オブジェクトと、そのオブジェクトが保持していたロード済み weight/session は API の選択対象から外れる。

## API 共通仕様

FastAPI の既定機能により OpenAPI schema は `/openapi.json`、Swagger UI は `/docs`、
ReDoc は `/redoc` で公開する。アプリケーション固有 endpoint に認証 dependency、認証
middleware、API key 検証はない。

parameter の型変換または `Query` 制約に違反すると FastAPI の validation error として
HTTP 422 を返す。モデル、speaker、style、ファイル path の独自検証エラーも HTTP 422 で、
本文は次の形になる。

```json
{
  "detail": [
    {
      "type": "invalid_params",
      "msg": "検証メッセージ",
      "loc": ["query", "parameter_name"]
    }
  ]
}
```

推論失敗やファイル読込失敗など、明示的に捕捉していない例外には API 独自の error 変換を
行わない。例外は `/voice` の実効テキスト空エラー (`EmptyEffectiveTextError`) で、
これだけは HTTP 400 へ変換する (「`GET|POST /voice`」節を参照)。

## Endpoint 一覧

| Method | Path | 成功応答 | 用途 |
|---|---|---|---|
| `GET`, `POST` | `/voice` | `200 audio/wav` | テキストから音声を生成する |
| `POST` | `/g2p` | `200 application/json` | 日本語 text を読みと tone の列へ変換する |
| `GET` | `/models/info` | `200 application/json` | API 内のモデル情報を取得する |
| `POST` | `/models/refresh` | `200 application/json` | モデル一覧と API 内モデル状態を再構築する |
| `GET` | `/status` | `200 application/json` | CPU、memory、GPU、利用可能 device の状態を取得する |
| `GET` | `/tools/get_audio` | `200 audio/wav` | サーバーから参照できるローカル WAV ファイルを返す |

## `GET|POST /voice`

音声合成を行う。`POST` の場合も request body は使わず、すべて query parameter で受け取る。
`GET` と `POST` の処理および応答は同じだが、`GET` の利用時は非推奨 warning を log に出す。

### Query parameter

| 名前 | 型 | 必須 | 既定値 | 制約・意味 |
|---|---|---|---|---|
| `text` | `str` | yes | なし | 1 文字以上。`server.limit >= 1` の場合はその文字数以下 |
| `encoding` | `str` | no | `None` | 指定時は validation 後に `urllib.parse.unquote(text, encoding=encoding)` を適用する |
| `model_name` | `str` | no | `None` | モデル directory 名。指定時は `model_id` より優先する |
| `model_id` | `int` | no | `0` | `/models/info` の top-level key に対応する index |
| `speaker_name` | `str` | no | `None` | `spk2id` の key。指定時は `speaker_id` より優先する |
| `speaker_id` | `int` | no | `0` | model の `id2spk` に存在する speaker ID |
| `sdp_ratio` | `float` | no | `0.2` | SDP/DP の混合比 |
| `noise` | `float` | no | `0.6` | サンプルノイズ量 |
| `noisew` | `float` | no | `0.8` | SDP ノイズ量。`TTSModel.infer()` の `noise_w` に渡す |
| `length` | `float` | no | `1.0` | 大きいほど長く遅い音声になる話速係数 |
| `language` | enum | no | `server.language` | `JP`、`EN`、`ZH` |
| `auto_split` | `bool` | no | `true` | text を改行単位で分割して生成する |
| `split_interval` | `float` | no | `0.5` | 分割した音声間に挿入する無音秒数 |
| `assist_text` | `str` | no | `None` | 声音・感情の補助 text |
| `assist_text_weight` | `float` | no | `1.0` | `assist_text` の適用強度 |
| `style` | `str` | no | `"Neutral"` | model の `style2id` に存在する style 名 |
| `style_weight` | `float` | no | `1.0` | style vector の適用強度 |
| `reference_audio_path` | `str` | no | `None` | サーバープロセスから参照できる style 用音声ファイル path |

`text` 以外の数値 parameter には API 層の上限・下限制約を設けない。

### モデルと speaker の選択

1. `model_id >= len(model_holder.model_names)` なら 422 を返す。
2. `model_name` が指定されていれば `model_holder.models_info` から同名 entry を探し、
   見つからない場合または複数見つかった場合は 422 を返す。
3. `model_name` が一意なら、その index で `model_id` を上書きする。
4. `speaker_name` がなければ `speaker_id` が model の `id2spk` に存在することを検証する。
5. `speaker_name` があれば `spk2id` に存在することを検証し、対応する ID で
   `speaker_id` を上書きする。
6. `style` が model の `style2id` に存在することを検証する。

`model_id` には 0 以上という Query 制約がない。負数は上限検証を通過し、Python の負 index として
`loaded_models` を参照する。

### 推論と応答

選択した `TTSModel.infer()` に parameter を渡す。`assist_text` が空または未指定なら
`use_assist_text=False`、それ以外なら `True` とする。推論が返した sampling rate と 16-bit PCM
音声を `scipy.io.wavfile.write()` で WAV byte 列にし、`Content-Type: audio/wav` で返す。

実効テキスト (改行で分割し空文字要素を除いた行の集合) が空の `text` (改行のみ等) では、
推論層の `EmptyEffectiveTextError` を HTTP 400 へ変換する。本文は 422 の独自検証エラーと
同じ構造 (`type: invalid_params`、`loc: ["query", "text"]`) を持つ。

## `POST /g2p`

必須 query parameter `text: str` を受け取る。request body は使わない。
`normalize_text(text)` を適用した後、`g2kata_tone()` で読みと tone の組へ変換する。
応答は JSON array で、各要素は `[読み文字列, tone整数]` である。

## `GET /models/info`

API 内の `loaded_models` を順に列挙し、文字列化した model ID を key とする JSON object を返す。
モデルがない場合は空 object `{}` になる。

各 model entry は次の field を持つ。

| field | 内容 |
|---|---|
| `config_path` | `config.json` の path |
| `model_path` | 選択された model weight の path |
| `device` | `TTSModel` の PyTorch device |
| `spk2id` | speaker 名から ID への mapping |
| `id2spk` | speaker ID から名前への mapping |
| `style2id` | style 名から ID への mapping |

## `POST /models/refresh`

parameter はない。`TTSModelHolder.refresh()` でモデル directory を再走査し、
`load_models()` で API 内の `loaded_models` を再構築する。成功時は更新後の
`GET /models/info` と同じ JSON object を返す。

refresh 後の model ID は、その時点の対象 directory の名前順に再採番される。

## `GET /status`

parameter はない。`psutil.cpu_percent(interval=1)` により 1 秒間隔で CPU 使用率を測定し、
memory 情報、PyTorch が認識する CUDA device、GPUtil が認識する GPU 情報と合わせて返す。

```json
{
  "devices": ["cpu", "cuda:0"],
  "cpu_percent": 0.0,
  "memory_total": 0,
  "memory_available": 0,
  "memory_used": 0,
  "memory_percent": 0.0,
  "gpu": [
    {
      "gpu_id": 0,
      "gpu_load": 0.0,
      "gpu_memory": {"total": 0.0, "used": 0.0, "free": 0.0}
    }
  ]
}
```

`devices` は常に `"cpu"` を含み、`torch.cuda.device_count()` の各 index を
`"cuda:<index>"` として追加する。GPU がない場合の `gpu` は空 array になる。

## `GET /tools/get_audio`

必須 query parameter `path: str` を受け取る。path が既存 file でない場合、または大文字小文字を
無視した suffix が `.wav` でない場合は 422 を返す。検証を通過した file を
`FileResponse` により `Content-Type: audio/wav` で返す。

path の許可 directory 制約や path 正規化による隔離は行わない。

## アクセス制御と実行上の制約

- 認証・認可、API key、rate limit はない。
- Uvicorn は `0.0.0.0` に bind するため、network 側で制限されていなければ全 endpoint が外部から到達可能になる。
- CORS は browser の cross-origin 制御であり、endpoint への認証・アクセス制御ではない。
- `/tools/get_audio` はサーバープロセスが読める任意の `.wav` path を受け取る。
- `/voice` の `reference_audio_path` もサーバープロセス側の path として推論処理へ渡す。
- `/models/info` は model と config の path を応答に含める。
- request の同時実行数を制御する semaphore、queue、lock は設けない。
- `/voice` は `async def` だが、内部の `model.infer()` は同期呼び出しで実行する。
- API 独自の request body size、音声 file size、生成音声 size の上限はない。
- API 層の入力長制限は `/voice` の `text` に対する `server.limit` だけである。

## 制約とエッジケース

- `model_name`、`speaker_name` は対応する数値 ID より優先する。
- `style` は既定値を含め、選択した model の `style2id` に存在しなければならない。
- `reference_audio_path` を指定すると、model の style 名ではなく参照音声から style vector を生成するが、
  API 層では引き続き `style` の存在検証を先に行う。
- JP-Extra model に `EN` または `ZH` を指定した場合など、推論層で生じる `ValueError` は
  API 層で 422 へ変換しない。
- `auto_split=true` の場合、空行を除いた改行単位で推論し、各音声の間に
  `split_interval` 秒の無音を挿入する。改行だけの text など実効テキストが空になる入力は
  推論層で `EmptyEffectiveTextError` になり、API 層が HTTP 400 へ変換する。
- `/models/refresh` と `/voice` の同時実行を調停する lock はない。
- `TTSModelHolder` の探索は `style_vectors.npy` の存在を検証しない。欠落または不正な
  config/style vector は `load_models()` による `TTSModel` 構築時の未捕捉例外になる。

## 関連テスト

- `tests/test_server_fastapi_api.py`
  - model asset なしの `/status`、`/models/info`、`/g2p`
  - model asset がある場合の `/voice` WAV 応答
  - 実効テキスト空の `/voice` に対する HTTP 400 変換
- `tests/test_server_fastapi_config.py`
  - `Server_config.port` の既定値と明示値
  - `default_config.yml` の port
  - CLI port と設定 port の優先順位
