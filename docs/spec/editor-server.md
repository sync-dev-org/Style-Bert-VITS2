# エディターサーバー仕様

## 対象範囲

本書は `server_editor.py` が提供する次の挙動を定義する。

- Style-Bert-VITS2-Editor 向け HTTP API
- エディターフロントエンドの取得と静的ファイル配信
- `dict_data/` に保存するユーザー辞書の CRUD と pyopenjtalk への反映
- コマンドライン引数、起動時初期化、CORS

日本語の正規化および G2P の内部処理は [nlp.md](nlp.md)、モデルのロードと推論の内部処理は [core-inference.md](core-inference.md) が扱う。本書では、それらをエディターサーバーがどのような入出力で呼び出すかを扱う。

## エントリポイントと起動時初期化

標準の起動方法は次のとおりである。

```bash
python server_editor.py [options]
```

モジュールのトップレベルでは、HTTP サーバーを起動する前に次の処理を順に行う。

1. pyopenjtalk worker を TCP port `7861` で初期化する。既存 worker に接続できなければ別プロセスを起動し、最大 20 回、0.5 秒間隔で接続を試みる。
2. `dict_data/` の辞書をコンパイルし、pyopenjtalk worker に適用する。
3. CLI 引数を解析する。
4. 日本語用 BERT モデルと tokenizer を指定 device へ事前ロードする。
5. `--preload_onnx_bert` 指定時は、日本語用 ONNX BERT モデルと tokenizer も事前ロードする。
6. `TTSModelHolder` を生成してモデル一覧を走査する。ONNX 音声合成モデルは一覧から除外する。
7. 利用可能な音声合成モデルが 1 件もなければ終了コード `1` で終了する。
8. FastAPI application、CORS middleware、`/api` router を構築する。

`python server_editor.py` として直接実行した場合だけ、フロントエンドの更新確認、root path への静的配信 mount、ブラウザー起動、Uvicorn 起動を行う。`server_editor:app` のように ASGI application を import した場合も 1〜8 は実行されるが、静的配信の mount と Uvicorn 起動は行われない。

## CLI 引数

| 引数 | 型・既定値 | 挙動 |
|---|---|---|
| `--model_dir` | path、paths 設定の `assets_root` | 音声合成モデルを探索する root directory |
| `--device` | string、`cuda` | BERT と音声合成に使用する device。`cuda` 指定時に CUDA が利用不能なら `cpu` へ切り替える |
| `--port` | integer、`8000` | Uvicorn の listen port と `--inbrowser` の URL に使用する |
| `--inbrowser` | flag、無効 | Uvicorn 起動直前に `http://localhost:<port>` を既定ブラウザーで開く |
| `--line_length` | integer、未指定 | 指定時、単発合成および複数行合成の各 `text` に文字数上限を設ける |
| `--line_count` | integer、未指定 | 指定時、複数行合成の `lines` 件数に上限を設ける |
| `--skip_static_files` | flag、無効 | フロントエンド release の更新確認と download を省略する。既存 `static/` の mount 自体は省略しない |
| `--preload_onnx_bert` | flag、無効 | 日本語用 ONNX BERT model/tokenizer を起動時に事前ロードする |

Uvicorn は host `0.0.0.0`、指定 port で起動する。`--skip_static_files` を指定しても `static/` が存在しなければ、静的配信の mount 時に起動が失敗する。

## CORS

許可 origin は次の 4 件に固定される。

- `http://localhost:3000`
- `http://localhost:8000`
- `http://127.0.0.1:3000`
- `http://127.0.0.1:8000`

credential を許可し、method と header はすべて許可する。CLI の `--port` を変更しても許可 origin は変化しない。

## HTTP API の共通契約

application 固有の endpoint はすべて `/api` prefix を持つ。音声合成 endpoint を除き、request と response は JSON である。FastAPI が提供する OpenAPI endpoint (`/openapi.json`) と対話 UI (`/docs`、`/redoc`) も有効である。

request body の必須 field 欠落、型不一致、未定義の `language` など、FastAPI/Pydantic による request validation failure は HTTP `422` となる。明示的に処理していない実行時例外は HTTP `500` となる。

### Endpoint 一覧

| Method | Path | 入力 | 成功 response |
|---|---|---|---|
| `GET` | `/api/version` | なし | version string、`200` |
| `POST` | `/api/normalize` | `TextRequest` | 正規化後の string、`200` |
| `POST` | `/api/g2p` | `TextRequest` | `MoraTone[]`、`200` |
| `GET` | `/api/models_info` | なし | `TTSModelInfo[]`、`200` |
| `POST` | `/api/synthesis` | `SynthesisRequest` | WAV binary (`audio/wav`)、`200` |
| `POST` | `/api/multi_synthesis` | `MultiSynthesisRequest` | WAV binary (`audio/wav`)、`200` |
| `GET` | `/api/user_dict` | なし | UUID を key とするユーザー辞書 object、`200` |
| `POST` | `/api/user_dict_word` | `UserDictWordRequest` | `{"uuid": string}`、`201` |
| `PUT` | `/api/user_dict_word/{uuid}` | path UUID string、`UserDictWordRequest` | `{"uuid": string}`、`200` |
| `DELETE` | `/api/user_dict_word/{uuid}` | path UUID string | `{"uuid": string}`、`200` |

## テキスト処理 API

### `TextRequest`

```json
{
  "text": "読み上げる文字列"
}
```

### `MoraTone`

```json
{
  "mora": "カ",
  "tone": 0
}
```

`mora` はカタカナのモーラまたは句読点、`tone` はその音高値である。

### `POST /api/normalize`

`text` を日本語 normalizer に渡し、その返り値を JSON string として返す。この endpoint 自身は例外を変換しない。

### `POST /api/g2p`

`text` を先に正規化し、正規化結果を日本語 G2P へ渡す。得られたモーラまたは句読点と tone の組を `MoraTone[]` として返す。

正規化または G2P が例外を送出した場合は HTTP `400` とし、response body の `detail` に元の入力 text と例外文字列を含める。

## モデル情報 API

### `GET /api/models_info`

起動時に `--model_dir` から走査した model 情報を配列で返す。各要素は次の形を持つ。

```json
{
  "name": "model-name",
  "files": ["model file path"],
  "styles": ["Neutral"],
  "speakers": ["speaker-name"]
}
```

音声合成 model file は `.pth`、`.pt`、`.safetensors` が対象であり、ONNX file はエディターサーバーでは除外される。model directory または file の情報は起動後に自動再走査しない。

## 音声合成 API

### `SynthesisRequest`

| Field | 型 | 必須・既定値 | 用途 |
|---|---|---|---|
| `model` | string | 必須 | model directory 名 |
| `modelFile` | string | 必須 | `models_info[].files` に含まれる model file path |
| `text` | string | 必須 | 推論へ渡す text |
| `moraToneList` | `MoraTone[]` | 必須 | 編集済みのモーラ・tone 列 |
| `style` | string | `Neutral` | style 名 |
| `styleWeight` | float | `1.0` | style の適用強度 |
| `assistText` | string | 空文字列 | 補助 text。空でなければ使用する |
| `assistTextWeight` | float | `1.0` | 補助 text の適用強度 |
| `speed` | float | `1.0` | 推論の length に `1 / speed` を渡す |
| `noise` | float | `0.6` | DP noise |
| `noisew` | float | `0.8` | SDP noise |
| `sdpRatio` | float | `0.2` | SDP/DP の混合比 |
| `language` | `JP` / `EN` / `ZH` | `JP` | 推論 language |
| `silenceAfter` | float | `0.5` | 複数行合成で、この行の後へ挿入する無音秒数 |
| `pitchScale` | float | `1.0` | 出力 pitch の倍率 |
| `intonationScale` | float | `1.0` | 出力 intonation 幅の倍率 |
| `speaker` | string または null | null | 合成に使用する speaker 名。単発・複数行合成の両方で使用する |

数値 field には schema 上の上下限を設けない。`speed` が `0` の場合の除算エラーや、推論が受理しない値による例外は HTTP `500` となる。

### G2P 編集から単発合成まで

エディターは次の経路で G2P 結果を編集して合成へ渡せる。

1. `/api/g2p` で正規化済み text に対応する `MoraTone[]` を取得する。
2. client 側で `mora` または `tone` を編集する。
3. 元の `text` と編集後の配列を `/api/synthesis` の `moraToneList` へ渡す。
4. server は `moraToneList` を phone/tone 列へ変換し、tone 列を `given_tone` として推論へ渡す。

server は `text` と `moraToneList` の対応関係、長さ、tone 値の範囲を事前検証しない。未知の `mora` や推論入力との不整合による例外は HTTP `500` となる。

### `POST /api/synthesis`

処理順は次のとおりである。

1. `--line_length` 指定時は Python の `len(text)` で上限を検査する。超過時は HTTP `400`。
2. `model` と `modelFile` の組から model を取得する。取得失敗は HTTP `500`。
3. `moraToneList` を phone/tone 列へ変換する。
4. `speaker` が null なら speaker ID `0`、指定時は model の `spk2id` による ID を使用する。未知の speaker は HTTP `400`。
5. 改行分割を無効にして推論する。
6. 推論結果を WAV に encode し、`audio/wav` で返す。

`silenceAfter` は単発合成では使用しない。

### `POST /api/multi_synthesis`

request body は `{"lines": SynthesisRequest[]}` である。各行を順番に単発推論し、最終行以外の後ろへその行の `silenceAfter` 秒分の `int16` zero data を挿入してから、全 data を連結して 1 個の WAV を返す。

- `--line_count` を超えると HTTP `400`。
- 各行の `text` が `--line_length` を超えると HTTP `400`。
- model 取得失敗は HTTP `500`。
- `lines` が空の場合は音声配列を連結できず HTTP `500`。
- 各行の sample rate が同一かどうかは検証しない。WAV の sample rate には最終行の値を使用する。
- 各行の `speaker` は単発合成と同じ規則で行ごとに解決する。null なら speaker ID `0`、指定時は model の `spk2id` による ID を使用し、未知の speaker は HTTP `400` とする。
- `pitchScale` と `intonationScale` は各行へ適用する。

## ユーザー辞書 API

### `UserDictWordRequest`

```json
{
  "surface": "表層形",
  "pronunciation": "ヨミ",
  "accent_type": 1,
  "priority": 5
}
```

- `surface`、`pronunciation`、`accent_type` は必須である。
- `priority` の既定値は `5`、許容範囲は `0` から `10` である。範囲外は HTTP `422`。
- word type は API から指定できず、常に固有名詞として生成する。
- `surface` 内の ASCII printable character は全角へ変換して保存する。
- `pronunciation` はカタカナと長音記号だけを受理し、捨て仮名の並びにも制約を設ける。
- `accent_type` は `0` から pronunciation のモーラ数までを受理する。`0` はアクセント核なしを表す。

pronunciation または accent の word model validation failure は endpoint で HTTP error に変換しないため HTTP `500` となる。

### `GET /api/user_dict`

`dict_data/user_dict.json` が存在しなければ空 object `{}` を返す。存在する場合は UUID string を key、`UserDictWord` を value とする object を返す。保存形式の `cost` は response では `priority` に変換される。

### `POST /api/user_dict_word`

新しい UUID を生成し、word を `user_dict.json` に追加する。辞書を再コンパイルして pyopenjtalk に適用した後、UUID を HTTP `201` で返す。

### `PUT /api/user_dict_word/{uuid}`

既存 UUID の word 全体を request の内容で置き換える。UUID が存在しなければ HTTP `422`。成功時は辞書を再コンパイルして UUID を HTTP `200` で返す。

### `DELETE /api/user_dict_word/{uuid}`

既存 UUID の word を削除する。UUID が存在しなければ HTTP `422`。成功時は辞書を再コンパイルして UUID を HTTP `200` で返す。

add、update、delete の各 helper は保存後に `update_dict()` を実行する。endpoint 側は helper を呼ぶだけで追加の `update_dict()` は行わず、成功する 1 回の変更 request につき辞書のコンパイルと適用は 1 回である。

## `dict_data/` の保存・適用契約

| Path | 役割 | Git tracking |
|---|---|---|
| `dict_data/default.csv` | 配布される既定単語を格納する OpenJTalk CSV | tracked |
| `dict_data/user_dict.json` | API で編集した word を UUID key で永続化する JSON | ignored |
| `dict_data/user.dic` | default と user dictionary を統合したコンパイル済み辞書 | ignored |

`user_dict.json` の各 word は `priority` ではなく、品詞の context ID と priority に対応する `cost` を保存する。読み出し時に `cost` を最も近い `priority` へ戻す。過去形式で `context_id` がない word は固有名詞の context ID を補う。

辞書更新時は次の処理を行う。

1. `default.csv` を読み、末尾に newline がなければ補う。
2. `user_dict.json` の全 word を OpenJTalk CSV 行へ変換して連結する。
3. UUID を含む一時 CSV と一時 compiled file を `dict_data/` に作る。
4. pyopenjtalk worker の `mecab_dict_index` でコンパイルする。
5. worker 上の現行 user dictionary を unset する。
6. 一時 compiled file を `user.dic` へ置き換える。
7. `user.dic` を worker の global OpenJTalk instance へ設定する。
8. 成否にかかわらず一時 file を削除する。

`default.csv` がない場合は warning を標準 error へ出して更新を終了する。コンパイルまたは適用に失敗した場合は例外を再送出する。辞書の読み書きとコンパイルには排他制御を設けていないため、同時 mutation request の直列化は保証しない。

## フロントエンドの取得と静的配信

直接実行時、`--skip_static_files` がなければ GitHub Releases API から `litagin02/Style-Bert-VITS2-Editor` の latest release を取得し、asset 名が厳密に `out.zip` と一致する file を使用する。

version 管理には release の `published_at` を使う。

- `static/last_download.txt` がなければ download 対象とする。
- file があれば保存済み日時と latest release の `published_at` を timezone 付き日時として比較し、latest の方が新しい場合だけ download する。
- download 成功後、latest release の `published_at` を `last_download.txt` へ保存する。
- release 情報を取得できない場合、該当 asset がない場合、または更新がない場合は既存 static file を維持して起動を続ける。

更新時は既存 `static/` を directory ごと削除して作り直し、ZIP を展開する。展開結果が単一 directory だけなら、その directory の直下を `static/` 直下へ移動する。`static/index.html` がなくても warning のみで処理を続ける。asset download、ZIP decode、展開時の例外は捕捉せず起動を失敗させる。

API router の登録後、`static/` を root `/` に `html=True` で mount する。これにより `index.html` と frontend assets を同じ server から配信し、API route、OpenAPI route、documentation route は static mount より先に解決される。`static/` は Git tracking の対象外である。

## エラー応答

明示的な主要 error は次のとおりである。

| 条件 | Status | `detail` の概要 |
|---|---|---|
| request schema validation failure | `422` | FastAPI/Pydantic の validation error list |
| G2P またはその前段の正規化失敗 | `400` | 入力 text と例外文字列 |
| `--line_length` 超過 | `400` | 設定した最大文字数 |
| `--line_count` 超過 | `400` | 設定した最大行数 |
| 単発・複数行合成で未知の speaker | `400` | speaker 名と model の `spk2id` |
| model 名または model file の取得失敗 | `500` | model 名、file、例外文字列 |
| 辞書の priority が `0..10` 外 | `422` | 優先度が無効 |
| PUT の UUID が辞書にない | `422` | UUID に該当する word がない |
| DELETE の UUID が辞書にない | `422` | ID に該当する word がない |
| その他の未処理例外 | `500` | FastAPI の Internal Server Error |

## 制約とエッジケース

- API に authentication または authorization はない。
- 起動時の model 一覧と CORS origin は server 実行中に更新しない。
- G2P は日本語処理に固定される一方、合成 request の `language` schema は `JP`、`EN`、`ZH` を受理する。model 側がその言語を受理しない場合は推論時に失敗する。
- user dictionary mutation と推論を含む request 間に明示的な lock はない。
- static release の HTTP request に timeout は指定しない。

## 関連テスト

次の test が endpoint と関連 component の挙動を検証する。

- `tests/test_server_editor_api.py`: 音声合成 endpoint の speaker 解決とユーザー辞書 endpoint の辞書更新回数
- `tests/test_tts_model_holder.py`: model metadata の走査と model 取得
- `tests/test_japanese_g2p_snapshot.py`: 日本語 G2P 出力
- `tests/test_normalizer.py`: 日本語正規化
- `tests/test_pyopenjtalk_worker_adapter.py`: pyopenjtalk worker adapter
