# NLP・テキスト処理仕様

## 対象範囲

本書は `style_bert_vits2.nlp` が提供する次の処理の挙動を定める。

- 日本語・英語・中国語のテキスト正規化
- 正規化済みテキストからの音素列、tone、`word2ph` の生成
- 日本語の形態素・読み情報と pyopenjtalk worker
- PyTorch / ONNX Runtime 向け BERT モデル管理と特徴量抽出
- 日本語ユーザー辞書の保存、コンパイル、pyopenjtalk への適用
- 音素、tone、言語の数値 ID 化

音声合成全体でこれらの値を使う手順は [core-inference.md](core-inference.md)、ユーザー辞書を操作する HTTP API は [editor-server.md](editor-server.md) が扱う。

## 公開エントリポイント

`style_bert_vits2.nlp` 直下の関数は、`Languages.JP`、`Languages.EN`、`Languages.ZH` に応じて言語別実装へ処理を委譲する。これら以外の言語には `ValueError` を送出する。

| 関数 | 入力 | 戻り値 |
|---|---|---|
| `clean_text` | 生テキスト、言語、`use_jp_extra`、`raise_yomi_error` | 正規化済みテキスト、`phones`、`tones`、`word2ph`、日本語の分割情報 |
| `clean_text_with_given_phone_tone` | `clean_text` の入力と任意の `given_phone` / `given_tone` | 指定値を反映した `clean_text` と同形式の tuple |
| `extract_bert_feature` | 正規化済みテキスト、`word2ph`、言語、device、任意の補助テキスト | phone 単位の PyTorch tensor |
| `extract_bert_feature_onnx` | 正規化済みテキスト、`word2ph`、言語、ExecutionProvider、任意の補助テキスト | phone 単位の NumPy 配列 |
| `cleaned_text_to_sequence` | `phones`、`tones`、言語 | 音素 ID、言語間で offset 済みの tone ID、言語 ID |

実装は言語別モジュールを分岐内で import する。`style_bert_vits2.nlp` の import だけでは PyTorch や各言語の重い処理を読み込まない。

## 共通データ契約

### `clean_text` の戻り値

戻り値は次の 7 要素である。

1. `norm_text: str`: 言語別に正規化したテキスト
2. `phones: list[str]`: 音素と punctuation の列
3. `tones: list[int]`: `phones` の各要素に対応する tone
4. `word2ph: list[int]`: BERT token または文字ごとに割り当てる phone 数
5. `sep_text: list[str] | None`: 日本語の形態素表層形
6. `sep_kata: list[str] | None`: 日本語の形態素ごとのカタカナ読みまたは punctuation
7. `sep_kata_with_joshi: list[str] | None`: 助詞・助動詞の読みを直前要素へ連結した日本語の読み

英語と中国語では末尾 3 要素はすべて `None` になる。

### g2p の不変条件

全言語の g2p は次を満たす。

- `phones` と `tones` の長さは等しい。
- `len(phones) == sum(word2ph)` である。
- `phones` の先頭と末尾は padding 記号 `_` である。
- 両端の `tones` は `0`、両端の `word2ph` は `1` である。
- punctuation は音素列内に保持され、その tone は `0` になる。

`word2ph` の要素が表す単位は言語ごとに異なる。

| 言語 | 対応単位 | g2p 内の対応付け |
|---|---|---|
| 日本語 | 日本語 BERT tokenizer が形態素ごとに生成した token | 形態素内の phone を token へ左から均等分配 |
| 英語 | DeBERTa tokenizer の token を単語状にまとめた要素 | 単語状要素内の phone を構成 token へ左から均等分配 |
| 中国語 | 正規化済みテキストの各文字 | pinyin が生成する 1 個または複数の phone 数 |

### tone の範囲

- 日本語は低音 `0` と高音 `1` を使う。
- 英語は ARPAbet の stress `0`、`1`、`2` をそれぞれ `1`、`2`、`3` に変換する。stress を持たない phone は `3`、punctuation は `0` になる。
- 中国語は声調 `1` から `5` を使い、punctuation は `0` になる。

### 指定 phone / tone の反映

`clean_text_with_given_phone_tone` は最初に通常の `clean_text` を実行し、次の規則で指定値を反映する。

- `given_phone` と `given_tone` が両方ある場合だけ、生成済みの phone と tone を両方置換する。
- 2 つの指定列の長さが異なる場合は `InvalidPhoneError` を送出する。
- 指定 phone 数が `sum(word2ph)` と異なる場合、日本語では生成 phone と指定 phone の最長共通部分列を基に `word2ph` を調整する。他言語では `InvalidPhoneError` を送出する。
- 日本語の調整後も `sum(word2ph)` と指定 phone 数が一致しなければ `InvalidPhoneError` を送出する。
- `use_jp_extra=False` の日本語では、指定 phone 内の `N` も `n` に変換する。
- `given_tone` だけがある場合は、生成 phone と同じ長さであれば tone だけを置換し、異なれば `InvalidToneError` を送出する。
- `given_phone` だけがある場合は反映せず、通常生成した値を返す。

日本語の `word2ph` 調整は両端の padding 対応を維持する。内部要素はまず 1 以上 6 以下へ寄せ、phone 数の差がその範囲で吸収できない場合も、最終的に指定 phone 数との合計一致を優先する。

### 数値 ID 化

`cleaned_text_to_sequence` は `SYMBOLS` の index を音素 ID とする。tone ID は言語別 tone 空間が重ならないように offset を加え、言語 ID は phone 数だけ複製する。

| 言語 | 言語 ID | tone offset |
|---|---:|---:|
| 中国語 | 0 | 0 |
| 日本語 | 1 | 6 |
| 英語 | 2 | 8 |

`SYMBOLS` は `_`、3 言語の音素の和集合、`! ? … , . ' -`、`SP`、`UNK` からなる。未知の文字列をこの関数へ渡すと、音素 ID の検索で失敗する。

## テキスト正規化

### 日本語

`japanese.normalizer.normalize_text` は次の順序で処理する。順序は変換結果の一部である。

1. カタカナ以外の全角 ASCII・数字を半角化する。
2. 数字列として扱う漢数字、区切り記号、ゼロ表記の揺れを整理する。
3. Unicode 正規化前に URL、メール、日付、時刻、数式、単位などの文脈依存記号を読みへ変換する。
4. 全角空白を `。`、ゼロ幅スペースを空文字へ変換する。
5. NFKC を適用し、異体字を `ITAIJI_MAP` で置換する。
6. 漢字の繰り返し記号 `〻` は直前が漢字系文字なら直前文字へ展開する。
7. 英字列を辞書・規則・推定によりカタカナ化する。
8. 数値に付く単位、通貨、桁区切りを読みやすい表現へ変換する。
9. `~`、`～`、`〜` を長音記号 `ー` にする。
10. punctuation を共通記号へ畳み、許可文字以外を除去する。
11. 単独の結合濁点・半濁点を除去する。

結果に残り得る文字は、日本語文字、漢字系文字、半角英数字、ギリシャ文字、共通 punctuation、`/`、`—` である。`/` は形態素解析に必要なため g2p まで保持し、読みにできなければ `.` として扱う。漢字直後の連続 `-` は pyopenjtalk の読み取得を保つため同数の `—` にし、g2p で `-` に戻す。

主な規則群と代表例は次のとおりである。

| 規則群 | 代表的な変換 |
|---|---|
| punctuation・空白 | `。` → `.`, `、` → `,`, `…` → `...`, 全角空白 → `.` |
| 単位 | `100m` → `100メートル`, `5GHz` → `5ギガヘルツ`, `3.5km/s` → `3.5キロメートル毎秒` |
| 通貨・桁区切り | `¥1,200` → `1200円`, `$100` → `100ドル` |
| 日付・時刻 | `2024/01/01` → `2024年1月1日`, `09:30` → `九時三十分` |
| 数式・比 | `5*8` → `5かける8`, `16:9` → `十六タイ九`, `2^3` → `2の3乗` |
| 電話・郵便番号 | 対象形式を一桁ずつのカタカナ読みにし、まとまりを `,` や `の` で区切る |
| 住所・部屋・階 | 住所文脈の `1-2-3` → `1の2の3`, `B1F` → `地下1階` |
| URL・メール | scheme、domain separator、`@`、path separator などをカタカナ読みへ展開 |
| 英字列 | 固定辞書、大小文字、CamelCase、略語、英語推定、C2K の順で解決可能な部分をカタカナ化 |
| 記号 | 数式内の `×` は `かける`、それ以外は `バツ`。対応表にある数学・集合・単位記号を読みへ変換 |

英字変換は固定の `KATAKANA_MAP` を優先し、敬称、複合語、CamelCase、数字付き語、NGram による英単語・アルファベット読み判定、C2K による推定を組み合わせる。解決できない英字は残り、pyopenjtalk 側のアルファベット読み対象になる。

### 英語

`english.normalizer.normalize_text` は数値を英単語へ展開してから punctuation を置換する。

- 桁区切り comma を除去する。
- `£` と `$` を pounds、dollars / cents の表現へ展開する。
- 小数点を `point`、序数を序数語、整数を英単語へ変換する。
- 1000 より大きく 3000 より小さい整数には年号風の読み分けを適用する。
- 全角 punctuation、括弧、引用符、dash、改行を共通 punctuation へ置換する。
- `, ; . ? !` の直後に word 文字が続く場合は空白を 1 個挿入する。

英語の punctuation 置換は許可文字による除去を行わず、置換対象外の文字を保持する。

### 中国語

`chinese.normalizer.normalize_text` は次を行う。

- 整数または小数を `cn2an.an2cn` で漢数字へ変換する。
- `嗯` を `恩`、`呣` を `母` に置換する。
- 全角 punctuation、括弧、引用符、dash、改行を共通 punctuation へ変換する。
- CJK 基本漢字と共通 punctuation 以外を除去する。

## grapheme-to-phoneme 変換

### 日本語

`japanese.g2p.g2p` は正規化済みテキストを受け取る。処理は次のとおりである。

1. pyopenjtalk frontend から形態素情報を得る。
2. full-context label から punctuation を除いた phone と prosody 記号を抽出し、アクセント句ごとの tone を `0` / `1` に正規化する。
3. frontend の表層形と読みから `sep_text`、`sep_kata`、`sep_kata_with_joshi` を作る。
4. カタカナの各モーラを phone へ変換し、長音を直前母音へ展開する。
5. punctuation を保持した phone 列へ、punctuation を除いた tone 列を照合する。
6. 日本語 BERT tokenizer で形態素を token 化し、形態素内の phone 数を token 数へ分配して `word2ph` を作る。
7. 両端の padding を追加する。

促音 `ッ` は `q`、撥音 `ン` は `N` になる。`use_jp_extra=False` の場合だけ、最終 phone 列の `N` を `n` に変換する。長音記号が語頭にある場合、または直前が母音でない場合は dash の phone `-` として扱う。

`text_to_sep_kata` は frontend の `pron` から無声化記号を除く。助詞または助動詞の読みは `sep_kata_with_joshi` の直前要素へ連結する。pyopenjtalk が読めず `pron == "、"` となった表層形は、次の規則で扱う。

- 共通 punctuation は元の記号を保持する。
- `/` は `.`、`—` は `-` にする。
- それ以外は `raise_yomi_error=True` なら `YomiError` を送出する。
- `raise_yomi_error=False` なら表層形の文字数と同数の `'` に置換し、warning を記録する。

`g2p_utils` は phone / tone とカタカナモーラ / tone の相互変換を提供する。子音と後続母音の tone は同一であることを要求し、両端の padding はカタカナ化時に除去し、phone 化時に再付与する。

### 英語

英語 g2p は DeBERTa tokenizer の SentencePiece token を `▁` と punctuation の境界で単語状要素へまとめる。各要素は次の順で発音を得る。

1. bundled CMU dictionary に大文字化した語があれば、その音節列を使う。
2. 辞書になければ `g2p_en.G2p` の結果を使う。
3. ARPAbet phone を小文字化し、stress を tone へ変換する。
4. `v` は共通 symbol の `V` にする。共通 `SYMBOLS` にない結果は `UNK` にする。

apostrophe を含む複数 token の語は 1 語へ結合してから辞書・fallback へ渡す。CMU dictionary は同梱 pickle cache があればそれを読み、なければ text dictionary の 49 行目以降を解析して cache を作る。

### 中国語

中国語 g2p は punctuation の直後で segment を分け、jieba の品詞付き分割と pypinyin を使う。

- ASCII 英字列は g2p segment から除去する。
- `一`、`不`、連続する第三声、軽声、重複語、児化などの変調に必要な segment を結合してから tone sandhi を適用する。
- pypinyin の initial と tone 付き final を文字ごとに取得する。
- `uei` / `iou` / `uen` や initial を持たない syllable の表記を OpenCPOP の key へ正規化する。
- `opencpop-strict.txt` の対応表で pinyin を 1 個または複数の phone に変換し、同じ tone を各 phone に複製する。
- punctuation は 1 phone、tone `0`、`word2ph` `1` とする。

padding 追加前には `len(word2ph) == len(norm_text)` を要求する。対応表にない pinyin や文字数不整合は assertion failure になる。

## BERT モデル管理

### 言語別モデル

| 言語 | PyTorch model class | tokenizer class | 想定モデル |
|---|---|---|---|
| 日本語 | `AutoModelForMaskedLM` | fast `AutoTokenizer` | `deberta-v2-large-japanese-char-wwm` |
| 英語 | `DebertaV2Model` | `DebertaV2TokenizerFast` | `deberta-v3-large` |
| 中国語 | `AutoModelForMaskedLM` | fast `AutoTokenizer` | `chinese-roberta-wwm-ext-large` |

PyTorch と ONNX は別々に、言語を key とする process-global cache へ model と tokenizer を保持する。既に同じ言語が load 済みなら、後続の path、device map、provider、revision の指定にかかわらず既存 object を返す。

### PyTorch

- path を省略した場合は言語別 default path の存在を assertion で要求する。
- 英語以外は masked language model、英語は `DebertaV2Model` として load する。
- model は float32 で load する。Transformers 5 以降は `dtype`、それより前は `torch_dtype` を渡す。
- `transfer_model` は未 load なら `ValueError`、現在 device の文字列表現が指定 device で始まる場合は何もしない。それ以外は `.to(device)` で移動する。
- PyTorch 用 default tokenizer がなく、同じ言語の ONNX tokenizer が load 済みの場合、PyTorch 側の `load_tokenizer` はその ONNX tokenizer を返す。
- unload は cache から削除して garbage collection を行い、model unload 時は利用可能なら CUDA cache も空にする。

### ONNX Runtime

- path を省略した場合は言語別 ONNX default path の存在を assertion で要求する。
- `owner/repository` の 2 segment と解釈できる指定は Hugging Face repository とし、`model_fp16.onnx` を取得する。英語では `spm.model` も取得する。
- それ以外の指定は directory とし、その直下の `model_fp16.onnx` を使う。
- graph optimization は無効、log severity は error のみに設定する。
- provider が空なら assertion failure になる。
- 最優先 provider が `CPUExecutionProvider` の場合、明示指定がない限り CPU memory arena を無効にする。`enable_cpu_mem_arena` が指定されればその値を優先する。
- default provider は `CPUExecutionProvider` で、arena の拡張方式に `kSameAsRequested` を指定する。
- unload は cache から削除し garbage collection を行う。

## BERT 特徴量抽出

### 共通処理

PyTorch 系は model の hidden states のうち末尾から 3 層目を使う。ONNX 系は session の最初の出力を使う。各 token の vector を対応する `word2ph[i]` 回複製し、連結後に転置する。

戻り値の shape は `(hidden_size, len(phones))`、第 2 軸は `sum(word2ph)` と一致する。PyTorch 系は device が文字列 `cuda` で CUDA が利用不能な場合だけ `cpu` へ fallback する。

`assist_text` が truthy の場合は補助テキストの token feature の平均を求め、各 main feature を次で混合してから複製する。

```text
main * (1 - assist_text_weight) + assist_mean * assist_text_weight
```

`assist_text_weight` の範囲検証は行わない。

### 言語差

| 言語 | `word2ph` と token の整合条件 | ONNX 入力 |
|---|---|---|
| 日本語 | `len(word2ph) == len("".join(sep_text)) + 2` | `input_ids`, `attention_mask` |
| 英語 | `len(word2ph) == model output の token 数` | `input_ids`, `attention_mask` |
| 中国語 | `len(word2ph) == len(text) + 2` | `input_ids`, `token_type_ids`, `attention_mask` |

日本語は `sep_text` が渡されれば再度 frontend を実行せず、それを連結したテキストを BERT 入力にする。未指定なら `text_to_sep_kata(..., raise_yomi_error=False)` で作る。日本語の補助テキストも同じ処理で読める表層形だけを連結する。

ONNX 系は指定 session と provider から device type、device ID、run options を求め、IOBinding で入力と出力を device に bind する。補助テキストの推論時は IOBinding を新しく作る。

## pyopenjtalk worker

### interface と fallback

`japanese.pyopenjtalk_worker` は次の関数を pyopenjtalk と同形で公開する。

- `run_frontend`
- `make_label`
- `mecab_dict_index`
- `update_global_jtalk_with_user_dict`
- `unset_user_dict`

`initialize_worker` が成功して `WORKER_CLIENT` がある場合は別 process へ dispatch し、未初期化なら呼び出し process 内で pyopenjtalk を直接実行する。

### frontend adapter

`run_frontend` は pyopenjtalk の signature が受け付ける範囲で、次の互換引数を渡す。

- `run_marine=False`
- `use_vanilla=False`
- `use_sudachi_kanji_yomi=True`
- `predict_nani=True`
- `normalize_mode="None"`
- `use_read_as_pron=False`
- `revert_long_vowels=False`
- `revert_yotsugana=False`

signature を取得できない場合は追加引数を渡さない。`**kwargs` を受ける実装には全項目を渡す。frontend 結果のうち、品詞が記号、`mora_size == 0`、表層形が共通 punctuation だけの要素は、疑問符だけなら `read` / `pron` を `？`、それ以外を `、` に統一する。

### process lifecycle

`initialize_worker` は既存 client があれば何もしない。既定 port は `7861` である。

1. host 名と port へ接続を試みる。
2. 接続できなければ `python -m style_bert_vits2.nlp.japanese.pyopenjtalk_worker --port ...` を別 session / process group で起動する。
3. 0.5 秒間隔で最大 20 回接続を再試行し、接続できなければ `TimeoutError` を送出する。
4. 成功後は `atexit` と、main thread で設定可能なら `SIGTERM` handler に `terminate_worker` を登録する。

client socket の timeout は 60 秒である。`terminate_worker` は server の client 数が 1 の場合だけ quit request を送り、client socket を閉じて process-global client を解除する。server は client が 0 の状態が既定 30 秒を超えた場合にも終了する。

### 通信 protocol

client と server は TCP 上で JSON object を送受信する。message は 4 byte big-endian の body 長と UTF-8 JSON body からなる。request type は status、server 終了、pyopenjtalk 呼び出しの 3 種である。

server が dispatch できる関数は公開 interface の 5 関数に固定される。複数 client を `select` で扱い、切断時に client 数を減らす。接続が body 受信途中で閉じた場合は `ConnectionClosedException` とする。

## 日本語ユーザー辞書

辞書管理 HTTP API の詳細を除き、NLP 層の保存・適用契約は次のとおりである。

### データ

- default CSV、user JSON、compiled dictionary の 3 file を使う。各関数の path 引数で default path を差し替えられる。
- user JSON は UUID 文字列を key、単語情報を value とする。
- 公開 model の `priority` は 0 から 10 だが、JSON には MeCab cost として保存する。読み出し時に品詞別 cost 候補の最近傍から priority へ戻す。
- `context_id` のない entry は固有名詞の context ID を補う。
- 品詞は固有名詞、普通名詞、動詞、形容詞、接尾辞をサポートする。品詞ごとに context ID、cost 候補、許可するアクセント結合規則が固定される。

`UserDictWord` は surface の半角 ASCII 記号・英数字を全角へ変換する。発音はカタカナと長音だけを許可し、不正な捨て仮名の連続や `ク` / `グ` 以外に続く `ヮ` を拒否する。モーラ数を省略すると発音から算出し、`accent_type` が 0 以上モーラ数以下であることを要求する。

単語作成時の既定は固有名詞、priority `5`、アクセント結合規則 `*` である。不明な品詞と範囲外 priority は HTTP 422 相当の例外になる。

### 更新と適用

追加、更新、削除、import は user JSON を書き換えた後に `update_dict` を呼ぶ。

1. default CSV がなければ warning を stderr へ出して更新を終了する。
2. default CSV と user JSON の entry を一時 CSV へ連結する。
3. UUID を suffix に持つ一時 file へ `mecab_dict_index` でコンパイルする。
4. 出力 file がなければ `RuntimeError` とする。
5. 現在の user dictionary を解除し、一時 compiled dictionary を本来の path へ置換して読み込む。
6. 成否にかかわらず一時 CSV と一時 compiled dictionary を削除する。

追加は新しい UUID を返す。更新・削除は UUID がなければ HTTP 422 相当の例外を送出する。import は UUID、model 型、品詞 metadata、アクセント結合規則を検証する。同じ UUID がある場合、`override=True` では import 側、`False` では既存側を残す。

辞書処理内の排他制御は有効化されていない。同じ辞書 file への並行書き込みは同期されない。

## 静的データ

- `symbols.py` は言語別 phone、tone 数、punctuation、padding、言語 ID を定義する。
- `japanese/mora_list.py` はカタカナモーラと子音・母音の双方向対応を定義する。
- `japanese/katakana_map.py` は英字列の固定カタカナ読みを定義する。
- `japanese/itaiji_map.py` は異体字の正規化先を定義する。
- `english/cmudict.rep` と `cmudict_cache.pickle` は英語発音辞書を構成する。
- `chinese/opencpop-strict.txt` は正規化済み pinyin と phone 列の対応を定義する。

これらの表に entry を追加・変更すると、正規化、g2p、利用可能 symbol の結果が変化する。

## 制約とエッジケース

- g2p は正規化済みテキストを前提とする。公開 `clean_text` は正規化を先に実行する。
- 日本語 g2p は BERT tokenizer を `word2ph` 生成にも使うため、model feature を抽出しない場合でも日本語 tokenizer が必要である。
- 英語 g2p も単語境界と `word2ph` のため英語 tokenizer を必要とする。
- model / tokenizer cache は言語単位であり、同じ process 内で異なる source や provider へ切り替えるには先に unload が必要である。
- assertion で表される token 数、文字数、phone 数の不整合は回復せず、呼び出し側へ失敗として返る。
- 日本語で読めない文字を `'` に置換すると、入力表層の文字数を維持して `word2ph` と BERT token の対応を継続する。
- `cleaned_text_to_sequence` は `phones` と `tones` の長さを検査しない。整合済みの g2p 出力を渡すことを前提とする。

## 関連テスト

- `tests/test_normalizer.py`
- `tests/test_japanese_g2p_snapshot.py`
- `tests/test_english_g2p.py`
- `tests/test_chinese_g2p.py`
- `tests/test_bert_tokenizers.py`
- `tests/test_japanese_bert_feature.py`
- `tests/test_multilingual_bert_feature.py`
- `tests/test_pyopenjtalk_worker_adapter.py`

日本語 g2p の代表文は `tests/snapshots/japanese_g2p_snapshot.json` に正規化結果、形態素、読み、phone、tone、`word2ph` の組として固定される。
