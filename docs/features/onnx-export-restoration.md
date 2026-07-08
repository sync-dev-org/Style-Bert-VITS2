# feature: ONNX エクスポートの復旧と AIVM 系統の削除

torch 現行系 (lock: torch 2.12.1) で ONNX エクスポート経路 (`convert_onnx.py` / `convert_bert_onnx.py`) を成立させ、AIVM / AIVMX 生成系統 (AivisSpeech 連携) を削除する。GPU 実行 (onnxruntime-gpu) / TensorRT 対応は本 feature の対象外 ([ROADMAP.md](../ROADMAP.md) 次候補)。

## 決定事項

- exporter は TorchScript 経路 (`dynamo=False` 明示) を維持する。torch 2.9 以降 `torch.onnx.export` の既定は dynamo ベース exporter (onnxscript 必須) に切り替わっており、明示なしの現行コードは import 段で停止する。TorchScript 経路に削除予定は出ておらず (torch 2.12 時点)、VITS 系の data-dependent graph は dynamo 経路で capture error になる既知 issue があるため、dynamo 移行は将来の別 feature とする
- opset は 20 を明示する (TorchScript 経路の上限であり、torch 2.12 の既定推奨値)
- onnxscript / aivmlib は依存に追加しない。依存 (pyproject.toml / uv.lock) は変更しない
- AIVM / AIVMX 生成機能は復活予定なしの削除とする

## 変更 scope

1. `convert_onnx.py`: `torch.onnx.export` 2 箇所 (JP-Extra / 非 JP-Extra) へ `dynamo=False` + `opset_version=20` を明示する。`--aivm` / `--aivmx` オプション、AIVM メタデータ生成、aivmlib import を削除する
2. `convert_bert_onnx.py`: `torch.onnx.export` へ同様の明示を行う。tokenizer 出力に `token_type_ids` が含まれない場合 (現行 transformers の fast tokenizer) に対応する。実行時に tracked file (`bert/<model>/tokenizer.json`) を上書きしない出力設計へ改める
3. `gradio_tabs/convert_onnx.py`: タブ説明文から AIVM Generator / AivisSpeech への誘導を削除する
4. `README.md`: ONNX 変換の説明から AIVM Generator / AivisSpeech への誘導文を削除する

## やらないこと

- dynamo exporter (`dynamo=True`) への移行 (`report=True` dry-run の詰まり棚卸しから始める将来の別 feature)
- onnxruntime-gpu / TensorRT / DirectML 対応 (ROADMAP 次候補)
- `katakana_map.py` の AIVM 系読み上げ辞書エントリの削除 (g2p 資産であり本機能と独立)
- 非 JP-Extra モデルでの実機エクスポート検証 (コード修正は両経路に適用するが、実機検証資産は JP-Extra のみ)

## 受入条件

1. torch 2.12.1 環境で `convert_onnx.py --model model_assets/<model>/<model>.safetensors` (JP-Extra モデル) が完走し、`.onnx` が生成される
2. 生成された ONNX モデルを onnxruntime (CPUExecutionProvider) で読み込み、既存の ONNX 推論経路で短文合成が成功する (出力波形が非零長かつ NaN を含まない)
3. `convert_bert_onnx.py` が現行 transformers 環境で完走し、fp32 / fp16 の BERT ONNX が生成されスクリプト内蔵の検証を通過する
4. `convert_onnx.py` / `convert_bert_onnx.py` の実行が git tracked file を変更しない
5. コードベースに aivmlib / `--aivm` / `--aivmx` への参照が残っていない (`katakana_map.py` の読み上げ辞書エントリは対象外)
6. WebUI「ONNX変換」タブおよび README に AivisSpeech / AIVM への誘導記述が残っていない
7. `torch.onnx.export` の全呼び出しで `dynamo` と `opset_version` が明示されている
8. 既存テストスイートが全通過する (darwin、54 passed / 2 skipped の水準を維持)

受入条件 1-3 は実機検証 (darwin / CPU) とし、実施記録を残す。
