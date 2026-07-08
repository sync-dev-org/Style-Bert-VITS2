# 計画正典

近代化計画の現在地 snapshot のみを保持する。経緯・履歴は git history を参照。要件とスコープ境界は [REQUIREMENTS.md](REQUIREMENTS.md) を参照。

## 完了

- 依存管理の pyproject.toml 統一 (`requirements*.txt` 廃止、dependency-groups 化)
- バージョン管理の hatch-vcs 化 (git tag `vX.Y.Z` を正典化)
- Python >= 3.10 化、gradio 5 系対応
- テキスト正規化・g2p 基盤の近代化 (上流 dev branch および公開 fork の成果取り込み: 記号・数値・電話番号・郵便番号・住所等の正規化、g2p API 拡張、読み上げ音韻基準の更新)
- src レイアウト化 (`src/style_bert_vits2/` 配下へ package を配置し、wheel は `style_bert_vits2` package として配布)
- pyproject 内の上流残置設定の掃除 (未使用 hatch envs / coverage 設定の撤去、sdist only-include の `.vscode` 除去、ruff 設定の `[tool.ruff.lint]` 節への移設)
- ONNX エクスポートの復旧と AIVM 系統の削除 ([features/onnx-export-restoration.md](features/onnx-export-restoration.md))

## 進行中 (常に高々 1)

- なし (次着手はユーザー合意で確定する)

## 実機検証待ち

- 訓練経路の torch 現行化 — 実装済み。Ada / Blackwell 実機での短時間訓練 smoke のみ残 (完遂時に完了へ移動。GPU 要件は [REQUIREMENTS.md](REQUIREMENTS.md) 対応環境を参照)

## 次候補 (未確定)

- ONNX 推論の GPU 実行 (onnxruntime-gpu CUDA EP の optional path 化)
- TensorRT 対応 + dynamo exporter 移行 (experimental)
