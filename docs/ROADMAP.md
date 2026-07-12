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
- テスト構造の整備 ([features/test-structure-overhaul.md](features/test-structure-overhaul.md))
- テストカバレッジの拡張 ([features/test-coverage-expansion.md](features/test-coverage-expansion.md))
- torch CUDA wheel の入手 index の cu129 統一 (pyproject explicit index pin、uv.lock 再生成、docs / Windows インストーラ追従)
- 挙動仕様正典 docs/spec/ の整備 (コードベースから逆算した全域 8 系統の現状挙動仕様、入口は [spec/README.md](spec/README.md)。維持規律は AGENTS.md 文書正典層に登録済み)
- 訓練経路の torch 現行化 ([features/train-torch-modernization.md](features/train-torch-modernization.md)。CUDA 実機での既定 CLI smoke 完了。Blackwell 筐体固有の互換検証は実機入手時の将来項目)

## 進行中 (常に高々 1)

- なし (次着手はユーザー合意で確定する)

## 次候補 (未確定)

- ONNX 推論の GPU 実行 (onnxruntime-gpu CUDA EP の optional path 化)
- TensorRT 対応 + dynamo exporter 移行 (experimental)
