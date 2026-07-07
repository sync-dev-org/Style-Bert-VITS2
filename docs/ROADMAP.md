# 計画正典

近代化計画の現在地 snapshot のみを保持する。経緯・履歴は git history を参照。要件とスコープ境界は [REQUIREMENTS.md](REQUIREMENTS.md) を参照。

## 完了

- 依存管理の pyproject.toml 統一 (`requirements*.txt` 廃止、dependency-groups 化)
- バージョン管理の hatch-vcs 化 (git tag `vX.Y.Z` を正典化)
- Python >= 3.10 化、gradio 5 系対応
- テキスト正規化・g2p 基盤の近代化 (上流 dev branch および公開 fork の成果取り込み: 記号・数値・電話番号・郵便番号・住所等の正規化、g2p API 拡張、読み上げ音韻基準の更新)

## 進行中 (常に高々 1)

- src レイアウト化: `style_bert_vits2/` → `src/style_bert_vits2/`。packaging 設定の変更を伴うため、別 branch で git URL install のビルド検証を通してから `sync-dev` へ統合する

## 次候補 (未確定)

- pyproject 内の上流残置設定の掃除 (`[tool.hatch.envs.*]` の Python matrix 不整合、sdist only-include の `.vscode` 等)
- ONNX エクスポート系統の再検討
