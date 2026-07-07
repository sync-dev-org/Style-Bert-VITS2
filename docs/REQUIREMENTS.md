# 要件正典

本 fork の確定済み要件とスコープ境界を記述する。計画の現在地は [ROADMAP.md](ROADMAP.md) を、個別機能の詳細仕様は `docs/features/` を参照。

## fork の位置づけ

- 本 repo は [litagin02/Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) の独立 fork である
- 上流は開発が停止していると見られ、上流への PR 送付・合流は行わない。上流の未リリース成果 (dev branch 等) や公開 fork の成果は、必要に応じて素材として取り込む
- ライセンスは AGPL-3.0 (上流準拠)。repo は public を維持する

## 互換方針

- 既存の学習済みモデル資産 (`model_assets/` 形式) の読み込み互換を維持する
- `sync-dev` branch は下流利用者が git URL で直接参照する安定線である。パッケージング・依存・API の破壊的変更は別 branch で検証してから統合する
- テキスト正規化・g2p の挙動は品質改善のため変更しうる。挙動変更は `tests/` の snapshot 差分として明示管理し、意図しない回帰と区別する

## 対応環境

- Python >= 3.10 (CI 対象は 3.10 / 3.11 / 3.12)
- PyTorch は optional extra (`torch>=2.1`)。推論は ONNX Runtime のみでも動作する
- 訓練環境は Ada 世代と Blackwell 世代の GPU の両方に対応する。Blackwell (sm_120) は torch >= 2.7 を要するため、訓練経路は torch 現行系を前提に組む
- 依存管理は `pyproject.toml` に一元化する (`requirements*.txt` は持たない)。install 経路は uv を正とする
- バージョンは git tag (`vX.Y.Z`) を正典とし、hatch-vcs で導出する。静的なバージョン定数は持たない

## スコープ境界 (やらないこと)

- 上流への PR 送付・合流準備
- ONNX / AIVMX エクスポート系統の拡張 (現時点ではスコープ外、将来再検討)
