# feature: 訓練経路の torch 現行化

訓練・前処理経路を torch 現行系 (lock: torch 2.12.1) で deprecation warning なく成立させる。GPU 要件 (Ada / Blackwell 両世代対応、Blackwell は torch >= 2.7 必須) は [REQUIREMENTS.md](../REQUIREMENTS.md) 「対応環境」を参照。

## 決定事項

- torch / torchaudio は lock の現行ペア (torch 2.12.1 + torchaudio 2.11.0) を維持する。torchaudio 2.11.0 は torch への version pin を宣言しておらず (メンテナンスモード移行後の措置)、本 repo の torchaudio 使用面は pure Python 実装の `transforms.Resample` 1 点 (`losses.py` の WavLMLoss) に閉じるため、版ズレが compiled 拡張の ABI 不整合を踏む経路はない。最終裏取りは GPU 実機 smoke (受入条件 7) が担う
- 依存 (pyproject.toml / uv.lock) は変更しない

## 変更 scope

1. `mel_processing.py`: `torch.stft` を `return_complex=True` へ移行する (`spectrogram_torch` / `mel_spectrogram_torch` の 2 箇所)。magnitude は complex tensor から導出し、出力 shape / 数値は従来と等価に保つ
2. weight_norm の新 API 移行: `torch.nn.utils.weight_norm` / `remove_weight_norm` を `torch.nn.utils.parametrizations.weight_norm` 系へ移行する (`src/style_bert_vits2/models/modules.py` / `models.py` / `models_jp_extra.py`)。旧形式 state_dict (`weight_g` / `weight_v` key) の読み込み互換は load 経路の key 変換で吸収する
3. `train_ms.py` / `train_ms_jp_extra.py`: 非推奨の bare call `torch.backends.cuda.sdp_kernel("flash")` を撤去する (実効設定は既存の `enable_*_sdp` 呼び出しが担う)
4. `torch.load` の `weights_only` 明示: cache (`.spec.pt` / `.bert.pt`) と自 repo 生成 checkpoint は `weights_only=True` とし、それ以外を許す load には明示指定と理由を付す

## やらないこと

- torchaudio の内製化・依存変更
- torch / torchaudio の version 変更
- ONNX エクスポート系統の拡張
- pyannote-audio import 時の環境 warning (Matplotlib cache 等) への対処

## 受入条件

1. 既存テストスイートが全通過する (darwin、`uv run pytest -k "not cuda and not directml"` で 49 passed / 2 skipped 相当の水準を維持)
2. `spectrogram_torch` / `mel_spectrogram_torch` が、移行前実装で生成した固定入力に対する参照出力と数値等価 (atol 1e-6 目安) の結果を返す
3. 既存の学習済みモデル資産 (`model_assets/` 形式) が新実装で load でき、同一入力・同一シードの推論出力が移行前と等価
4. 旧 weight_norm 形式 (`weight_g` / `weight_v` key) の training checkpoint を新実装で resume load できる
5. 訓練スクリプト (`train_ms.py` / `train_ms_jp_extra.py`) の import とモデル構築で torch の deprecation / FutureWarning (weight_norm / stft return_complex / sdp_kernel 由来) が発生しない
6. `torch.load` の全呼び出しで `weights_only` が明示されている
7. GPU 実機 smoke: Ada / Blackwell 各筐体で短時間の訓練が完走する (repo 外の手動検証、実施記録を残す)

受入条件 2-4 の参照 fixture は移行前実装から生成し、テスト資材として commit する。
