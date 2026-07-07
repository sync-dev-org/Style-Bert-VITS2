# AGENTS.md

## 公開リポジトリの機微情報規律 (厳守)

この repo は公開リポジトリである。git tracked な全ファイル (コード / docs / テスト / fixture / 設定) と commit message に、以下を一切含めない。

- ローカル環境の絶対パス (ホームディレクトリ配下のパス等)・OS ユーザー名・ホスト名
- この repo の外にある内部プロジェクト名・案件/顧客の文脈・業務詳細
- 認証情報・トークン・API キー・内部 URL
- 個人情報

内部文脈・運用記録は `.nf/` 配下 (gitignore 済み・追跡対象外) に置き、`.nf/` を git 追跡へ復帰させない。テスト fixture の文例には一般的な文のみを使用する。

commit / merge の前に、diff と commit message を上記の観点で確認してから打つ。

## リポジトリ運用方針

- 本 repo は litagin02/Style-Bert-VITS2 の独立 fork である。上流は開発が停止していると見られ、上流への合流・PR 送付は前提としない。上流の既存成果 (dev branch 等) は必要に応じて素材として取り込む
- `sync-dev` branch は下流利用者が git URL で直接参照する。依存の大幅変更・API 変更など破壊的変更は別 branch で進め、検証後に統合する
- ライセンスは AGPL-3.0 (上流準拠)。repo は public を維持する
