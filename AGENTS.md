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

## 文書正典層

確定事項は `docs/` 配下の正典が持ち、仕掛かり・未決の追跡は issue 台帳 (`.nf/` 配下、追跡対象外) が担う。

- 要件正典: `docs/REQUIREMENTS.md` — 確定済み要件とスコープ境界
- 計画正典: `docs/ROADMAP.md` — 計画の現在地 snapshot。feature 完遂 commit と同時に更新する
- feature 仕様: `docs/features/<name>.md` — 着手時に 1 枚起こす。受入条件 (番号付き・挙動ベース) がテストの導出元になる
- 挙動仕様正典: `docs/spec/` — コードベースから逆算した現状挙動の仕様 (入口は `docs/spec/README.md`)。挙動を変える commit と同時に該当 spec を更新する
