# Dashboard テストガイド

Reactコンポーネントを修正した後のテスト方法を説明します。

## クイックテスト (推奨)

統合テストスクリプトを使用するのが最も簡単です:

```bash
./dashboard/tests/test_dashboard.sh
```

このスクリプトは以下を自動実行します:
1. フロントエンドのビルド (`npm run build`)
2. ダミーデータの生成
3. ダッシュボードHTMLの生成
4. ブラウザでの表示

### オプション

```bash
# ビルドをスキップ(既存のdist/index.htmlを使用)
./dashboard/tests/test_dashboard.sh --no-build

# ブラウザを開かない
./dashboard/tests/test_dashboard.sh --no-open

# 両方を組み合わせ
./dashboard/tests/test_dashboard.sh --no-build --no-open
```

## 手動テスト

より細かい制御が必要な場合:

### 1. フロントエンドをビルド

```bash
cd dashboard/frontend
npm run build
```

`dashboard/frontend/dist/index.html` が生成されます。

### 2. テストダッシュボードを生成

```bash
cd ../..  # プロジェクトルートに戻る
uv run python -c "
from dashboard.tests.dummy_data import generate_circular_trajectory
from dashboard.generator import HTMLDashboardGenerator
from pathlib import Path

log = generate_circular_trajectory()
generator = HTMLDashboardGenerator()
generator.generate(log, Path('test_dashboard.html'))
"
```

### 3. ブラウザで確認

```bash
xdg-open test_dashboard.html  # Linux
# または
open test_dashboard.html      # macOS
```

## 確認項目

### TrajectoryView (X-Y プロット)
- [ ] マウスホイールでズーム
- [ ] ドラッグでパン
- [ ] ホバーで座標情報表示
- [ ] ダブルクリックでリセット
- [ ] Plotlyツールバー表示
- [ ] 現在位置マーカー表示

### TimeSeriesPlot (時系列グラフ)
- [ ] マウスホイールでズーム
- [ ] ドラッグでパン
- [ ] ホバーで値表示
- [ ] ダブルクリックでリセット
- [ ] Plotlyツールバー表示
- [ ] 現在時刻の垂直線表示
- [ ] グラフタイトル横に現在値表示

### 全体
- [ ] タイムスライダーで全グラフが同期更新
- [ ] レイアウトが正常
- [ ] MUIテーマとの統一感

## 開発サーバー (参考)

開発中のプレビュー用に開発サーバーを使用できますが、**実際のダッシュボード生成では使用されません**:

```bash
cd dashboard/frontend
npm run dev
```

開発サーバーは `public/simulation_log.json` からデータを読み込みます。最終的な検証は必ず上記のビルド→生成フローで行ってください。

## トラブルシューティング

### ビルドエラー

TypeScriptエラーを確認:

```bash
cd dashboard/frontend
npm run build
```

### ダッシュボード生成エラー

`dist/index.html` の存在を確認:

```bash
ls -lh dashboard/frontend/dist/index.html
```

### ダッシュボードが表示されない

ブラウザの開発者ツール (F12) でコンソールエラーを確認してください。

## テストデータのカスタマイズ

`dashboard/tests/dummy_data.py` を編集してテストデータをカスタマイズできます:

```python
from dashboard.tests.dummy_data import (
    generate_circular_trajectory,
    generate_figure_eight_trajectory,
)

# 円形軌跡
log = generate_circular_trajectory(num_steps=200, radius=100.0)

# 8の字軌跡
log = generate_figure_eight_trajectory(num_steps=200, radius=30.0)
```
