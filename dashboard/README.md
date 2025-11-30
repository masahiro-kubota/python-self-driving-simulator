# Dashboard

シミュレーション結果を可視化するインタラクティブなWebダッシュボード。

## 概要

React + TypeScript + Viteで構築されたダッシュボードです。以下の機能を提供します:

- リアルタイム軌跡可視化(Plotly.js)
- 速度、ステアリング、加速度、ヨー角の時系列プロット
- インタラクティブなタイムスライダー
- ズーム、パン、ホバー情報表示

### ダッシュボード生成の仕組み

ダッシュボードは以下の2ステップで生成されます:

1. **フロントエンドのビルド** (`npm run build`)
   - Reactアプリケーションを単一のHTMLファイル(`dist/index.html`)にビルド
   - このHTMLファイルがダッシュボードのテンプレートとなる

2. **データの注入** (Pythonの`HTMLDashboardGenerator`)
   - `SimulationLog`型のデータをテンプレートHTMLに注入
   - 完成したHTMLファイルを出力

この仕組みにより、**`npm run build`が必須**となります。開発サーバー(`npm run dev`)は開発プレビュー用であり、実際のダッシュボード生成には使用できません。

## 開発

### 前提条件

```bash
cd dashboard/frontend
npm install
```

### ビルド

Reactコードを変更した後は、必ずプロダクションビルドを実行してください:

```bash
npm run build
```

これにより `dashboard/frontend/dist/index.html` が生成されます。このファイルがPythonの `HTMLDashboardGenerator` で使用されます。

### テスト

Reactコンポーネントを修正した後は、統合テストスクリプトを実行してください:

```bash
./dashboard/tests/test_dashboard.sh
```

このスクリプトは以下を自動実行します:
1. フロントエンドのビルド
2. ダミーデータの生成
3. ダッシュボードHTMLの生成
4. ブラウザでの表示

**オプション:**
- `--no-build`: ビルドをスキップ(既存のdist/index.htmlを使用)
- `--no-open`: ブラウザを開かない

詳細は [TESTING.md](./TESTING.md) を参照してください。

## アーキテクチャ

- **フロントエンド**: React + TypeScript + Vite + MUI
- **状態管理**: Zustand
- **グラフ**: Plotly.js (react-plotly.js経由)
- **ビルド**: 単一HTMLファイル出力 (vite-plugin-singlefile)

## 統合

ダッシュボードは実験ランナーと統合されています:

1. `experiment-runner` がシミュレーションデータを生成
2. `HTMLDashboardGenerator` (Python) がデータを `dist/index.html` に注入
3. 生成されたHTMLがMLflowにアーティファクトとしてアップロード

## カスタマイズ

### テーマ

MUIのダークテーマを使用しています。色をカスタマイズするには `src/components/DashboardLayout.tsx` を編集してください:

```typescript
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    // カスタムカラーをここに追加
  },
});
```

### コンポーネント

- `DashboardLayout.tsx`: メインレイアウトとテーマプロバイダー
- `TrajectoryView.tsx`: 2D軌跡可視化 (Plotly)
- `TimeSeriesPlot.tsx`: 時系列グラフ (Plotly)
- `TimeSlider.tsx`: 再生コントロール

## テストデータ

`tests/dummy_data.py` には以下のダミーデータ生成関数があります:

- `generate_circular_trajectory()`: 円形軌跡
- `generate_figure_eight_trajectory()`: 8の字軌跡

新しいテストパターンを追加する場合は、このモジュールに関数を追加してください。
