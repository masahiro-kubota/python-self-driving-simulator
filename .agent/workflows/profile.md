---
description: experiment-runnerのプロファイリング実行
---

# experiment-runnerのプロファイリング

このワークフローでは、`experiment-runner`のパフォーマンスをプロファイリングし、ボトルネックを特定します。

## 手順

### 1. Flamegraph形式でプロファイリング

SVG形式のflamegraphを生成します。ブラウザで開いて視覚的にボトルネックを確認できます。

```bash
// turbo
py-spy record -o profile_flamegraph.svg -- uv run experiment-runner
```

実行後、`profile_flamegraph.svg`をブラウザで開いてください。

### 2. Speedscope形式でプロファイリング（推奨）

インタラクティブに解析できるSpeedscope形式で出力します。

```bash
// turbo
py-spy record -o profile.speedscope.json --format speedscope -- uv run experiment-runner
```

実行後、https://www.speedscope.app/ にアクセスして`profile.speedscope.json`をドラッグ&ドロップしてください。

### 3. プロファイル結果の確認

- **Flamegraph (SVG)**: 横幅が実行時間に比例。上に行くほど呼び出し階層が深い
- **Speedscope**: ズーム、検索、時系列表示が可能

### 4. ボトルネック特定のヒント

プロファイル結果で以下を確認:
- 最も幅の広い（時間を消費している）関数
- 予想外に時間がかかっている処理
- 繰り返し呼ばれている関数

### 5. 特定の関数に絞ってプロファイリング

特定のモジュールだけを見たい場合:

```bash
py-spy record -o profile.svg --subprocesses -- uv run experiment-runner
```

## オプション

- `--rate <Hz>`: サンプリングレート（デフォルト100Hz）
- `--duration <秒>`: 最大記録時間
- `--subprocesses`: サブプロセスも含めてプロファイリング
- `--native`: C/C++拡張も含める（要root権限）

## トラブルシューティング

権限エラーが出た場合:
```bash
sudo env "PATH=$PATH" py-spy record -o profile.svg -- uv run experiment-runner
```
