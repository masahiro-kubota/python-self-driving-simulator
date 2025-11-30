# Visualization Tools

可視化ツールとスクリプト集。

## スクリプト

### 1. `plot_logs.py` - 静止画プロット

SimulationLog (JSON) から静止画を生成します。

```bash
uv run python tools/scripts/plot_logs.py \
  results/log_nn.json \
  -o output.png
```

### 2. `animate_logs.py` - アニメーション生成

SimulationLog (JSON) からアニメーション (GIF/MP4) を生成します。

```bash
uv run python tools/scripts/animate_logs.py \
  data/training/raw/log_pure_pursuit.json \
  results/log_nn.json \
  -o comparison.gif \
  --skip 5
```

オプション:
- `-i, --interval`: フレーム間隔 [ms] (デフォルト: 50)
- `-s, --skip`: ステップスキップ数 (デフォルト: 5)
- `--no-track`: トラックを表示しない

### 3. `mcap_to_log.py` - MCAP変換

MCAP形式のログをSimulationLog (JSON) に変換します。

```bash
# MCAPをJSONに変換
uv run python tools/scripts/mcap_to_log.py \
  simulation.mcap \
  -o log.json

# 変換後、既存のツールで可視化
uv run python tools/scripts/plot_logs.py log.json -o plot.png
uv run python tools/scripts/animate_logs.py log.json -o animation.gif
```

## MLflowからのMCAP可視化ワークフロー

1. MLflow UIからMCAPファイルをダウンロード
2. MCAPをJSONに変換
3. 既存ツールで可視化

```bash
# 1. MLflow UIから simulation_nn.mcap をダウンロード

# 2. 変換
uv run python tools/scripts/mcap_to_log.py \
  ~/Downloads/simulation_nn.mcap \
  -o log_from_mlflow.json

# 3. 可視化
uv run python tools/scripts/animate_logs.py \
  log_from_mlflow.json \
  -o animation.gif
```

## Python API

```python
from tools.visualization import SimulationPlotter, SimulationAnimator
from core.data import SimulationLog

# 静止画
log = SimulationLog.load("log.json")
plotter = SimulationPlotter()
plotter.add_log(log, label="Experiment 1")
plotter.save("output.png")

# アニメーション
animator = SimulationAnimator()
animator.add_log(log, label="Experiment 1")
animator.save_animation("output.gif", interval=50)
```
