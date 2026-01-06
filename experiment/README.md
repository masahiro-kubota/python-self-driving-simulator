# Experiment Runner

統一実験実行フレームワーク。YAML設定ファイルで実験を定義・実行できます。

## Usage

### 1. 評価/実験 (Evaluation/Experiment)

デフォルトの実験設定（Pure Pursuit制御 + 障害物回避監視）を実行します。

```bash
uv run experiment-runner --config experiment/configs/experiments/default_experiment.yaml
```

### 2. データ収集 (Data Collection)

(設定ファイルを作成し、`experiment`セクションの `type` を `data_collection` に設定することで実行可能です)

### 3. 学習 (Training)

(学習用スクリプトまたは設定ファイルを用意して実行します)

## 4. プロファイリング (Profiling)

学習時のパフォーマンス分析（PyTorch Profiler）はデフォルトで有効になっています。

### プロファイル結果の確認方法

TensorBoard を使用してプロファイル結果を可視化します。
1. TensorBoard を起動します：
   ```bash
   # 実験結果のディレクトリを日時で指定して表示します
   uv run tensorboard --logdir outputs/YYYY-MM-DD/HH-MM-SS --port 6006 --bind_all
   ```

2. ブラウザで [http://localhost:6006/#pytorch_profiler](http://localhost:6006/#pytorch_profiler) にアクセスします。

### 出力先

プロファイルデータ（JSONトレース）は以下のディレクトリに出力されます：
`outputs/YYYY-MM-DD/HH-MM-SS/profiler/`

## Configuration

`experiment.type` によって動作が変わります。

### 共通設定

```yaml
experiment:
  name: "my_experiment"
  type: "data_collection"  # data_collection, training, evaluation
  description: "..."
```

### Data Collection

```yaml
data_collection:
  storage_backend: "s3"  # s3 or local
  project: "e2e_aichallenge"
  scenario: "pure_pursuit"
  version: "v1.0"
  stage: "raw"
```

### Simulator Configuration

シミュレータ等のシステム設定は `systems/*.yaml` で、ADコンポーネント構成は `modules/*.yaml` で管理されます。メインの実験設定ファイル（`experiments/*.yaml`）からこれらを参照します。

```yaml
experiment:
  # ...

system:
  config_path: "experiment/configs/systems/default_systems.yaml"

module:
  config_path: "experiment/configs/modules/default_module.yaml"
```

## Testing

### Testing

```bash
# Run unit tests
uv run pytest

# Run integration tests
uv run pytest experiment/tests -m integration -v
```

### テスト用MLflow Experimentのクリーンアップ

テストは `test_pure_pursuit_tracking` という専用のMLflow experimentを使用します。
テスト後、以下の方法でクリーンアップできます（アーティファクトも含めて削除されます）：

#### 方法1: MLflow UI経由（推奨）

1. http://localhost:5000 にアクセス
2. 左サイドバーから `test_pure_pursuit_tracking` experimentを選択
3. 上部の "Delete" ボタンをクリック
4. 確認ダイアログで "Delete" を選択

> [!NOTE]
> Experimentを削除すると、関連するすべてのruns、メトリクス、パラメータ、アーティファクト（MCAP、ダッシュボードHTML）が自動的に削除されます。

#### 方法2: Python API経由

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# Experimentを取得
experiment = mlflow.get_experiment_by_name("test_pure_pursuit_tracking")
if experiment:
    # Experimentを削除（すべてのrunsとアーティファクトも削除される）
    mlflow.delete_experiment(experiment.experiment_id)
    print(f"Deleted experiment: {experiment.experiment_id}")
```

#### 方法3: 複数のテストExperimentをまとめて削除

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# test_ プレフィックスのexperimentをすべて削除
experiments = mlflow.search_experiments()
for exp in experiments:
    if exp.name.startswith("test_"):
        mlflow.delete_experiment(exp.experiment_id)
        print(f"Deleted: {exp.name}")
```

## Dependencies

- `pydantic`: 設定のバリデーション
- `pyyaml`: YAML設定の読み込み
- `mlflow`: 実験トラッキング
- `boto3`: MLflow S3アーティファクトストレージ
- `core`, `simulator`, `dashboard`, `experiment-training`: 内部パッケージ
- `pure_pursuit`, `pid_controller`, `neural_controller`, `planning_utils`: コンポーネントパッケージ

## Experiment Results (2026-01-06)

### Optimization: In-Memory Caching (v8)

IOボトルネック解消のため、`ScanControlDataset` にインメモリキャッシュ (`cache_to_ram=True`) を実装しました。

| Metric | Before (mmap) | **After (RAM)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Training Speed** | ~20 batches/sec | **~500 batches/sec** | **25x** |
| **CPU I/O Wait** | ~25-30% | **2-4%** | Eliminated |
| **Disk Read (bi)** | High (continuous) | **0** (during train) | Optimized |

### Evaluation Results (Epoch 38 Model)

7時間学習したモデル (Epoch 38) で評価を実施しましたが、学習不足の可能性があります。

*   **Model:** `outputs/2026-01-06/03-36-56/checkpoints/best_model.npy`
*   **Result (Standard Eval):**
    *   **Success Rate:** 0.0% (0/6 episodes)
    *   **Failure Reason:** All episodes failed with `off_track`.
*   **Debug Eval:**
    *   Pure Pursuit with TinyLidarNet monitoring (Debug Eval) was successful, confirming the pipeline itself works.
