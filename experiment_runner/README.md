# Experiment Runner

統一実験実行フレームワーク。YAML設定ファイルで実験を定義・実行できます。

## Usage

### 実行方法

```bash
uv run experiment-runner --config configs/experiments/pure_pursuit.yaml
```

## Testing

### Testing

```bash
# Run unit tests
uv run pytest

# Run integration tests
uv run pytest experiment_runner/tests -m integration -v
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
- `core`, `simulators`, `tools`: 内部パッケージ
- `pure-pursuit`, `pid-controller`, `neural-controller`, `planning-utils`: コンポーネントパッケージ
