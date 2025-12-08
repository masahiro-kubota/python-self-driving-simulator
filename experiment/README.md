# Experiment Runner

統一実験実行フレームワーク。YAML設定ファイルで実験を定義・実行できます。

## Usage

### 1. データ収集 (Data Collection)

シミュレーションを実行し、ログデータをMinIOに保存します。

```bash
uv run experiment-runner --config experiment/configs/experiments/data_collection_pure_pursuit.yaml
```

### 2. 学習 (Training)

MinIOからデータをダウンロードし、モデルを学習します。

```bash
uv run experiment-runner --config experiment/configs/experiments/imitation_learning_s3.yaml
```

### 3. 評価 (Evaluation)

学習済みモデル（またはルールベースコントローラー）を評価します。

```bash
uv run experiment-runner --config experiment/configs/experiments/pure_pursuit.yaml
```

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

シミュレータの設定は `simulator` セクションで行います。

```yaml
simulator:
  type: "simulator.KinematicSimulator"
  params:
    dt: 0.1
    initial_state:
      from_track: true
    # 車両・シーン設定ファイルのパス（オプション）
    # 指定しない場合はデフォルト値または後方互換パラメータが使用されます
    vehicle_config: "experiment/configs/vehicles/default_vehicle.yaml"
    scene_config: "experiment/configs/scenes/default_scene.yaml"
```

### Training

```yaml
training:
  dataset_project: "e2e_aichallenge"
  dataset_scenario: "pure_pursuit"
  dataset_version: "v1.0"
  dataset_stage: "raw"
  epochs: 100
  # ...
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
