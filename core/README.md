# Core Package

自動運転コンポーネントのコアフレームワーク。

## 概要

このパッケージは、自動運転システムの各コンポーネント（認識・計画・制御・シミュレータ）が従うべき抽象インターフェースと、共通のデータ構造を提供します。

## 構成

### Interfaces (`core.interfaces`)

各コンポーネントの抽象基底クラス：

- **`Perception`**: 認識コンポーネント
- **`Planner`**: 計画コンポーネント
- **`Controller`**: 制御コンポーネント
- **`Simulator`**: シミュレータ

### Data Structures (`core.data`)

共通のデータ構造：

#### AD Components (`core.data.ad_components`)
コンポーネント間で共有されるデータ型：
- **`VehicleState`**: 車両状態（位置、速度、姿勢など）
- **`Action`**: 制御指令（ステアリング、加速度）
- **`Trajectory`**: 軌道（経路点の列）
- **`TrajectoryPoint`**: 軌道上の1点
- **`Sensing`**: センサーデータ基底クラス
- **`ADComponentConfig`**: コンポーネント設定
- **`ADComponentLog`**: コンポーネントログ

#### Simulation (`core.data.simulation`)
シミュレーション関連：
- **`SimulationConfig`**: シミュレーション設定
- **`SimulationLog`**: シミュレーションログ
- **`SimulationResult`**: シミュレーション結果
- **`SimulationStep`**: シミュレーションステップ

#### Environment (`core.data.environment`)
環境関連：
- **`Scene`**: シーン定義
- **`Obstacle`**: 障害物
- **`TrackBoundary`**: トラック境界

#### Experiment (`core.data.experiment`)
実験関連：
- **`ExperimentConfig`**: 実験設定
- **`ExperimentResult`**: 実験結果

### Utilities (`core.utils`)

共通ユーティリティ関数：

#### Geometry (`core.utils.geometry`)
- `normalize_angle()`: 角度の正規化
- `distance()`: 2点間の距離
- `angle_between_points()`: 2点間の角度
- `rotate_point()`: 点の回転
- `nearest_point_on_line()`: 線分への最近点
- `curvature_from_points()`: 3点から曲率を計算

#### Transforms (`core.utils.transforms`)
- `global_to_local()`: グローバル座標→ローカル座標
- `local_to_global()`: ローカル座標→グローバル座標
- `transform_angle_to_local()`: 角度変換（グローバル→ローカル）
- `transform_angle_to_global()`: 角度変換（ローカル→グローバル）
- `rotation_matrix_2d()`: 2D回転行列
- `transformation_matrix_2d()`: 2D同次変換行列

#### Config (`core.utils.config`)
- `load_yaml()`: YAML設定ファイルの読み込み
- `save_yaml()`: YAML設定ファイルの保存
- `merge_configs()`: 設定のマージ
- `get_nested_value()`: ネストされた設定値の取得
- `set_nested_value()`: ネストされた設定値の設定

## 使用例

### インターフェースの実装

### インターフェースの実装

```python
from core.interfaces import Planner
from core.data.ad_components import Trajectory
from core.data import VehicleState
from ad_components_core.data import Observation

class MyPlanner(Planner):
    def plan(self, observation: Observation, vehicle_state: VehicleState) -> Trajectory:
        # 計画ロジックを実装
        pass

    def reset(self) -> None:
        # リセット処理
        pass
```

### データ構造の使用

```python
from core.data.ad_components import VehicleState, Action

# 車両状態の作成
state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=5.0)

# numpy配列に変換
state_array = state.to_array()  # [0.0, 0.0, 0.0, 5.0]

# 制御指令の作成
action = Action(steering=0.1, acceleration=1.0)
```

### ユーティリティの使用

```python
from core.utils import normalize_angle, global_to_local, load_yaml

# 角度の正規化
angle = normalize_angle(3.5)  # -2.78...

# 座標変換
local_x, local_y = global_to_local(10.0, 5.0, 0.0, 0.0, 0.5)

# 設定ファイルの読み込み
config = load_yaml("config.yaml")
```

## 依存関係

- `pydantic>=2.0.0`: データバリデーション
- `pyyaml>=6.0`: YAML設定ファイル
- `numpy>=2.3.5`: 数値計算

## 開発依存関係

- `ruff>=0.8.0`: リンター・フォーマッター
- `pyright>=1.1.0`: 型チェッカー
- `pytest>=8.0.0`: テストフレームワーク

## インストール

```bash
# ワークスペース全体をインストール
cd e2e_aichallenge_playground
uv sync

# 開発依存関係も含めてインストール
uv sync --extra dev

# coreパッケージのみインストール
cd core
uv pip install -e .
```

## 開発

### コード品質チェック

```bash
# リントチェック
uv run ruff check src/

# 自動修正
uv run ruff check --fix src/

# フォーマット
uv run ruff format src/

# 型チェック（strict mode）
uv run pyright src/
```

### テスト

```bash
# テスト実行（カバレッジ計測も自動で行われます）
uv run pytest
```

> [!IMPORTANT]
> ROS環境（`source /opt/ros/.../setup.bash` 等を実行済み）で実行する場合、環境変数の競合によりエラーが発生することがあります。
> その場合は `PYTHONPATH` を空にして実行してください：
>
> ```bash
> PYTHONPATH= uv run pytest
> ```

## コード品質

このパッケージは以下のツールで厳密にチェックされています：

- **Ruff**: 高速なPythonリンター・フォーマッター
  - pycodestyle, pyflakes, isort, flake8-annotations など多数のルールを適用
  - 行長: 100文字

- **Pyright**: 厳密な型チェック
  - `typeCheckingMode = "strict"`
  - すべての関数に型アノテーションが必要
  - 型の不整合を検出

すべてのコードは型安全で、スタイルガイドに準拠しています。
