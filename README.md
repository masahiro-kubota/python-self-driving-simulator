# E2E AI Challenge Playground

自動運転の認識・計画・制御コンポーネントを柔軟に組み合わせて実験できるプラットフォーム。

---

## 🚀 クイックスタート

### 必要な環境

- uv
- Docker

### セットアップと実行

```bash
# 1. リポジトリをクローン
git clone https://github.com/masahiro-kubota/e2e_aichallenge_playground.git
cd e2e_aichallenge_playground

# 2. 依存関係をインストール
uv sync

# 3. 実験トラッキングサーバーを起動（MLflow + MinIO）
cd mlflow
docker compose up -d
cd ..

# 4. 実験を実行
# 評価（Pure Pursuit）
uv run experiment-runner --config experiment/configs/experiments/pure_pursuit.yaml

# 5. 結果を確認
# MLflow UI: http://localhost:5000
# MinIO Console: http://localhost:9001 (minioadmin / minioadmin)
```

### サーバーの停止

```bash
cd mlflow
docker compose down  # データを保持
docker compose down -v  # データも削除
```

---

## 📊 CI/CD & Dashboard

| Status | Description |
| :--- | :--- |
| [![Unit Tests](https://github.com/masahiro-kubota/e2e_aichallenge_playground/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/masahiro-kubota/e2e_aichallenge_playground/actions/workflows/unit-tests.yml) | 単体テスト |
| [![Integration Tests](https://github.com/masahiro-kubota/e2e_aichallenge_playground/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/masahiro-kubota/e2e_aichallenge_playground/actions/workflows/integration-tests.yml) | 統合テスト |
| [![codecov](https://codecov.io/gh/masahiro-kubota/e2e_aichallenge_playground/branch/main/graph/badge.svg)](https://codecov.io/gh/masahiro-kubota/e2e_aichallenge_playground) | テストカバレッジ |
| [**Simulation Dashboard**](https://masahiro-kubota.github.io/e2e_aichallenge_playground/) | 最新のテスト結果（シミュレーションダッシュボード） |
| [**Sphinx Documentation**](https://masahiro-kubota.github.io/e2e_aichallenge_playground/docs/) | プロジェクトドキュメント (API Reference) |

---

## 📁 ディレクトリ構成

```
e2e_aichallenge_playground/
├── core/                           # プロジェクト基盤（データ構造・インターフェース）
├── ad_components/             # コンポーネントパッケージ
│   ├── core/                      # コンポーネント共通基盤
│   ├── planning/                  # 計画コンポーネント
│   │   ├── pure_pursuit/
│   │   └── planning_utils/
│   └── control/                   # 制御コンポーネント
│       ├── pid_controller/
│       └── neural_controller/
├── simulator/                     # シミュレータ実装
├── experiment/                    # 実験フレームワーク
│   ├── configs/                  # 実験設定ファイル
│   │   ├── experiments/          # 実験設定
│   │   ├── modules/              # モジュール設定(ADコンポーネント構成)
│   │   ├── scenes/               # シーン設定
│   │   ├── systems/              # システム設定(車両・シーン・モジュールの組み合わせ)
│   │   └── vehicles/             # 車両パラメータ
│   └── src/
│       └── experiment/           # 実験実行メインロジック
│           ├── runner/           # 実行エンジン
│           ├── postprocessing/   # 後処理 (評価・可視化)
│           └── preprocessing/    # 前処理 (Config解析)
├── dashboard/                    # 可視化ダッシュボード
├── supervisor/                   # シミュレーション監視・判定
├── logger/                       # ログ記録
├── data/                         # 一時データ(Git対象外)
└── mlflow/                       # MLflow + MinIO サーバー
```

### アーキテクチャ概要

> **Note**: この図は `scripts/generate_architecture_diagram.py` によって自動生成されています。更新する際は以下のコマンドを実行してください：
> ```bash
> uv run python scripts/generate_architecture_diagram.py
> ```

<!-- ARCHITECTURE_DIAGRAM_START -->
```mermaid
graph TD
    %% Styling
    classDef core fill:#f9f,stroke:#333,stroke-width:2px;
    classDef base fill:#fbb,stroke:#333,stroke-width:2px;
    classDef impl fill:#bbf,stroke:#333,stroke-width:2px;
    classDef app fill:#bfb,stroke:#333,stroke-width:2px;
    classDef default fill:#fff,stroke:#333,stroke-width:1px;
    subgraph group_other [other]
        logger["logger<br/>Logging node for rec.."]
        class logger impl;
        supervisor["supervisor<br/>Supervision and moni.."]
        class supervisor impl;
    end
    subgraph group_core [core]
        core["core<br/>Core data structures.."]
        class core core;
    end
    subgraph group_simulator [simulator]
        simulator["simulator<br/>Simulator package"]
        class simulator impl;
    end
    subgraph group_experiment [experiment]
        experiment["experiment<br/>Unified experiment e.."]
        class experiment impl;
    end
    subgraph group_dashboard [dashboard]
        dashboard["dashboard<br/>Interactive HTML das.."]
        class dashboard impl;
    end
    subgraph group_ad_components [ad_components]
        ad_component_core["ad-component-core<br/>Core interfaces and .."]
        class ad_component_core base;
        pure_pursuit["pure-pursuit<br/>Pure Pursuit path tr.."]
        class pure_pursuit impl;
        planning_utils["planning-utils<br/>Planning utilities"]
        class planning_utils impl;
        pid_controller["pid-controller<br/>PID controller"]
        class pid_controller impl;
    end
    %% Dependencies
    logger --> core
    simulator --> core
    experiment --> core
    experiment --> dashboard
    experiment --> logger
    experiment --> supervisor
    dashboard --> core
    supervisor --> core
    ad_component_core --> core
    pure_pursuit --> core
    pure_pursuit --> planning_utils
    planning_utils --> core
    pid_controller --> core
```
<!-- ARCHITECTURE_DIAGRAM_END -->

---

## 🏗️ アーキテクチャ詳細

### SingleProcessExecutor

このプロジェクトでは、**SingleProcessExecutor**を使用してシミュレーションを実行します。これは、複数のノード（Simulator、Sensor、Planning、Control、Supervisor、Logger）を単一プロセス内で協調動作させる実行エンジンです。

#### Nodeベースアーキテクチャ

すべてのコンポーネント（シミュレータ、センサー、プランナー、コントローラーなど）は`Node`基底クラスを継承し、統一されたインターフェースで実行されます：

```python
class Node(ABC, Generic[ConfigT]):
    """実行可能なノードの基底クラス"""

    def __init__(self, name: str, rate_hz: float, config: ConfigT):
        self.name = name
        self.rate_hz = rate_hz
        self.config = config  # Pydanticで検証済みの設定

    @abstractmethod
    def on_run(self, current_time: float) -> NodeExecutionResult:
        """ノードの実行ロジック"""
        pass
```

#### 設定ベースのノード構築

YAML設定ファイルから動的にノードを構築します。`loader.py`が設定を読み込み、`node_factory.create_node()`が各ノードをインスタンス化します：

```yaml
# experiment/configs/modules/pure_pursuit_pid.yaml
module:
  name: "pure_pursuit_pid"
  components:
    ad_component:
      params:
        nodes:
          - name: "Sensor"
            type: "experiment.processors.sensor.IdealSensorNode"
            params: {}
            rate_hz: 50.0

          - name: "Planning"
            type: "pure_pursuit.PurePursuitNode"
            params:
              min_lookahead_distance: 3.0
              max_lookahead_distance: 15.0
              lookahead_speed_ratio: 1.5
              track_path: "path/to/track.csv"
            rate_hz: 10.0

          - name: "Control"
            type: "pid_controller.PIDControllerNode"
            params:
              kp: 1.0
              ki: 0.1
              kd: 0.05
              u_min: -3.0
              u_max: 3.0
            rate_hz: 30.0
```

---

## 📖 開発フロー

### 基本的な実験実行

```bash
# Pure Pursuit コントローラーでシミュレーション
uv run experiment-runner --config experiment/configs/experiments/pure_pursuit.yaml

# Imitation Learning（ニューラルコントローラー）でシミュレーション
uv run experiment-runner --config experiment/configs/experiments/imitation_learning.yaml
```

### テストの実行

> **注意**: ROSがインストールされている環境では、`PYTHONPATH`環境変数にROSのパスが含まれているため、pytestが干渉を受けます。`PYTHONPATH=""`を付けてテストを実行してください。

```bash
# 全テストの実行
PYTHONPATH="" uv run pytest

# 統合テストの実行
PYTHONPATH="" uv run pytest -m integration -v

# 統合テストを除外（単体テストのみ）
PYTHONPATH="" uv run pytest -m "not integration"

# 特定のテストファイルを実行
PYTHONPATH="" uv run pytest core/tests/test_config.py -v

# Pre-commitフックの実行（全ファイル）
uv run pre-commit run --all-files
```
