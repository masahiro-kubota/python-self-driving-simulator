# E2E AI Challenge Playground

自動運転の認識・計画・制御コンポーネントを柔軟に組み合わせて実験できる、モジュラーな研究プラットフォームです。

---

## 🚀 クイックスタート

### 必要な環境

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) (パッケージマネージャー)
- Docker & Docker Compose (実験トラッキング用)

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

# 4. シミュレーションを実行
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
| [![Integration Tests](https://github.com/masahiro-kubota/e2e_aichallenge_playground/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/masahiro-kubota/e2e_aichallenge_playground/actions/workflows/integration-tests.yml) | 最新の統合テスト実行ステータス |
| [**Simulation Dashboard**](https://masahiro-kubota.github.io/e2e_aichallenge_playground/) | 最新のテスト結果（シミュレーションダッシュボード） |

---

## 📁 ディレクトリ構成

### アーキテクチャ方針

このプロジェクトは**プラグイン型モジュラーアーキテクチャ**を採用しています：

```
e2e_aichallenge_playground/
├── core/                           # コアフレームワーク
├── experiment/runner/              # 統一実験実行フレームワーク
├── simulators/                     # シミュレータ実装
├── dashboard/                      # シミュレーション可視化ダッシュボード
├── components_packages/            # コンポーネントパッケージ
│   ├── planning/                   # 計画コンポーネント
│   │   ├── pure_pursuit/
│   │   └── planning_utils/
│   └── control/                    # 制御コンポーネント
│       ├── pid/
│       └── neural_controller/
├── experiment/configs/             # 実験設定ファイル
│   └── experiments/                # 実験設定
│       ├── pure_pursuit.yaml
│       └── imitation_learning.yaml
├── data/                           # データ(.gitignore、MLflow/W&Bで管理)
└── mlflow/     # MLflow + MinIO サーバー
```

### 詳細構成

#### 📦 `core/` - コアフレームワーク
```
core/
├── pyproject.toml
└── src/core/
    ├── interfaces/              # 抽象インターフェース定義
    │   ├── perception.py       # 認識コンポーネントIF
    │   ├── planning.py         # 計画コンポーネントIF
    │   ├── control.py          # 制御コンポーネントIF
    │   └── simulator.py        # シミュレータIF
    ├── data/                    # データ構造定義
    │   ├── vehicle_state.py
    │   ├── observation.py
    │   ├── trajectory.py
    │   └── action.py
    └── utils/                   # 共通ユーティリティ
        ├── geometry.py
        ├── transforms.py
        └── config.py
```

**役割**: すべてのコンポーネントが従うべきインターフェースと共通データ構造を定義

**依存関係**: なし（最も基礎的なパッケージ）

#### 🎮 `simulators/` - シミュレータ実装
```
simulators/
├── pyproject.toml
└── src/simulators/
    └── simple_2d/              # 軽量2Dシミュレータ
        ├── simulator.py
        ├── vehicle.py
        ├── track.py
        └── obstacles.py
```

**役割**: 開発・学習用の軽量シミュレータ（ROS2不要）

**依存関係**: `core`

#### 🧩 `components_packages/` - 自動運転コンポーネント
```
components_packages/
├── planning/                   # 計画モジュール
│   ├── pure_pursuit/          # Pure Pursuit プランナー
│   └── planning_utils/        # トラックローダー等
└── control/                    # 制御モジュール
    ├── pid/                   # PID コントローラー
    └── neural_controller/     # ニューラルコントローラー
```

**役割**: 計画・制御の各コンポーネント実装（ルールベース・学習ベース）

**依存関係**: `core`

#### 🧪 `experiment/runner/` - 統一実験実行フレームワーク
```
experiment/runner/
├── pyproject.toml
├── src/experiment/runner/
│   ├── cli.py                 # CLIエントリーポイント
│   ├── config.py              # 設定管理
│   └── runner.py              # 実験実行ロジック
└── tests/                     # 統合テスト
```

**役割**: YAML設定ファイルで実験を定義・実行

**依存関係**: `core`, `simulators`, `dashboard`, コンポーネントパッケージ

#### 📊 `dashboard/` - シミュレーション可視化ダッシュボード

React/Viteベースのインタラクティブなダッシュボード。

```
dashboard/
├── src/                        # Reactコンポーネント
├── dist/                       # ビルド成果物
├── inject_data.py              # データ注入スクリプト
└── package.json
```

**役割**: シミュレーション結果の可視化（GitHub Pagesで公開）

**依存関係**: なし（独立したフロントエンドアプリ）


#### ⚙️ `experiment/configs/` - 実験設定ファイル

YAMLファイルで実験の再現性を保証。

```
experiment/configs/
├── experiments/                # 実験設定
│   ├── pure_pursuit.yaml
│   ├── pure_pursuit_dynamic.yaml
│   └── imitation_learning.yaml
└── current_experiment.yaml     # 現在の実験設定（自動生成）
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

```bash
# ユニットテストの実行
uv run pytest

# 統合テストの実行
uv run pytest experiment/runner/tests -m integration -v
```

### 開発用ツールのセットアップ

```bash
# 開発用依存関係（pre-commit等）をインストール
uv sync --extra dev
uv run pre-commit install
```

### コンポーネントの組み合わせ

設定ファイルでコンポーネントを自由に組み合わせ：

```yaml
# experiment/configs/experiments/custom.yaml
experiment:
  name: "custom_experiment"
  simulator: "simple_2d"

simulator:
  track_file: "data/tracks/raceline_awsim_1500.csv"

components:
  planning:
    type: "pure_pursuit"  # または "neural_planner"
    config:
      lookahead_distance: 5.0

  control:
    type: "pid"  # または "neural_controller"
    config:
      kp: 1.0
```
