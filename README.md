# E2E AI Challenge Playground

自動運転の認識・計画・制御コンポーネントを柔軟に組み合わせて実験できる、モジュラーな研究プラットフォームです。

## 🎯 プロジェクトの目的

このリポジトリは、自動運転の各コンポーネント（認識・計画・制御）をニューラルネットワークベースの手法で代替し、様々なアプローチを試すための**ROS2フリーな研究環境**を提供します。

### 主な特徴

- **プラグイン型アーキテクチャ**: コンポーネントを自由に組み合わせ可能
- **ROS2フリー**: 開発・学習・評価はROS2不要で高速イテレーション
- **拡張性**: 新しいシミュレータや手法を簡単に追加
- **再利用性**: 各パッケージは独立して他プロジェクトでも利用可能

---

## 📁 ディレクトリ構成

### アーキテクチャ方針

このプロジェクトは**プラグイン型モジュラーアーキテクチャ**を採用しています：

```
e2e_aichallenge_playground/
├── core/                           # コアフレームワーク
├── experiment_runner/              # 統一実験実行フレームワーク
├── simulators/                     # シミュレータ実装
├── tools/                          # 共通ツール
├── components_packages/            # コンポーネントパッケージ
│   ├── planning/                   # 計画コンポーネント
│   │   ├── pure_pursuit/
│   │   └── planning_utils/
│   └── control/                    # 制御コンポーネント
│       ├── pid/
│       └── neural_controller/
├── configs/                        # 設定ファイル
│   └── experiments/                # 実験設定
│       ├── pure_pursuit.yaml
│       └── imitation_learning.yaml
├── data/                           # データ（.gitignore、MLflow/W&Bで管理）
└── experiment-tracking-server/     # MLflow + MinIO サーバー
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

#### 🧪 `experiment_runner/` - 統一実験実行フレームワーク
```
experiment_runner/
├── pyproject.toml
├── src/experiment_runner/
│   ├── cli.py                 # CLIエントリーポイント
│   ├── config.py              # 設定管理
│   └── runner.py              # 実験実行ロジック
└── tests/                     # 統合テスト
```

**役割**: YAML設定ファイルで実験を定義・実行

**依存関係**: `core`, `simulators`, `tools`, コンポーネントパッケージ

#### 🛠️ `tools/` - 共通ツール

複数の実験で再利用可能なツール群。

```
tools/
├── data_collection/            # データ収集ツール
├── training/                   # 学習フレームワーク
├── evaluation/                 # 評価フレームワーク
└── visualization/              # 可視化ツール
```

#### ⚙️ `configs/` - 設定ファイル

YAMLファイルで実験の再現性を保証。

```
configs/
├── simulators/                 # シミュレータ設定
│   └── simple_2d.yaml
├── scenarios/                  # シナリオ定義
│   ├── static_track.yaml
│   └── dynamic_obstacles.yaml
└── pipelines/                  # パイプライン設定
    ├── full_nn.yaml           # 全てNN（E2E）
    ├── modular_nn.yaml        # モジュラーNN
    └── pure_pursuit.yaml      # ルールベース
```

---

## 🔗 ROS2連携アーキテクチャ

### リポジトリ分離戦略

このリポジトリは**ROS2フリー**に保ち、ROS2連携は別リポジトリで実装します。

```
📁 e2e_aichallenge_playground/     # このリポジトリ（ROS2フリー）
   ├── packages/                   # コアロジック
   ├── experiments/                # 実験コード
   └── tools/                      # 共通ツール

📁 aichallenge_ros2_wrapper/       # 別リポジトリ（ROS2専用）
   └── src/aichallenge_controller/ # ROS2パッケージ
       ├── launch/                 # launchファイル
       ├── nodes/                  # ROS2ノード
       ├── adapters/               # ROS2メッセージ変換
       └── package.xml
```

### ROS2ラッパーの役割

ROS2ラッパーリポジトリは以下を担当します：

1. **ROS2トピックとの通信**: センサーデータ受信、制御指令送信
2. **メッセージ変換**: ROS2メッセージ ↔ Pythonデータ構造
3. **Unityシミュレータ連携**: ROS2経由でUnityと通信
4. **コンポーネント統合**: このリポジトリのコンポーネントをインポートして使用

### 統合方法

ROS2ラッパーは`vcs import`でこのリポジトリを取得します：

```yaml
# aichallenge_ros2_wrapper/dependencies.repos
repositories:
  e2e_aichallenge_playground:
    type: git
    url: https://github.com/masahiro-kubota/e2e_aichallenge_playground.git
    version: main
```

セットアップ手順：

```bash
# ROS2ワークスペースを作成
mkdir -p ~/aichallenge_ws/src
cd ~/aichallenge_ws/src

# ROS2ラッパーをクローン
git clone https://github.com/yourusername/aichallenge_ros2_wrapper.git

# 依存リポジトリ（このリポジトリ）を取得
vcs import < aichallenge_ros2_wrapper/dependencies.repos

# Pythonパッケージをインストール
pip install -r aichallenge_ros2_wrapper/requirements.txt

# ROS2パッケージをビルド
cd ~/aichallenge_ws
colcon build --symlink-install
source install/setup.bash

# 実行
ros2 launch aichallenge_controller controller.launch.py
```

---

## 🚀 セットアップ

### 必要な環境

- Python >= 3.12
- uv (パッケージマネージャー)

### インストール

```bash
# リポジトリをクローン
git clone https://github.com/masahiro-kubota/e2e_aichallenge_playground.git
cd e2e_aichallenge_playground

# 依存関係をインストール
uv sync
```

---

## 📖 使用方法

### 開発フロー

#### 1. このリポジトリでの開発（ROS2不要）

```bash
cd e2e_aichallenge_playground
uv sync

# シミュレーションの実行
uv run experiment-runner --config configs/experiments/pure_pursuit.yaml

# 統合テストの実行
uv run pytest experiment_runner/tests -m integration -v
```

#### 2. ROS2ラッパーでの実行（Unityシミュレータ）

```bash
cd ~/aichallenge_ws
source install/setup.bash

# Unityシミュレータ + ROS2で実行
ros2 launch aichallenge_controller controller.launch.py
```

### コンポーネントの組み合わせ

設定ファイルでコンポーネントを自由に組み合わせ：

```yaml
# configs/pipelines/hybrid.yaml
pipeline:
  simulator: simple_2d
  perception: perfect           # ルールベース
  planning: transformer         # NNベース
  control: pid                  # ルールベース

simulator_config:
  dynamic_obstacles: true
  num_obstacles: 3
```

---

## 🎯 設計原則

### 1. インターフェース駆動設計

すべてのコンポーネントは抽象インターフェースを実装：

```python
from core.interfaces import PlanningComponent

class TransformerPlanner(PlanningComponent):
    def plan(self, observation, vehicle_state):
        # Transformer実装
        pass
```

### 2. ROS2からの完全分離

- このリポジトリにはROS2依存を一切含めない
- ROS2連携は別リポジトリで実装
- 開発・学習・評価は高速に実行可能

### 3. uvワークスペース管理

- 複数のパッケージを1つのリポジトリで管理
- 依存関係は`uv.lock`で統一管理
- 各パッケージは独立して再利用可能

---

## ✅ このアーキテクチャのメリット

1. **拡張性**: 新しいコンポーネントやシミュレータを簡単に追加
2. **再利用性**: コンポーネントを自由に組み合わせ
3. **高速開発**: ROS2なしで開発・デバッグ
4. **実験管理**: 設定ファイルで実験を再現可能
5. **テスト容易**: 純粋Pythonでユニットテスト可能

---

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

**Happy Autonomous Driving! 🚗💨**
