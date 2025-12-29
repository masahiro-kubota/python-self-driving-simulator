# Tiny LiDAR Net

ROS 2ベースのエンドツーエンド自動運転のための軽量なLiDAR処理モデルです。
1D CNNを使用してLiDARスキャンデータから直接制御コマンド（操舵角と加速度）を推論します。

## 特徴

- **軽量**: パラメータ数が少なく（約20万）、CPUでも高速に動作します。
- **純粋なNumPy推論**: ROS 2ノードでの推論にはPyTorchを使用せず、NumPyのみで実装された推論エンジンを使用するため、デプロイが容易です。
- **エンドツーエンド学習**: センサー入力から制御出力までを直接学習します。

## ディレクトリ構造

```
tiny_lidar_net/
├── src/tiny_lidar_net/      # ソースコード
│   ├── core.py              # 推論コアロジック（前処理、推論）
│   ├── node.py              # ROS 2ノード実装
│   └── model/               # NumPyモデル定義
├── scripts/                 # 学習・データ処理スクリプト
│   ├── extract_data_from_mcap.py
│   ├── train.py
│   └── convert_weight.py
├── data/                    # 学習データ（.npy）と重みファイル
├── ckpt/                    # PyTorchチェックポイント
└── tests/                   # ユニットテスト
```

## インストール

本パッケージは `uv` ワークスペースの一部として管理されています。
ルートディレクトリで以下を実行して依存関係をインストールしてください。

```bash
uv sync
```

学習機能を使用する場合は、追加の依存関係が必要です（`torch`, `mcap` 等）。これらは `pyproject.toml` に定義されており、`uv sync` で自動的にインストールされます。

## 使用方法

### 実験の実行

`experiment-runner` を使用して、Tiny LiDAR Netを使用したシミュレーション実験を実行できます。

```bash
uv run experiment-runner --config experiment/configs/experiments/tiny_lidar_net_experiment.yaml
```

設定ファイル (`include/tiny_lidar_net_module.yaml`) でモデルパスやパラメータを変更できます。

```yaml
params:
  model_path: "ad_components/control/tiny_lidar_net/data/tinylidarnet_weights.npy"
  input_dim: 1080
  architecture: "large"  # or "small"
```

## 学習パイプライン

独自のMCAPファイル（シミュレーションログなど）を使用してモデルを学習させる手順です。

### 1. データの抽出

MCAPファイルからLiDARスキャンと制御コマンドを抽出し、NumPy配列 (`.npy`) として保存します。
抽出されるデータは、LiDARのトピック `/sensing/lidar/scan` と 制御コマンドのトピック `/control/command/control_cmd` です。

```bash
# プロジェクトルートで実行
uv run python ad_components/control/tiny_lidar_net/scripts/extract_data_from_mcap.py \
  --mcap /path/to/your/simulation.mcap \
  --output ad_components/control/tiny_lidar_net/data/train
```

出力:
- `scans.npy`: LiDAR距離データ (N, 1080)
- `steers.npy`: ステアリング角 (N,)
- `accelerations.npy`: 加速度 (N,)

### 2. モデルの学習

PyTorchを使用してモデルを学習させます。

```bash
uv run python ad_components/control/tiny_lidar_net/scripts/train.py \
  --train-dir ad_components/control/tiny_lidar_net/data/train \
  --val-dir ad_components/control/tiny_lidar_net/data/val \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001 \
  --checkpoint-dir ad_components/control/tiny_lidar_net/checkpoints
```

ベストモデルは `checkpoints/best_model.pth` に保存されます。

### 3. 重みの変換

学習済みのPyTorchモデル (`.pth`) を、推論エンジンで使用可能なNumPy形式 (`.npy`) に変換します。

```bash
uv run python ad_components/control/tiny_lidar_net/scripts/convert_weight.py \
  --ckpt ad_components/control/tiny_lidar_net/checkpoints/best_model.pth \
  --output ad_components/control/tiny_lidar_net/data/tinylidarnet_weights.npy
```

変換された `.npy` ファイルパスを実験設定ファイルの `model_path` に指定することで、学習したモデルを使用できます。

## 開発者向け

### テストの実行

```bash
PYTHONPATH="" uv run pytest ad_components/control/tiny_lidar_net/tests/
```
