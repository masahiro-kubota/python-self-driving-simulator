# Self-Driving Simulator MLOps Pipeline Guide

このドキュメントでは、**Spatial-Temporal LidarNet** および **TinyLidarNet** の開発で使用された、データ収集から学習、評価までの完全なワークフローを解説します。

---

## 0. アーキテクチャ概要

| モデル | 特徴 | 用途 |
|---|---|---|
| **TinyLidarNet** | 単一フレーム入力 (Batch, 1, 1080) | 軽量、ベースライン |
| **Spatial-Temporal LidarNet** | 時系列入力 (Batch, 1, 20, 1080) | **推奨**: 高速域(10m/s)での安定性、遅延耐性 |

---

## 1. データ収集 (Data Collection)

シミュレーターを実行し、ランダムな初期位置からゴールを目指すエピソードを大量に収集します。

**コマンド (Train/Val収集):**

```bash
# Trainデータ収集 (例: 1000エピソード)
# total_episodes を指定すると、num_jobs (デフォルトは24、調整可能: execution.num_jobs=X) に応じて自動的に分割・並列実行されます。
# num_episodes は自動計算されるため、指定不要です。
uv run experiment-runner -m \
  experiment=data_collection_random_start \
  execution.total_episodes=1000 \
  execution.base_seed=0 \
  experiment.name=data_collection_train_v7

# Valデータ収集 (例: 200エピソード)
uv run experiment-runner -m \
  experiment=data_collection_random_start \
  execution.total_episodes=200 \
  execution.base_seed=2000 \
  experiment.name=data_collection_val_v7
```

- **出力先**: `outputs/<date>/<time>/...` (multirunの各ジョブが独立したディレクトリに出力)
- **所要時間**: 並列数によりますが、数時間程度。

**結果集計 (multirun完了後):**

データ収集が完了したら、以下のスクリプトで全エピソードの統計を集計できます：

```bash
# 最新のmultirun出力を自動検出
uv run python experiment/scripts/aggregate_multirun.py

# または明示的にディレクトリ指定
uv run python experiment/scripts/aggregate_multirun.py outputs/<date>/<time>
```

- **出力**: `collection_summary.json` (統計データ), `collection_summary.png` (可視化グラフ)
- **集計内容**: 成功率、失敗理由の内訳 (off_track/collision/timeout等)

---

## 2. 特徴量抽出 (Feature Extraction)

MCAPファイルからLiDARデータと操作量（ステアリング、アクセル）を抽出し、学習用の `.npy` 形式に変換・正規化します。

**スクリプト:** `scripts/prepare_fine_tuning_data_v3.py` (または同等の処理)

```bash
# Trainデータの処理 (timeoutのみを含む = off_track, collisionを除外)
uv run experiment-runner \
  experiment=extraction \
  input_dir=outputs/YYYY-MM-DD/HH-MM-SS/data_collection_train_v7 \
  output_dir=data/processed/train_v7 \
  exclude_failure_reasons=[off_track,collision,unknown]

# Valデータの処理
uv run experiment-runner \
  experiment=extraction \
  input_dir=outputs/YYYY-MM-DD/HH-MM-SS/data_collection_val_v7 \
  output_dir=data/processed/val_v7 \
  exclude_failure_reasons=[off_track,collision,unknown]
```

- **input_dir**: データ収集ステップの出力ディレクトリ (`outputs/日付/時刻/...`)
- **output_dir**: 加工済みデータの保存先
- **exclude_failure_reasons**: 除外する失敗理由のリスト
  - `null` (未定義): 失敗エピソード全除外
  - `[]` (空リスト): 全失敗を含む
  - `["off_track"]`: off_trackのみ除外、collisionは含む

- **出力**: `scans.npy` (入力), `steers.npy`, `accelerations.npy` (正解ラベル)

---

## 4. モデル学習 (Training)

作成したデータセットを使ってモデルを学習させます。

### A. TinyLidarNet
単一フレームモデルの学習です。

```bash
uv run experiment-runner -m \
  experiment=training \
  train_data=data/processed/train_v7 \
  val_data=data/processed/val_v7
```

**Note**: `experiment-runner` は学習完了時に自動的に `.npy` 形式の重みファイルも保存します (`outputs/latest/checkpoints/best_model.npy`)。手動での変換は不要です。

---

## 5. 評価 (Evaluation)

学習済みモデルをシミュレーター上で評価します。

### B. TinyLidarNet
```bash
# モデルパスとターゲット速度を指定可能
uv run experiment-runner experiment=evaluation \
    ad_components=tiny_lidar \
    ad_components.model_path=$(pwd)/checkpoints/tiny_lidar_net_v7_3000/best_model.npy \
    env=no_obstacle \
    ad_components.nodes.tiny_lidar_net.params.target_velocity=10.0
```

---

## 6. 結果の分析

評価実行後、出力された `simulation.mcap` を **Foxglove** で確認します。
ターミナルに表示されるリンクをクリックするか、ブラウザで `https://app.foxglove.dev` を開き、ローカルのMCAPファイルをロードしてください。

**確認ポイント**:
- **Trajectory**: 車両がコースを逸脱していないか
- **Steering**: ステアリング操作が滑らかか、発散していないか
- **Checkpoints**: 緑色のチェックポイントを通過できているか
