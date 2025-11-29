# Experiment Tracking Server

MLflow + PostgreSQL + MinIO を使用した実験トラッキングサーバー。

## セットアップ

### 1. サーバーの起動

```bash
cd experiment-tracking-server
docker-compose up -d
```

### 2. アクセス

- **MLflow UI**: http://localhost:5000
- **MinIO Console**: http://localhost:9001 (minioadmin / minioadmin)

### 3. サーバーの停止

```bash
docker-compose down
```

データを削除する場合:
```bash
docker-compose down -v
```

## 実験からの使用方法

### Python スクリプトから

```python
import mlflow

# MLflow サーバーに接続
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    # パラメータ記録
    mlflow.log_params({"param1": value1})
    
    # メトリクス記録
    mlflow.log_metrics({"metric1": value1})
    
    # アーティファクト記録
    mlflow.log_artifact("path/to/file.mcap")
```

### 環境変数

実験スクリプト実行時に以下の環境変数を設定:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
```

## トラブルシューティング

### ポートが使用中

別のポートを使用する場合は `docker-compose.yml` を編集:

```yaml
ports:
  - "15000:5000"  # MLflow
  - "19000:9000"  # MinIO
```

### データのバックアップ

```bash
docker-compose exec postgres pg_dump -U mlflow mlflow > backup.sql
```
