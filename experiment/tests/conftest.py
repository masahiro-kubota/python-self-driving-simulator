import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_mlflow_env():
    """テスト実行中に MLflow 関連の環境変数と関数をモックします。"""
    with (
        patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://localhost:5000"}),
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run"),
        patch("mlflow.set_tag"),
        patch("mlflow.log_dict"),
        patch("mlflow.log_metric"),
        patch("mlflow.log_artifact"),
        patch("mlflow.active_run") as mock_run,
    ):
        # active_run().info.run_id を使う箇所があるため、モックを設定
        mock_run.return_value.info.run_id = "test_run_id"
        yield
