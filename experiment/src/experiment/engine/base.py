import os
from abc import ABC, abstractmethod
from typing import Any

from omegaconf import DictConfig, OmegaConf

import mlflow


class BaseEngine(ABC):
    """実験フェーズの基底クラス"""

    def run(self, cfg: DictConfig) -> Any:
        """エンジンを実行。MLflowのトラッキング設定が必須です (Fail-Fast)。"""
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise RuntimeError(
                "環境変数 'MLFLOW_TRACKING_URI' が設定されていません。実験の追跡を必須とするため実行を停止します。\n"
                "例: MLFLOW_TRACKING_URI=http://localhost:5000 uv run experiment-runner ..."
            )

        # MLflowの設定
        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = cfg.experiment.get("name", "e2e-playground")
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(
            run_name=f"{cfg.experiment.type}_{cfg.experiment.get('id', 'unnamed')}"
        ):
            # 基本タグの記録
            mlflow.set_tag("phase", cfg.experiment.type)
            if "id" in cfg.experiment:
                mlflow.set_tag("experiment_id", cfg.experiment.id)

            # 設定をアーティファクトとして保存
            mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")

            return self._run_impl(cfg)

    @abstractmethod
    def _run_impl(self, cfg: DictConfig) -> Any:
        """各エンジンでの具体的な処理内容

        Args:
            cfg: Hydra設定オブジェクト
        """
        pass
