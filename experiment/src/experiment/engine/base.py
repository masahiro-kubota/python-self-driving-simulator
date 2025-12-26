import os
import subprocess
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
            mlflow.set_tag("git_commit", self._get_git_commit())
            if "id" in cfg.experiment:
                mlflow.set_tag("experiment_id", cfg.experiment.id)

            # 設定をアーティファクトとして保存
            container = OmegaConf.to_container(cfg, resolve=True)
            if isinstance(container, dict):
                # パラメータをフラット化して記録
                flat_params = self._flatten_config(container)
                # MLflowの制限（100パラメータ/回）を考慮して分割登録はMLflow clientがやる場合もあるが
                # ここでは単純に渡す（大量にある場合はbatch loggingが必要だが一旦そのまま）
                for i in range(0, len(flat_params), 100):
                    chunk = dict(list(flat_params.items())[i : i + 100])
                    mlflow.log_params(chunk)

            try:
                mlflow.log_dict(container, "config.yaml")
            except Exception as e:
                # Fallback if storage is full
                print(f"Warning: Failed to upload config.yaml to MLflow: {e}")

            return self._run_impl(cfg)

    def _flatten_config(
        self, d: dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> dict[str, Any]:
        """設定辞書をフラット化 (dot-notation)"""
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_config(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _get_git_commit(self) -> str:
        """現在のGitコミットハッシュを取得"""
        try:
            commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
            return commit
        except Exception:
            return "unknown"

    @abstractmethod
    def _run_impl(self, cfg: DictConfig) -> Any:
        """各エンジンでの具体的な処理内容

        Args:
            cfg: Hydra設定オブジェクト
        """
        pass
