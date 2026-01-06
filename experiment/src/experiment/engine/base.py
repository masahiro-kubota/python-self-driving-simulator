import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import mlflow
from omegaconf import DictConfig, OmegaConf


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

    def _load_env_file(self) -> None:
        """Load .env file manually if it exists."""
        try:
            # Find project root
            current_dir = Path(__file__).resolve().parent
            project_root = None
            for parent in [current_dir, *list(current_dir.parents)]:
                if (parent / ".env").exists():
                    project_root = parent
                    break
            
            if project_root:
                env_path = project_root / ".env"
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            key, value = line.split("=", 1)
                            # Only set if not already set (env vars take precedence)
                            if key.strip() == "MCAP_BASE_URL" and "MCAP_BASE_URL" not in os.environ:
                                os.environ["MCAP_BASE_URL"] = value.strip()
        except Exception:
            pass

    def _get_foxglove_url(self, mcap_path: Path) -> Optional[str]:
        """Generate Foxglove URL for the given MCAP file."""
        try:
            import urllib.parse
            
            # Ensure env is loaded
            self._load_env_file()

            # Find project root by looking for uv.lock or .git
            current_dir = Path(__file__).resolve().parent
            project_root = None
            for parent in [current_dir, *list(current_dir.parents)]:
                if (parent / "uv.lock").exists() or (parent / ".git").exists():
                    project_root = parent
                    break

            if project_root:
                rel_mcap_path = mcap_path.resolve().relative_to(project_root.resolve())
                
                # Use MCAP_BASE_URL from env
                base_url = os.getenv("MCAP_BASE_URL")
                
                if base_url:
                    # Strip trailing slash if present
                    base_url = base_url.rstrip("/")
                    mcap_url = f"{base_url}/{rel_mcap_path}"
                else:
                    # Fallback to local default logic
                    # Only used if MCAP_BASE_URL is missing
                    host = os.getenv("FOXGLOVE_HOST_IP", "127.0.0.1")
                    if "ts.net" in host:
                        mcap_url = f"https://{host}/{rel_mcap_path}"
                    else:
                        mcap_url = f"http://{host}:8080/{rel_mcap_path}"

                encoded_url = urllib.parse.quote(mcap_url, safe="")
                return f"https://app.foxglove.dev/view?ds=remote-file&ds.url={encoded_url}"
        except Exception:
            pass
        return None
