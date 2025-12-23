import json
import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.optim as optim
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import mlflow
import wandb
from experiment.data.dataset import ScanControlDataset
from experiment.engine.base import BaseEngine
from experiment.engine.loss import WeightedSmoothL1Loss
from experiment.models.tiny_lidar import TinyLidarNet

logger = logging.getLogger(__name__)


class TrainerEngine(BaseEngine):
    """学習エンジン"""

    def _run_impl(self, cfg: DictConfig) -> Any:
        # WandB初期化
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        # MLflowのRunIDをタグ付け
        wandb_config["mlflow_run_id"] = mlflow.active_run().info.run_id

        # データのDVCハッシュを取得して記録
        train_hash = self._get_dvc_hash(Path(cfg.train_data))
        val_hash = self._get_dvc_hash(Path(cfg.val_data))

        wandb_config["train_data_hash"] = train_hash
        wandb_config["val_data_hash"] = val_hash

        # MLflowにもパラメータとして記録
        mlflow.log_param("train_data_hash", train_hash)
        mlflow.log_param("val_data_hash", val_hash)

        wandb.init(
            project="e2e-playground",
            config=wandb_config,
        )

        train_dir = Path(cfg.train_data)
        val_dir = Path(cfg.val_data)

        # 統計量のロード (もしあれば)
        stats = None
        stats_path = train_dir / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            logger.info(f"Loaded statistics from {stats_path}")

        train_dataset = ScanControlDataset(train_dir, stats=stats)
        val_dataset = ScanControlDataset(val_dir, stats=stats)

        _train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
        _val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TinyLidarNet(input_dim=cfg.model.input_width, output_dim=2).to(device)

        _optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
        _criterion = WeightedSmoothL1Loss()

        logger.info(f"Starting training for {cfg.training.num_epochs} epochs on {device}")

        for epoch in range(cfg.training.num_epochs):
            model.train()
            train_loss = 0.0
            for batch_idx, (scans, targets) in enumerate(_train_loader):
                scans, targets = scans.to(device), targets.to(device)

                _optimizer.zero_grad()
                # model expects (batch, 1, input_dim)
                outputs = model(scans.unsqueeze(1))
                loss = _criterion(outputs, targets)
                loss.backward()
                _optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(_train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for scans, targets in _val_loader:
                    scans, targets = scans.to(device), targets.to(device)
                    # model expects (batch, 1, input_dim)
                    outputs = model(scans.unsqueeze(1))
                    loss = _criterion(outputs, targets)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(_val_loader) if len(_val_loader) > 0 else 0.0

            logger.info(
                f"Epoch {epoch+1}/{cfg.training.num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
            )

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                }
            )

        # Save the model
        try:
            hydra_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
        except (ValueError, AttributeError):
            hydra_dir = Path("outputs/latest")

        output_dir = hydra_dir / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. PyTorch weights
        model_path = output_dir / "best_model.pth"
        torch.save(model.state_dict(), model_path)

        # 2. Numpy weights (for simulation)
        numpy_path = output_dir / "best_model.npy"
        state_dict = model.state_dict()
        numpy_weights = {}
        for key, value in state_dict.items():
            new_key = key.replace(".", "_")
            numpy_weights[new_key] = value.cpu().numpy()
        np.save(numpy_path, numpy_weights)

        # 3. ONNX for deployment if needed
        onnx_path = output_dir / "best_model.onnx"
        dummy_input = torch.randn(1, 1, cfg.model.input_width).to(device)
        torch.onnx.export(model, dummy_input, onnx_path)

        logger.info(f"Training completed. Models saved to {output_dir}")

        # Log artifacts to MLflow
        mlflow.log_artifact(str(model_path), "models")
        mlflow.log_artifact(str(numpy_path), "models")
        mlflow.log_artifact(str(onnx_path), "models")

        wandb.finish()
        return model_path

    def _get_dvc_hash(self, data_dir: Path) -> str:
        """Get DVC hash from .dvc file associated with the data directory."""
        dvc_file = data_dir.with_suffix(".dvc")
        # また、ディレクトリ自体が .dvc ファイルである場合も考慮（通常は data_dir.dvc）
        # ただし data_dir が "data/train_set" の場合、探すのは "data/train_set.dvc"

        if not dvc_file.exists():
            return "unknown"

        try:
            with open(dvc_file) as f:
                dvc_data = yaml.safe_load(f)
                return dvc_data["outs"][0]["md5"]
        except Exception as e:
            logger.warning(f"Failed to read DVC hash from {dvc_file}: {e}")
            return "unknown"
