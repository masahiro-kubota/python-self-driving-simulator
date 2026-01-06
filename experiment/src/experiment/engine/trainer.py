import logging
from pathlib import Path
from typing import Any

import hydra
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.optim as optim
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import wandb
from experiment.data.dataset import ScanControlDataset
from experiment.engine.base import BaseEngine
from experiment.engine.loss import WeightedHuberLoss
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

        # 統計量のロード (無効化: 推論時のMax-Min正規化に合わせるため)
        stats = None
        # stats_path = train_dir / "stats.json"
        # if stats_path.exists():
        #     with open(stats_path) as f:
        #         stats = json.load(f)
        #     logger.info(f"Loaded statistics from {stats_path}")
        logger.info("Using Max-Min normalization (stats=None) to match inference logic.")

        # Enable in-memory caching for maximum performance
        cache_to_ram = True
        train_dataset = ScanControlDataset(train_dir, stats=stats, cache_to_ram=cache_to_ram)
        val_dataset = ScanControlDataset(val_dir, stats=stats, cache_to_ram=cache_to_ram)

        num_workers = cfg.training.dataloader.get("num_workers", 4)
        pin_memory = cfg.training.dataloader.get("pin_memory", True)
        persistent_workers = cfg.training.dataloader.get("persistent_workers", True)

        _train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )
        _val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Config keys available: {list(cfg.keys())}")
        if "pretrained_model_path" in cfg:
            logger.info(f"pretrained_model_path in cfg: {cfg.pretrained_model_path}")
        else:
            logger.info("pretrained_model_path NOT in cfg")

        model = TinyLidarNet(input_dim=cfg.model.input_width, output_dim=2).to(device)

        if "pretrained_model_path" in cfg.training and cfg.training.pretrained_model_path:
            pretrained_path = cfg.training.pretrained_model_path
            logger.info(f"Loading pretrained weights from {pretrained_path}")
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
        else:
            logger.info("No pretrained model path found in cfg.training, training from scratch.")

        _optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
        _criterion = WeightedHuberLoss()

        # Output directory setup
        try:
            hydra_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
        except (ValueError, AttributeError):
            hydra_dir = Path("outputs/latest")
        
        output_dir = hydra_dir / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints will be saved to {output_dir}")

        best_val_loss = float("inf")

        logger.info(f"Starting training for {cfg.training.num_epochs} epochs on {device}")

        import psutil
        process = psutil.Process()

        from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

        # Profiler output setup
        prof_dir = hydra_dir / "profiler"
        prof_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Profiler logs will be saved to {prof_dir}")

        # Define profiler schedule: wait 1, warmup 1, active 3, repeat 1
        # This will profile steps 2,3,4.
        my_schedule = schedule(wait=1, warmup=1, active=3, repeat=1) 

        for epoch in range(cfg.training.num_epochs):
            model.train()
            train_loss = 0.0
            
            # Log epoch start
            logger.info(f"Starting Epoch {epoch + 1}/{cfg.training.num_epochs}")
            
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=my_schedule,
                on_trace_ready=tensorboard_trace_handler(str(prof_dir)),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                for batch_idx, (scans, targets) in enumerate(_train_loader):
                    prof.step() # Signal step start
                    
                    # Log progress every 100 batches
                    if batch_idx % 100 == 0:
                        mem_info = process.memory_info()
                        mem_gb = mem_info.rss / (1024 ** 3)
                        logger.info(f"[Epoch {epoch+1}] Batch {batch_idx}/{len(_train_loader)} processing... (RAM: {mem_gb:.2f} GB)")

                    with record_function("data_transfer"):
                         scans, targets = scans.to(device), targets.to(device)

                    _optimizer.zero_grad()
                    
                    with record_function("model_forward"):
                        # model expects (batch, 1, input_dim)
                        outputs = model(scans.unsqueeze(1))
                    
                    with record_function("loss_calc"):
                        loss = _criterion(outputs, targets)
                    
                    with record_function("backward_step"):
                        loss.backward()
                        _optimizer.step()

                    train_loss += loss.item()
                    

            avg_train_loss = train_loss / len(_train_loader) if len(_train_loader) > 0 else 0.0

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
                f"Epoch {epoch + 1}/{cfg.training.num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
            )

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                }
            )

            # Save Checkpoints
            # 1. Save last model (every epoch)
            last_model_path = output_dir / "last_model.pth"
            torch.save(model.state_dict(), last_model_path)

            # 2. Save best model (if val loss improved)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info(f"Validation Loss Improved! Saving best model... (Loss: {best_val_loss:.6f})")
                
                # Save PyTorch weights
                model_path = output_dir / "best_model.pth"
                torch.save(model.state_dict(), model_path)

                # Save Numpy weights (for simulation)
                numpy_path = output_dir / "best_model.npy"
                state_dict = model.state_dict()
                numpy_weights = {}
                for key, value in state_dict.items():
                    new_key = key.replace(".", "_")
                    numpy_weights[new_key] = value.cpu().numpy()
                np.save(numpy_path, numpy_weights)

                # Save ONNX (optional, but good to keep updated)
                # onnx_path = output_dir / "best_model.onnx"
                # dummy_input = torch.randn(1, 1, cfg.model.input_width).to(device)
                # torch.onnx.export(model, dummy_input, onnx_path)
                logger.info("Skipping ONNX export to prevent crash.")
        
        logger.info(f"Training completed. Best Val Loss: {best_val_loss:.6f}")
        logger.info(f"Models saved to {output_dir}")

        # Return path to best model
        return output_dir / "best_model.pth"

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
