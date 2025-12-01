"""Trainer implementation."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from components.control.neural_controller import NeuralController
from torch.utils.data import DataLoader, random_split

import mlflow
from core.data import Trajectory
from experiment_training.data.dataset import TrajectoryDataset


class Trainer:
    """Trainer for Neural Controller."""

    def __init__(
        self,
        config: dict[str, Any],
        reference_trajectory: Trajectory,
        workspace_root: Path,
    ) -> None:
        """Initialize trainer.

        Args:
            config: Training configuration
            reference_trajectory: Reference trajectory for feature calculation
            workspace_root: Root directory of the workspace
        """
        self.config = config
        self.reference_trajectory = reference_trajectory
        self.workspace_root = workspace_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model parameters
        self.input_size = config.get("input_size", 4)
        self.output_size = config.get("output_size", 2)
        self.hidden_size = config.get("hidden_size", 64)

        # Training parameters
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 100)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.validation_split = config.get("validation_split", 0.2)

        # Initialize model (using the one from components_packages)
        # We instantiate NeuralController to access its internal model structure
        # In a real scenario, we might want to separate the model definition
        controller = NeuralController(
            model_path="dummy",
            scaler_path="dummy",
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
        )
        self.model = controller.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, data_paths: list[str | Path]) -> None:
        """Execute training loop.

        Args:
            data_paths: List of paths to training data
        """
        print(f"Loading data from {len(data_paths)} files...")
        dataset = TrajectoryDataset(data_paths, self.reference_trajectory)

        if len(dataset) == 0:
            print("No training data found!")
            return

        # Split dataset
        val_size = int(len(dataset) * self.validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        print(f"Training on {train_size} samples, validating on {val_size} samples")
        print(f"Device: {self.device}")

        # MLflow logging
        mlflow.log_params(
            {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "train_samples": train_size,
                "val_samples": val_size,
            }
        )

        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for features, targets in train_loader:
                features, targets = features.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * features.size(0)

            train_loss /= train_size

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for features, targets in val_loader:
                    features, targets = features.to(self.device), targets.to(self.device)
                    outputs = self.model(features)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item() * features.size(0)

            val_loss /= val_size

            # Logging
            print(
                f"Epoch {epoch+1}/{self.epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model("best_model.pth")

        # Save final model
        self._save_model("final_model.pth")
        print("Training completed.")

    def _save_model(self, filename: str) -> None:
        """Save model to artifact directory."""
        # Save locally first
        local_path = Path("outputs") / filename
        local_path.parent.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), local_path)

        # Log to MLflow
        mlflow.log_artifact(str(local_path), artifact_path="models")
