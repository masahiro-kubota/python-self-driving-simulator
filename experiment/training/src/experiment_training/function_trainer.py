"""Simple trainer for function approximation."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import mlflow
from experiment_training.data.function_dataset import FunctionDataset


class SimpleMLP(nn.Module):
    """Simple MLP for function approximation."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FunctionTrainer:
    """Trainer for function approximation."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model parameters
        arch = config.get("architecture", {})
        self.input_size = arch.get("input_size", 1)
        self.hidden_size = arch.get("hidden_size", 32)
        self.output_size = arch.get("output_size", 1)

        # Training parameters
        self.batch_size = config.get("batch_size", 16)
        self.epochs = config.get("epochs", 200)
        self.learning_rate = config.get("learning_rate", 0.01)
        self.validation_split = config.get("validation_split", 0.2)

        # Initialize model
        self.model = SimpleMLP(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, data_path: str | Path) -> None:
        """Execute training loop.

        Args:
            data_path: Path to training data
        """
        print(f"Loading data from {data_path}...")
        dataset = FunctionDataset(data_path)

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
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * x.size(0)

            train_loss /= train_size

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                    val_loss += loss.item() * x.size(0)

            val_loss /= val_size

            # Logging
            if (epoch + 1) % 10 == 0:
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
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")

    def _save_model(self, filename: str) -> None:
        """Save model to artifact directory."""
        local_path = Path("outputs") / filename
        local_path.parent.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), local_path)
        mlflow.log_artifact(str(local_path), artifact_path="models")
