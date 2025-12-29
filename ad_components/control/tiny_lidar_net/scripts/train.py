"""Training script for Tiny LiDAR Net."""

import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
from lib.data import ScanControlDataset
from lib.loss import WeightedSmoothL1Loss
from lib.model import TinyLidarNet, TinyLidarNetSmall
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for scans, targets in tqdm(train_loader, desc="Training"):
        # Add channel dimension: (batch, length) -> (batch, 1, length)
        scans = scans.unsqueeze(1).to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(scans)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for scans, targets in tqdm(val_loader, desc="Validation"):
            scans = scans.unsqueeze(1).to(device)
            targets = targets.to(device)

            outputs = model(scans)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

    return total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description="Train Tiny LiDAR Net")
    parser.add_argument("--train-dir", type=Path, required=True, help="Training data directory")
    parser.add_argument("--val-dir", type=Path, required=True, help="Validation data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--model", type=str, default="large", choices=["large", "small"], help="Model architecture"
    )
    parser.add_argument("--input-dim", type=int, default=1080, help="Input dimension")
    parser.add_argument("--output-dim", type=int, default=2, help="Output dimension")
    parser.add_argument("--max-range", type=float, default=30.0, help="Maximum LiDAR range")
    parser.add_argument("--accel-weight", type=float, default=1.0, help="Acceleration loss weight")
    parser.add_argument("--steer-weight", type=float, default=1.0, help="Steering loss weight")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=10, help="Early stopping patience"
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create datasets
    train_dataset = ScanControlDataset(args.train_dir, max_range=args.max_range)
    val_dataset = ScanControlDataset(args.val_dir, max_range=args.max_range)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Create model
    if args.model == "small":
        model = TinyLidarNetSmall(input_dim=args.input_dim, output_dim=args.output_dim).to(device)
    else:
        model = TinyLidarNet(input_dim=args.input_dim, output_dim=args.output_dim).to(device)

    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create loss and optimizer
    criterion = WeightedSmoothL1Loss(accel_weight=args.accel_weight, steer_weight=args.steer_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create checkpoint directory
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Train Loss: {train_loss:.6f}")

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_loss:.6f}")

        # Save last model
        torch.save(model.state_dict(), args.checkpoint_dir / "last_model.pth")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.checkpoint_dir / "best_model.pth")
            logger.info(f"Best model saved with val loss: {val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.early_stop_patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
