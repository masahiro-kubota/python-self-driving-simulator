"""Training script."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from components.control.neural_controller import NeuralController
from core.data import SimulationLog
from torch.utils.data import DataLoader, TensorDataset
from track_loader import load_track_csv


def main() -> None:
    """Train neural controller."""
    # Paths
    workspace_root = Path(__file__).parent.parent.parent.parent
    experiment_root = workspace_root / "experiments/imitation_learning"
    data_dir = experiment_root / "data"
    raw_data_path = data_dir / "raw/log_pure_pursuit.json"
    model_dir = data_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data exists
    if not raw_data_path.exists():
        print(f"Error: Data file not found at {raw_data_path}")
        print("Please run 'uv run collect_data' first.")
        return

    # Load log
    print(f"Loading log from {raw_data_path}...")
    log = SimulationLog.load(raw_data_path)
    
    # Load track
    track_path = log.metadata.get("track")
    if not track_path:
        print("Error: Track path not found in log metadata")
        return
    
    print(f"Loading track from {track_path}...")
    track = load_track_csv(track_path)
    
    # Initialize controller for error calculation
    # We don't need a trained model here, just the logic
    controller = NeuralController(model_path="dummy", scaler_path="dummy")
    controller.set_reference_trajectory(track)
    
    # Prepare data
    print("Processing data...")
    X_list = []
    y_list = []
    
    for step in log.steps:
        # Inputs: e_lat, e_yaw, v, v_ref
        e_lat, e_yaw, v_ref = controller.calculate_errors(step.vehicle_state)
        v = step.vehicle_state.velocity
        
        X_list.append([e_lat, e_yaw, v, v_ref])
        
        # Outputs: steering, acceleration
        y_list.append([step.action.steering, step.action.acceleration])
        
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Normalize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0) + 1e-8
    
    # Save scaler params
    scaler_params = {
        "X_mean": X_mean.tolist(),
        "X_std": X_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
    }
    scaler_path = model_dir / "scaler.json"
    with open(scaler_path, "w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"Saved scaler params to {scaler_path}")
    
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std
    
    # Convert to tensors
    X_tensor = torch.from_numpy(X_norm)
    y_tensor = torch.from_numpy(y_norm)
    
    # Split
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    # Note: NeuralController has the model definition inside, but we can access it or recreate it.
    # Here we access the class from NeuralController's module or just use the one inside if we refactored.
    # Since MLP is not exported, we should probably export it or define it here.
    # For now, let's assume we can access it via NeuralController's instance or redefine it.
    # Better: Import MLP from neural_controller if possible, but it's not in __all__.
    # Let's redefine it here for simplicity or modify neural_controller.py to export it.
    # Modifying neural_controller.py is better.
    
    # Wait, NeuralController has .model attribute which is an instance of MLP.
    # We can train that instance directly!
    model = controller.model
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 50
    print("Starting training...")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
    # Save model
    model_path = model_dir / "nn_controller.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(data_dir / "training_loss.png")
    print(f"Saved loss plot to {data_dir / 'training_loss.png'}")


if __name__ == "__main__":
    main()
