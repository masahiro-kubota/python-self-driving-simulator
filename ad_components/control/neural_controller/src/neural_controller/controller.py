"""Neural Network Controller implementation."""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ad_component_core.data import Observation
from ad_component_core.interfaces import Controller

from core.data import Action, VehicleState
from core.data.ad_components import Trajectory
from core.utils.geometry import distance, normalize_angle


class MLP(nn.Module):
    """Simple MLP model."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64) -> None:
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


class NeuralController(Controller):
    """Neural Network based controller."""

    def __init__(
        self,
        model_path: str | Path,
        scaler_path: str | Path,
        input_size: int = 4,
        output_size: int = 2,
        hidden_size: int = 64,
    ) -> None:
        """Initialize Neural Controller.

        Args:
            model_path: Path to trained model weights (.pth)
            scaler_path: Path to scaler parameters (.json)
            input_size: Input dimension
            output_size: Output dimension
            hidden_size: Hidden layer size
        """
        self.device = torch.device("cpu")

        # Load model
        self.model = MLP(input_size, output_size, hidden_size).to(self.device)
        if Path(model_path).exists():
            print(f"Loading model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Warning: Model file not found at {model_path}")

        self.model.eval()

        # Load scaler
        if Path(scaler_path).exists():
            with open(scaler_path) as f:
                self.scaler_params = json.load(f)
        else:
            print(f"Warning: Scaler file not found at {scaler_path}")
            self.scaler_params = {}

        self.reference_trajectory: Trajectory | None = None

    def set_reference_trajectory(self, trajectory: Trajectory) -> None:
        """Set reference trajectory for error calculation.

        Args:
            trajectory: Reference trajectory
        """
        self.reference_trajectory = trajectory

    def calculate_errors(self, vehicle_state: VehicleState) -> tuple[float, float, float]:
        """Calculate lateral error, yaw error, and reference velocity.

        Args:
            vehicle_state: Current vehicle state

        Returns:
            Tuple of (lateral_error, yaw_error, ref_velocity)
        """
        if self.reference_trajectory is None or len(self.reference_trajectory) == 0:
            return 0.0, 0.0, 0.0

        # Find nearest point
        min_dist = float("inf")
        nearest_idx = 0

        # Simple search (can be optimized)
        for i, point in enumerate(self.reference_trajectory):
            d = distance(vehicle_state.x, vehicle_state.y, point.x, point.y)
            if d < min_dist:
                min_dist = d
                nearest_idx = i

        nearest_point = self.reference_trajectory[nearest_idx]

        # Calculate lateral error
        # Vector from nearest point to vehicle
        dx = vehicle_state.x - nearest_point.x
        dy = vehicle_state.y - nearest_point.y

        # Path direction vector
        path_yaw = nearest_point.yaw
        path_dx = math.cos(path_yaw)
        path_dy = math.sin(path_yaw)

        # Cross product to determine side (left/right)
        # e_lat = (vehicle - nearest) x path_dir
        # If > 0, vehicle is to the left (assuming standard coord system)
        e_lat = dx * path_dy - dy * path_dx

        # Calculate yaw error
        e_yaw = normalize_angle(vehicle_state.yaw - path_yaw)

        return e_lat, e_yaw, nearest_point.velocity

    def control(
        self,
        trajectory: Trajectory,
        vehicle_state: VehicleState,
        observation: Observation | None = None,
    ) -> Action:
        """Compute control action using NN.

        Args:
            trajectory: Target trajectory (unused, uses internal reference)
            vehicle_state: Current vehicle state
            observation: Current observation (unused)

        Returns:
            Control action
        """
        if self.reference_trajectory is None:
            return Action(steering=0.0, acceleration=0.0)

        # Calculate features
        e_lat, e_yaw, v_ref = self.calculate_errors(vehicle_state)
        v = vehicle_state.velocity

        # Prepare input [e_lat, e_yaw, v, v_ref]
        features = np.array([e_lat, e_yaw, v, v_ref], dtype=np.float32)

        # Normalize
        if self.scaler_params:
            mean = np.array(self.scaler_params["X_mean"])
            std = np.array(self.scaler_params["X_std"])
            features_norm = (features - mean) / std
        else:
            features_norm = features

        # Inference
        input_tensor = torch.from_numpy(features_norm).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        output_norm = output_tensor.cpu().numpy()[0]

        # Denormalize
        if self.scaler_params:
            y_mean = np.array(self.scaler_params["y_mean"])
            y_std = np.array(self.scaler_params["y_std"])
            output = output_norm * y_std + y_mean
        else:
            output = output_norm

        return Action(steering=float(output[0]), acceleration=float(output[1]))

    def reset(self) -> None:
        """Reset controller state."""
        pass
