"""Loss functions for Tiny LiDAR Net training."""

import torch
import torch.nn as nn


class WeightedSmoothL1Loss(nn.Module):
    """Weighted sum of Smooth L1 losses for Acceleration and Steering.

    This loss function calculates the Smooth L1 loss independently for the
    acceleration and steering components, averages them across the batch,
    and then computes a weighted sum.
    """

    def __init__(self, accel_weight: float = 1.0, steer_weight: float = 1.0):
        """Initialize the WeightedSmoothL1Loss.

        Args:
            accel_weight: Coefficient to scale the acceleration loss
            steer_weight: Coefficient to scale the steering loss
        """
        super().__init__()
        self.accel_weight = accel_weight
        self.steer_weight = steer_weight

        # Use reduction='none' to calculate losses per element individually
        self.criterion = nn.SmoothL1Loss(reduction="none")

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate the weighted loss.

        Args:
            outputs: Model predictions. Shape: (Batch_Size, 2)
                     Index 0: Acceleration, Index 1: Steering
            targets: Ground truth values. Shape: (Batch_Size, 2)
                     Index 0: Acceleration, Index 1: Steering

        Returns:
            Scalar tensor representing the weighted combined loss
        """
        # Calculate element-wise loss
        loss = self.criterion(outputs, targets)

        # Separate losses based on channel index
        loss_accel = loss[:, 0]
        loss_steer = loss[:, 1]

        # Compute the weighted sum of means
        weighted_loss = (self.accel_weight * loss_accel.mean()) + (
            self.steer_weight * loss_steer.mean()
        )

        return weighted_loss
