"""PyTorch model definitions for Tiny LiDAR Net."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class TinyLidarNet(nn.Module):
    """Standard CNN architecture for 1D LiDAR data processing.

    Architecture:
        Conv1d(1→24, k=10, s=4) + ReLU
        Conv1d(24→36, k=8, s=4) + ReLU
        Conv1d(36→48, k=4, s=2) + ReLU
        Conv1d(48→64, k=3, s=1) + ReLU
        Conv1d(64→64, k=3, s=1) + ReLU
        Flatten
        FC(flatten_dim→100) + ReLU
        FC(100→50) + ReLU
        FC(50→10) + ReLU
        FC(10→2) + Tanh
    """

    def __init__(self, input_dim: int = 1080, output_dim: int = 2):
        super().__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)

        # Calculate flatten dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy)))))
            self.flatten_dim = out.view(1, -1).shape[1]

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming Normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d | nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, input_dim)

        Returns:
            Output tensor of shape (batch, output_dim)
        """
        # Feature Extraction (Conv + ReLU)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Regression Head (FC + ReLU)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Output Layer
        x = torch.tanh(self.fc4(x))

        return x


class TinyLidarNetSmall(nn.Module):
    """Lightweight CNN architecture for 1D LiDAR data processing."""

    def __init__(self, input_dim: int = 1080, output_dim: int = 2):
        super().__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)

        # Calculate flatten dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            out = self.conv3(self.conv2(self.conv1(dummy)))
            self.flatten_dim = out.view(1, -1).shape[1]

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming Normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d | nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, input_dim)

        Returns:
            Output tensor of shape (batch, output_dim)
        """
        # Feature Extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Regression Head
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output Layer
        x = torch.tanh(self.fc3(x))

        return x
