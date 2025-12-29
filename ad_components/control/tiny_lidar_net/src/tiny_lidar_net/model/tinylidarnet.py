# ============================================================
# NumPy Inference Models (Exact Naming Match with PyTorch)
# ============================================================


# Import NumPy layer functions
from . import (
    conv1d,
    flatten,
    kaiming_normal_init,
    linear,
    relu,
    tanh,
    zeros_init,
)


class TinyLidarNetNp:
    """NumPy implementation of TinyLidarNet (Conv5 + FC4).

    This class provides a pure NumPy inference implementation that matches the
    architecture of the PyTorch `TinyLidarNet` class.

    Attributes:
        params (dict): Stores weights and biases for all layers.
        strides (dict): Stores stride values for convolutional layers.
        shapes (dict): Stores parameter shapes for initialization.
    """

    def __init__(self, input_dim=1080, output_dim=2):
        """Initializes TinyLidarNetNp.

        Args:
            input_dim (int): The size of the input LiDAR scan array. Defaults to 1080.
            output_dim (int): The size of the output prediction. Defaults to 2.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {}

        # Stride definitions
        self.strides = {"conv1": 4, "conv2": 4, "conv3": 2, "conv4": 1, "conv5": 1}

        # Shape definitions matching PyTorch
        self.shapes = {
            "conv1_weight": (24, 1, 10),
            "conv1_bias": (24,),
            "conv2_weight": (36, 24, 8),
            "conv2_bias": (36,),
            "conv3_weight": (48, 36, 4),
            "conv3_bias": (48,),
            "conv4_weight": (64, 48, 3),
            "conv4_bias": (64,),
            "conv5_weight": (64, 64, 3),
            "conv5_bias": (64,),
        }

        flatten_dim = self._get_conv_output_dim()
        self.shapes.update(
            {
                "fc1_weight": (100, flatten_dim),
                "fc1_bias": (100,),
                "fc2_weight": (50, 100),
                "fc2_bias": (50,),
                "fc3_weight": (10, 50),
                "fc3_bias": (10,),
                "fc4_weight": (output_dim, 10),
                "fc4_bias": (output_dim,),
            }
        )

        self._initialize_weights()

    def _get_conv_output_dim(self):
        """Calculates the flattened dimension after the last convolution layer."""
        length = self.input_dim
        for i in range(1, 6):
            k = self.shapes[f"conv{i}_weight"][2]
            s = self.strides[f"conv{i}"]
            length = (length - k) // s + 1
        c = self.shapes["conv5_weight"][0]
        return c * length

    def _initialize_weights(self):
        """Initializes weights using Kaiming Normal (fan_out) and biases to zero."""
        for name, shape in self.shapes.items():
            if name.endswith("_weight"):
                fan_out = shape[0] * (shape[2] if "conv" in name else 1)
                self.params[name] = kaiming_normal_init(shape, fan_out)
            elif name.endswith("_bias"):
                self.params[name] = zeros_init(shape)

    def __call__(self, x):
        """Performs the forward pass of the model.

        Args:
            x (np.ndarray): Input array of shape (batch_size, 1, input_dim).

        Returns:
            np.ndarray: Output array of shape (batch_size, output_dim).
        """
        x = relu(
            conv1d(x, self.params["conv1_weight"], self.params["conv1_bias"], self.strides["conv1"])
        )
        x = relu(
            conv1d(x, self.params["conv2_weight"], self.params["conv2_bias"], self.strides["conv2"])
        )
        x = relu(
            conv1d(x, self.params["conv3_weight"], self.params["conv3_bias"], self.strides["conv3"])
        )
        x = relu(
            conv1d(x, self.params["conv4_weight"], self.params["conv4_bias"], self.strides["conv4"])
        )
        x = relu(
            conv1d(x, self.params["conv5_weight"], self.params["conv5_bias"], self.strides["conv5"])
        )
        x = flatten(x)
        x = relu(linear(x, self.params["fc1_weight"], self.params["fc1_bias"]))
        x = relu(linear(x, self.params["fc2_weight"], self.params["fc2_bias"]))
        x = relu(linear(x, self.params["fc3_weight"], self.params["fc3_bias"]))
        return tanh(linear(x, self.params["fc4_weight"], self.params["fc4_bias"]))


class TinyLidarNetSmallNp:
    """NumPy implementation of TinyLidarNetSmall (Conv3 + FC3).

    This class provides a pure NumPy inference implementation that matches the
    architecture of the PyTorch `TinyLidarNetSmall` class.
    """

    def __init__(self, input_dim=1080, output_dim=2):
        """Initializes TinyLidarNetSmallNp.

        Args:
            input_dim (int): The size of the input LiDAR scan array. Defaults to 1080.
            output_dim (int): The size of the output prediction. Defaults to 2.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {}
        self.strides = {"conv1": 4, "conv2": 4, "conv3": 2}

        self.shapes = {
            "conv1_weight": (24, 1, 10),
            "conv1_bias": (24,),
            "conv2_weight": (36, 24, 8),
            "conv2_bias": (36,),
            "conv3_weight": (48, 36, 4),
            "conv3_bias": (48,),
        }

        flatten_dim = self._get_conv_output_dim()
        self.shapes.update(
            {
                "fc1_weight": (100, flatten_dim),
                "fc1_bias": (100,),
                "fc2_weight": (50, 100),
                "fc2_bias": (50,),
                "fc3_weight": (output_dim, 50),
                "fc3_bias": (output_dim,),
            }
        )

        self._initialize_weights()

    def _get_conv_output_dim(self):
        """Calculates the flattened dimension after the last convolution layer."""
        length = self.input_dim
        for i in range(1, 4):
            k = self.shapes[f"conv{i}_weight"][2]
            s = self.strides[f"conv{i}"]
            length = (length - k) // s + 1
        c = self.shapes["conv3_weight"][0]
        return c * length

    def _initialize_weights(self):
        """Initializes weights using Kaiming Normal (fan_out) and biases to zero."""
        for name, shape in self.shapes.items():
            if name.endswith("_weight"):
                fan_out = shape[0] * (shape[2] if "conv" in name else 1)
                self.params[name] = kaiming_normal_init(shape, fan_out)
            elif name.endswith("_bias"):
                self.params[name] = zeros_init(shape)

    def __call__(self, x):
        """Performs the forward pass of the model.

        Args:
            x (np.ndarray): Input array of shape (batch_size, 1, input_dim).

        Returns:
            np.ndarray: Output array of shape (batch_size, output_dim).
        """
        x = relu(
            conv1d(x, self.params["conv1_weight"], self.params["conv1_bias"], self.strides["conv1"])
        )
        x = relu(
            conv1d(x, self.params["conv2_weight"], self.params["conv2_bias"], self.strides["conv2"])
        )
        x = relu(
            conv1d(x, self.params["conv3_weight"], self.params["conv3_bias"], self.strides["conv3"])
        )
        x = flatten(x)
        x = relu(linear(x, self.params["fc1_weight"], self.params["fc1_bias"]))
        x = relu(linear(x, self.params["fc2_weight"], self.params["fc2_bias"]))
        return tanh(linear(x, self.params["fc3_weight"], self.params["fc3_bias"]))
