"""Simple script to generate training data for function approximation."""

import json
from pathlib import Path

import numpy as np

# Create output directory
output_dir = Path("data/raw/function_approx")
output_dir.mkdir(parents=True, exist_ok=True)

# Generate data for sin(x)
np.random.seed(42)
n_samples = 1000

# Generate x values in range [-2π, 2π]
x = np.random.uniform(-2 * np.pi, 2 * np.pi, n_samples)
y = np.sin(x)

# Save as simple JSON format
data = {"x": x.tolist(), "y": y.tolist()}

output_path = output_dir / "sin_data.json"
with open(output_path, "w") as f:
    json.dump(data, f)

print(f"Generated {n_samples} samples for sin(x) function")
print(f"Saved to: {output_path}")
print(f"x range: [{x.min():.2f}, {x.max():.2f}]")
print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
