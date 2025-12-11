"""Debug script to check dashboard data generation."""

from datetime import datetime

from core.data import Action, VehicleState
from core.data.experiment import ExperimentResult
from core.data.simulator import SimulationLog, SimulationResult, SimulationStep
from dashboard.generator import HTMLDashboardGenerator

# Create a minimal simulation log with obstacles
log = SimulationLog(
    steps=[
        SimulationStep(
            timestamp=0.0,
            vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0, timestamp=0.0),
            action=Action(acceleration=0.0, steering=0.0),
        )
    ],
    metadata={
        "obstacles": [
            {
                "type": "static",
                "shape": {"type": "rectangle", "width": 2.0, "length": 4.0},
                "position": {"x": 10.0, "y": 5.0, "yaw": 0.0},
            }
        ]
    },
)

# Create experiment result
result = ExperimentResult(
    experiment_name="test",
    experiment_type="evaluation",
    execution_time=datetime.now(),
    simulation_results=[SimulationResult(log=log, success=True, done_reason="test")],
    metrics={},
)

# Generate dashboard
generator = HTMLDashboardGenerator()

# Check what data would be generated
metadata = log.metadata.copy()
print("Original metadata obstacles:")
print(metadata.get("obstacles"))

# Simulate sanitization
sanitized_metadata = {}
for k, v in metadata.items():
    if k == "obstacles":
        continue
    elif isinstance(v, dict | list):
        sanitized_metadata[k] = str(v)
    else:
        sanitized_metadata[k] = v

print("\nSanitized metadata (should NOT have obstacles):")
print(sanitized_metadata.get("obstacles", "NOT FOUND - CORRECT!"))

# Simulate data dict
data = {
    "metadata": {
        "experiment_name": "test",
        **sanitized_metadata,
    },
    "steps": [],
}

# Add obstacles
if "obstacles" in metadata and isinstance(metadata["obstacles"], list):
    data["obstacles"] = metadata["obstacles"]

print("\nFinal data obstacles:")
print(data.get("obstacles"))

print("\nFinal data metadata obstacles (should be NOT FOUND):")
print(data["metadata"].get("obstacles", "NOT FOUND - CORRECT!"))
