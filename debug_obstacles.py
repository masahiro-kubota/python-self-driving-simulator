"""Debug script to check if obstacles are in simulation log."""

from pathlib import Path

from core.data import VehicleState
from experiment.preprocessing.loader import load_experiment_config
from simulator.simulator import Simulator, SimulatorConfig

# Load config
config_path = Path("experiment/configs/experiments/pure_pursuit.yaml")
config = load_experiment_config(config_path)

# Get simulator config
sim_config_dict = config.simulator.params

print("Simulator config obstacles:")
print(sim_config_dict.get("obstacles", "NOT FOUND"))

# Create simulator

initial_state = VehicleState(
    x=sim_config_dict.get("initial_state", {}).get("x", 0.0),
    y=sim_config_dict.get("initial_state", {}).get("y", 0.0),
    yaw=sim_config_dict.get("initial_state", {}).get("yaw", 0.0),
    velocity=sim_config_dict.get("initial_state", {}).get("velocity", 0.0),
    timestamp=0.0,
)

sim_config = SimulatorConfig(
    vehicle_params=sim_config_dict["vehicle_params"],
    initial_state=initial_state,
    obstacles=sim_config_dict.get("obstacles", []),
)

print("\nSimulatorConfig obstacles:")
print(sim_config.obstacles)

# Initialize simulator
simulator = Simulator(config=sim_config, rate_hz=100.0)
simulator.on_init()

print("\nSimulation log metadata:")
print(simulator.log.metadata)
