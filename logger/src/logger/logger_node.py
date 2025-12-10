"""Logger node for recording FrameData."""

from pathlib import Path
from typing import Any

from core.data import SimulationLog, SimulationStep
from core.data.node_io import NodeIO
from core.interfaces.node import Node, NodeConfig, NodeExecutionResult


class LoggerConfig(NodeConfig):
    """Configuration for LoggerNode (empty - no parameters needed)."""

    pass


class LoggerNode(Node[LoggerConfig]):
    """Node responsible for recording FrameData to simulation log."""

    def __init__(self, config: LoggerConfig = LoggerConfig(), rate_hz: float = 10.0):
        """Initialize LoggerNode.

        Args:
            config: Validated configuration
            rate_hz: Logging rate [Hz]
        """
        super().__init__("Logger", rate_hz, config)
        self.log = SimulationLog(steps=[], metadata={})
        self.current_time = 0.0

    def get_node_io(self) -> NodeIO:
        """Define node IO.

        Logger reads all available fields from FrameData but doesn't write anything.
        """
        return NodeIO(
            inputs={},  # Will read all fields dynamically
            outputs={},  # No outputs
        )

    def on_run(self, current_time: float) -> NodeExecutionResult:
        """Record current FrameData to log.

        Args:
            current_time: Current simulation time

        Returns:
            True if logging completed successfully
        """
        if self.frame_data is None:
            return NodeExecutionResult.FAILED

        self.current_time = current_time

        # Extract data from FrameData
        from core.data import Action, ADComponentLog, VehicleState

        # Get vehicle state
        sim_state = getattr(self.frame_data, "sim_state", None)
        if sim_state is None:
            sim_state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0, timestamp=current_time)

        # Get action
        action = getattr(self.frame_data, "action", None)
        if action is None:
            action = Action(steering=0.0, acceleration=0.0)

        # Create AD component log with trajectory data if available
        data: dict[str, Any] = {}
        trajectory = getattr(self.frame_data, "trajectory", None)
        if trajectory is not None and hasattr(trajectory, "points"):
            data["trajectory"] = [
                {"x": p.x, "y": p.y, "velocity": p.velocity} for p in trajectory.points
            ]

        ad_component_log = ADComponentLog(component_type="frame_data", data=data)

        # Create simulation step
        step = SimulationStep(
            timestamp=current_time,
            vehicle_state=sim_state,
            action=action,
            ad_component_log=ad_component_log,
            info={
                "goal_count": getattr(self.frame_data, "goal_count", 0),
            },
        )

        self.log.steps.append(step)
        return NodeExecutionResult.SUCCESS

    def get_log(self) -> SimulationLog:
        """Get the recorded simulation log.

        Returns:
            SimulationLog containing all recorded steps
        """
        return self.log

    def save_log(self, output_path: Path) -> Path:
        """Save log to file.

        Args:
            output_path: Path to save the log

        Returns:
            Path to the saved log file
        """
        from simulator.io import JsonSimulationLogRepository

        repo = JsonSimulationLogRepository()
        return repo.save(self.log, output_path)
