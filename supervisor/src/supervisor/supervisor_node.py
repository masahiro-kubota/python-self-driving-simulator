"""Supervisor node for simulation judgment and monitoring."""

from core.data.node_io import NodeIO
from core.interfaces.node import Node, NodeConfig


class SupervisorConfig(NodeConfig):
    """Configuration for SupervisorNode."""

    goal_x: float = 0.0
    goal_y: float = 0.0
    goal_radius: float = 5.0
    max_steps: int = 1000
    min_elapsed_time: float = 20.0


class SupervisorNode(Node[SupervisorConfig]):
    """Node responsible for supervising simulation success/failure and termination conditions."""

    def __init__(self, config: dict, rate_hz: float):
        """Initialize SupervisorNode.

        Args:
            config: Configuration dictionary
            rate_hz: Evaluation rate [Hz]
        """
        super().__init__("Supervisor", rate_hz, config, config_model=SupervisorConfig)
        self.step_count = 0

    def get_node_io(self) -> NodeIO:
        """Define node IO."""
        from core.data import VehicleState

        return NodeIO(
            inputs={
                "sim_state": VehicleState,
            },
            outputs={
                "success": bool,
                "done": bool,
                "done_reason": str,
                "termination_signal": bool,
                "termination_reason": str,
            },
        )

    def on_run(self, _current_time: float) -> bool:
        """Evaluate simulation state and set termination flags.

        Args:
            _current_time: Current simulation time

        Returns:
            True if evaluation completed successfully
        """
        if self.frame_data is None:
            return False

        # Skip if already terminated
        if self.frame_data.termination_signal:
            return True

        self.step_count += 1

        # Get current state
        sim_state = getattr(self.frame_data, "sim_state", None)
        if sim_state is None:
            return True

        # 1. Check off-track (collision with non-drivable area)
        if hasattr(sim_state, "off_track") and sim_state.off_track:
            self.frame_data.done = True
            self.frame_data.done_reason = "off_track"
            self.frame_data.success = False
            self.frame_data.termination_signal = True
            self.frame_data.termination_reason = "off_track"
            return True

        # 2. Check goal reached
        dist = (
            (sim_state.x - self.config.goal_x) ** 2 + (sim_state.y - self.config.goal_y) ** 2
        ) ** 0.5
        elapsed_time = self.step_count * (1.0 / self.rate_hz)

        if dist <= self.config.goal_radius and elapsed_time >= self.config.min_elapsed_time:
            self.frame_data.done = True
            self.frame_data.done_reason = "goal_reached"
            self.frame_data.success = True
            self.frame_data.termination_signal = True
            self.frame_data.termination_reason = "goal_reached"
            return True

        # 3. Check timeout (max steps)
        if self.step_count >= self.config.max_steps:
            self.frame_data.done = True
            self.frame_data.done_reason = "timeout"
            self.frame_data.success = False
            self.frame_data.termination_signal = True
            self.frame_data.termination_reason = "timeout"
            return True

        # No termination condition met
        self.frame_data.success = False
        self.frame_data.done = False
        self.frame_data.done_reason = ""
        self.frame_data.termination_signal = False
        self.frame_data.termination_reason = ""

        return True
