"""Supervisor node for simulation judgment and monitoring."""

from core.data.node_io import NodeIO
from core.interfaces.node import Node


class SupervisorNode(Node):
    """Node responsible for supervising simulation success/failure and termination conditions."""

    def __init__(
        self,
        goal_x: float | None = None,
        goal_y: float | None = None,
        goal_radius: float = 5.0,
        max_steps: int | None = None,
        min_elapsed_time: float = 20.0,
        rate_hz: float = 10.0,
    ):
        """Initialize SupervisorNode.

        Args:
            goal_x: Goal X coordinate [m]
            goal_y: Goal Y coordinate [m]
            goal_radius: Goal threshold [m]
            max_steps: Maximum steps before timeout
            min_elapsed_time: Minimum elapsed time [s] before goal check
            rate_hz: Evaluation rate [Hz]
        """
        super().__init__("Supervisor", rate_hz)
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.min_elapsed_time = min_elapsed_time
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
        if self.goal_x is not None and self.goal_y is not None:
            dist = ((sim_state.x - self.goal_x) ** 2 + (sim_state.y - self.goal_y) ** 2) ** 0.5
            elapsed_time = self.step_count * (1.0 / self.rate_hz)

            if dist < self.goal_radius and elapsed_time > self.min_elapsed_time:
                self.frame_data.done = True
                self.frame_data.done_reason = "goal_reached"
                self.frame_data.success = True
                self.frame_data.termination_signal = True
                self.frame_data.termination_reason = "goal_reached"
                return True

        # 3. Check timeout (max steps)
        if self.max_steps is not None and self.step_count >= self.max_steps:
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
