"""Supervisor node for simulation judgment and monitoring."""

from core.data.node_io import NodeIO
from core.interfaces.node import Node, NodeConfig, NodeExecutionResult


class GoalConfig(NodeConfig):
    """Goal-related termination configuration."""

    enabled: bool = True
    x: float
    y: float
    radius: float
    min_elapsed_time: float = 0.0


class OffTrackConfig(NodeConfig):
    """Off-track termination configuration."""

    enabled: bool = True


class CollisionConfig(NodeConfig):
    """Collision termination configuration."""

    enabled: bool = True


class SupervisorConfig(NodeConfig):
    """Configuration for SupervisorNode."""

    goal: GoalConfig
    off_track: OffTrackConfig = OffTrackConfig()
    collision: CollisionConfig = CollisionConfig()


class SupervisorNode(Node[SupervisorConfig]):
    """Node responsible for supervising simulation success/failure and termination conditions."""

    def __init__(self, config: SupervisorConfig | dict, rate_hz: float):
        """Initialize SupervisorNode.

        Args:
            config: Validated configuration or configuration dict
            rate_hz: Evaluation rate [Hz]
        """
        # Convert dict to SupervisorConfig if needed
        if isinstance(config, dict):
            config = SupervisorConfig(**config)

        super().__init__("Supervisor", rate_hz, config)
        self.step_count = 0
        self.goal_count = 0
        self.is_in_goal = False

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
                "goal_count": int,
            },
        )

    def on_run(self, _current_time: float) -> NodeExecutionResult:
        """Evaluate simulation state and set termination flags.

        Args:
            _current_time: Current simulation time

        Returns:
            True if evaluation completed successfully
        """
        if self.frame_data is None:
            return NodeExecutionResult.FAILED

        # Skip if already terminated (and termination signal was set previously)
        if hasattr(self.frame_data, "termination_signal") and self.frame_data.termination_signal:
            return NodeExecutionResult.SUCCESS

        self.step_count += 1

        # Get current state
        sim_state = getattr(self.frame_data, "sim_state", None)
        if sim_state is None:
            return NodeExecutionResult.SKIPPED

        # Initialize outputs if not present
        self.frame_data.goal_count = self.goal_count

        # 1. Check off-track (collision with non-drivable area)
        if (
            hasattr(sim_state, "off_track")
            and sim_state.off_track
            and self.config.off_track.enabled
        ):
            self.frame_data.done = True
            self.frame_data.done_reason = "off_track"
            self.frame_data.success = False
            self.frame_data.termination_signal = True
            self.frame_data.termination_reason = "off_track"
            return NodeExecutionResult.SUCCESS

        # 2. Check collision with obstacles
        if (
            hasattr(sim_state, "collision")
            and sim_state.collision
            and self.config.collision.enabled
        ):
            self.frame_data.done = True
            self.frame_data.done_reason = "collision"
            self.frame_data.success = False
            self.frame_data.termination_signal = True
            self.frame_data.termination_reason = "collision"
            return NodeExecutionResult.SUCCESS

        # 3. Check goal reached
        dist = (
            (sim_state.x - self.config.goal.x) ** 2 + (sim_state.y - self.config.goal.y) ** 2
        ) ** 0.5
        elapsed_time = self.step_count * (1.0 / self.rate_hz)

        if dist <= self.config.goal.radius:
            if not self.is_in_goal and elapsed_time >= self.config.goal.min_elapsed_time:
                # Entered goal
                self.goal_count += 1
                self.is_in_goal = True
                self.frame_data.goal_count = self.goal_count

                # Check termination
                if self.config.goal.enabled:
                    self.frame_data.done = True
                    self.frame_data.done_reason = "goal_reached"
                    self.frame_data.success = True
                    self.frame_data.termination_signal = True
                    self.frame_data.termination_reason = "goal_reached"
                    return NodeExecutionResult.SUCCESS
        else:
            # Left goal area
            self.is_in_goal = False

        # No termination condition met
        if not hasattr(self.frame_data, "success"):
            self.frame_data.success = False
        self.frame_data.done = False
        self.frame_data.done_reason = ""
        self.frame_data.termination_signal = False
        self.frame_data.termination_reason = ""

        return NodeExecutionResult.SUCCESS
