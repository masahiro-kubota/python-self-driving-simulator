"""Supervisor node for simulation judgment and monitoring."""

from core.data import (
    ComponentConfig,
    VehicleState,
)
from core.data.node_io import NodeIO
from core.interfaces.node import Node, NodeExecutionResult
from pydantic import Field


class GoalConfig(ComponentConfig):
    """Goal-related termination configuration."""

    enabled: bool
    x: float
    y: float
    radius: float
    min_elapsed_time: float


class CheckpointConfig(ComponentConfig):
    """Checkpoint configuration."""

    x: float
    y: float
    tolerance: float = 1.0


class OffTrackConfig(ComponentConfig):
    """Off-track termination configuration."""

    enabled: bool = Field(...)


class CollisionConfig(ComponentConfig):
    """Collision termination configuration."""

    enabled: bool = Field(...)


class SupervisorConfig(ComponentConfig):
    """Configuration for SupervisorNode."""

    goal: GoalConfig
    checkpoints: list[CheckpointConfig] = Field(default_factory=list)
    off_track: OffTrackConfig
    collision: CollisionConfig


class SupervisorNode(Node[SupervisorConfig]):
    """Node responsible for supervising simulation success/failure and termination conditions."""

    def __init__(self, config: SupervisorConfig, rate_hz: float, priority: int):
        """Initialize SupervisorNode.

        Args:
            config: Validated configuration
            rate_hz: Evaluation rate [Hz]
            priority: Execution priority
        """
        super().__init__("Supervisor", rate_hz, config, priority)
        self.step_count = 0
        self.goal_count = 0
        self.checkpoint_count = 0
        self.is_in_goal = False
        self.current_checkpoint_idx = 0

    def get_node_io(self) -> NodeIO:
        """Define node IO."""

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
                "checkpoint_count": int,
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
        if self.subscribe("termination_signal"):
            return NodeExecutionResult.SUCCESS

        self.step_count += 1

        # Get current state
        sim_state = self.subscribe("sim_state")
        if sim_state is None:
            return NodeExecutionResult.SKIPPED

        # Initialize outputs if not present
        self.publish("goal_count", self.goal_count)
        self.publish("checkpoint_count", self.checkpoint_count)

        # 1. Check off-track (collision with non-drivable area)
        if (
            hasattr(sim_state, "off_track")
            and sim_state.off_track
            and self.config.off_track.enabled
        ):
            self.publish("done", True)
            self.publish("done_reason", "off_track")
            self.publish("success", False)
            self.publish("termination_signal", True)
            self.publish("termination_reason", "off_track")
            return NodeExecutionResult.SUCCESS

        # 2. Check collision with obstacles
        if (
            hasattr(sim_state, "collision")
            and sim_state.collision
            and self.config.collision.enabled
        ):
            self.publish("done", True)
            self.publish("done_reason", "collision")
            self.publish("success", False)
            self.publish("termination_signal", True)
            self.publish("termination_reason", "collision")
            return NodeExecutionResult.SUCCESS

        # 3. Check checkpoints or goal reached
        if self.current_checkpoint_idx < len(self.config.checkpoints):
            # Checkpoint logic
            checkpoint = self.config.checkpoints[self.current_checkpoint_idx]
            dist = ((sim_state.x - checkpoint.x) ** 2 + (sim_state.y - checkpoint.y) ** 2) ** 0.5

            if dist <= checkpoint.tolerance:
                self.checkpoint_count += 1
                self.publish("checkpoint_count", self.checkpoint_count)
                self.current_checkpoint_idx += 1
                # Log checkpoint reached?
        else:
            # Final Goal logic
            dist = (
                (sim_state.x - self.config.goal.x) ** 2 + (sim_state.y - self.config.goal.y) ** 2
            ) ** 0.5
            elapsed_time = self.step_count * (1.0 / self.rate_hz)

            if dist <= self.config.goal.radius:
                if not self.is_in_goal and elapsed_time >= self.config.goal.min_elapsed_time:
                    # Entered goal
                    self.goal_count += 1
                    self.is_in_goal = True
                    self.publish("goal_count", self.goal_count)

                    # Check termination
                    if self.config.goal.enabled:
                        self.publish("done", True)
                        self.publish("done_reason", "goal_reached")
                        self.publish("success", True)
                        self.publish("termination_signal", True)
                        self.publish("termination_reason", "goal_reached")
                        return NodeExecutionResult.SUCCESS
            else:
                # Left goal area
                self.is_in_goal = False

        # No termination condition met
        if self.subscribe("success") is None:
            self.publish("success", False)

        self.publish("done", False)
        self.publish("done_reason", "")
        self.publish("termination_signal", False)
        self.publish("termination_reason", "")

        return NodeExecutionResult.SUCCESS
