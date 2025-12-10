from core.data import Action
from core.data.node_io import NodeIO
from core.interfaces import Simulator
from core.interfaces.node import Node


class PhysicsNode(Node):
    """Node responsible for stepping the simulator physics."""

    def __init__(self, simulator: Simulator, rate_hz: float):
        """Initialize PhysicsNode.

        Args:
            simulator: Simulator instance
            rate_hz: Physics update rate [Hz]
        """
        super().__init__("Physics", rate_hz)
        self.simulator = simulator

    def get_node_io(self) -> NodeIO:
        """Define node IO."""
        from core.data import Action, VehicleState

        return NodeIO(
            inputs={
                "action": Action,
            },
            outputs={
                "sim_state": VehicleState,
            },
        )

    def on_run(self, _current_time: float) -> bool:
        """Execute physics simulation step.

        Args:
            _current_time: Current simulation time

        Returns:
            True if physics step completed successfully
        """
        if self.frame_data is None:
            return False

        # Skip if simulation is terminated
        if hasattr(self.frame_data, "termination_signal") and self.frame_data.termination_signal:
            return True

        # Use previous action or default
        action = getattr(self.frame_data, "action", None)
        if action is None:
            action = Action(steering=0.0, acceleration=0.0)

        # Step simulator (pure physics update)
        state, _done, _info = self.simulator.step(action)

        # Update ground truth state in FrameData
        self.frame_data.sim_state = state

        return True
