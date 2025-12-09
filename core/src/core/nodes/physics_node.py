from typing import Any

from core.data import Action, ADComponentLog
from core.interfaces import Simulator
from core.interfaces.node import Node


class PhysicsNode(Node):
    """Node responsible for stepping the simulator physics."""

    def __init__(self, simulator: Simulator, rate_hz: float, goal_radius: float = 5.0):
        super().__init__("Physics", rate_hz)
        self.simulator = simulator
        self.step_count = 0
        self.goal_radius = goal_radius

    def on_run(self, _current_time: float) -> bool:
        if self.context is None:
            # Context not ready
            return False

        if self.context.done:
            return True

        # Use previous action or default
        action = self.context.action or Action(steering=0.0, acceleration=0.0)

        # Step simulator
        state, done, info = self.simulator.step(action)
        self.step_count += 1

        # Inject detailed AD logs into the simulation step
        if hasattr(self.simulator, "log") and self.simulator.log.steps:
            step_log = self.simulator.log.steps[-1]

            # Create data dictionary
            data: dict[str, Any] = {}
            if self.context.trajectory:
                # We save the trajectory points
                data["trajectory"] = [
                    {"x": p.x, "y": p.y, "velocity": p.velocity}
                    for p in self.context.trajectory.points
                ]

            step_log.ad_component_log = ADComponentLog(component_type="split_nodes", data=data)

        # Update ground truth state
        self.context.sim_state = state

        # Check termination conditions
        # 1. Simulator native done (collision, etc)
        if done:
            self.context.done = True
            self.context.done_reason = "simulator_done"
            self.context.success = False
            return True

        # 2. Off track
        if state.off_track:
            self.context.done = True
            self.context.done_reason = "off_track"
            self.context.success = False
            return True

        # 3. Goal checking (if supported by simulator)
        if hasattr(self.simulator, "goal_x") and hasattr(self.simulator, "goal_y"):
            goal_x = getattr(self.simulator, "goal_x")
            goal_y = getattr(self.simulator, "goal_y")

            if goal_x is not None and goal_y is not None:
                dist = ((state.x - goal_x) ** 2 + (state.y - goal_y) ** 2) ** 0.5
                if dist < self.goal_radius:
                    self.context.done = True
                    self.context.done_reason = "goal_reached"
                    self.context.success = True
                    return True

        return True
