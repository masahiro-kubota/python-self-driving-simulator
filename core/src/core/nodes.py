"""Concrete Node implementations."""

from typing import Any

from core.data import Action, ADComponentLog, Observation
from core.interfaces import Controller, Planner, Simulator
from core.interfaces.node import Node, SimulationContext


class PhysicsNode(Node):
    """Node responsible for stepping the simulator physics."""

    def __init__(self, simulator: Simulator, rate_hz: float):
        super().__init__("Physics", rate_hz)
        self.simulator = simulator
        self.step_count = 0

    def on_run(self, context: SimulationContext) -> None:
        if context.done:
            return

        # Use previous action or default
        action = context.action or Action(steering=0.0, acceleration=0.0)

        # Step simulator
        state, done, info = self.simulator.step(action)
        self.step_count += 1

        # Inject detailed AD logs into the simulation step
        if hasattr(self.simulator, "log") and self.simulator.log.steps:
            step_log = self.simulator.log.steps[-1]

            # Create data dictionary
            data: dict[str, Any] = {}
            if context.trajectory:
                # We save the trajectory points
                data["trajectory"] = [
                    {"x": p.x, "y": p.y, "velocity": p.velocity} for p in context.trajectory.points
                ]

            step_log.ad_component_log = ADComponentLog(component_type="split_nodes", data=data)

        # Update ground truth state
        context.sim_state = state

        # Check termination conditions
        # 1. Simulator native done (collision, etc)
        if done:
            context.done = True
            context.done_reason = "simulator_done"
            context.success = False
            return

        # 2. Off track
        if state.off_track:
            context.done = True
            context.done_reason = "off_track"
            context.success = False
            return

        # 3. Goal checking (if supported by simulator)
        if hasattr(self.simulator, "goal_x") and hasattr(self.simulator, "goal_y"):
            goal_x = getattr(self.simulator, "goal_x")
            goal_y = getattr(self.simulator, "goal_y")

            if goal_x is not None and goal_y is not None:
                dist = ((state.x - goal_x) ** 2 + (state.y - goal_y) ** 2) ** 0.5
                if dist < 5.0:  # Hardcoded 5.0m threshold
                    context.done = True
                    context.done_reason = "goal_reached"
                    context.success = True


class SensorNode(Node):
    """Node responsible for perception/sensing."""

    def __init__(self, rate_hz: float):
        super().__init__("Sensor", rate_hz)

    def on_run(self, context: SimulationContext) -> None:
        if context.sim_state:
            context.vehicle_state = context.sim_state
            context.observation = Observation(
                lateral_error=0.0,
                heading_error=0.0,
                velocity=context.vehicle_state.velocity,
                target_velocity=0.0,
                timestamp=context.current_time,
            )


class PlanningNode(Node):
    """Node responsible for path planning."""

    def __init__(self, planner: Planner, rate_hz: float):
        super().__init__("Planning", rate_hz)
        self.planner = planner

    def on_run(self, context: SimulationContext) -> None:
        if context.vehicle_state:
            observation = context.observation or Observation(
                lateral_error=0.0,
                heading_error=0.0,
                velocity=context.vehicle_state.velocity,
                target_velocity=0.0,
                timestamp=context.current_time,
            )
            # Ensure observation has valid values if defaults were used
            if observation.timestamp is None:
                observation.timestamp = context.current_time

            context.trajectory = self.planner.plan(observation, context.vehicle_state)


class ControlNode(Node):
    """Node responsible for vehicle control."""

    def __init__(self, controller: Controller, rate_hz: float):
        super().__init__("Control", rate_hz)
        self.controller = controller

    def on_run(self, context: SimulationContext) -> None:
        if context.vehicle_state and context.trajectory:
            observation = context.observation
            context.action = self.controller.control(
                context.trajectory, context.vehicle_state, observation
            )
