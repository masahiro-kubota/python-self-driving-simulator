"""Experiment executor implementation."""

from typing import TYPE_CHECKING

from core.data import (
    SimulationResult,
)
from core.interfaces.node import Node, SimulationContext
from core.nodes import PhysicsNode

if TYPE_CHECKING:
    pass


class SingleProcessExecutor:
    """Time-based scheduler for single process execution."""

    def __init__(self, nodes: list[Node], context: SimulationContext):
        self.nodes = nodes
        self.context = context

    def run(self, duration: float, dt: float = 0.01) -> SimulationResult:
        """Run the simulation loop."""
        sim_time = 0.0
        step_count = 0

        # Get simulator from PhysicsNode for final log retrieval
        physics_node = next((n for n in self.nodes if isinstance(n, PhysicsNode)), None)
        assert physics_node is not None, "PhysicsNode required"
        simulator = physics_node.simulator

        # Reset simulator if not already done?
        # Expectation: caller calls simulator.reset() and initializes context.

        while sim_time < duration and not self.context.done:
            self.context.current_time = sim_time

            for node in self.nodes:
                if node.should_run(sim_time):
                    node.on_run(self.context)
                    node.next_time += node.period

            sim_time += dt
            step_count += 1

        return SimulationResult(
            success=self.context.success,
            reason=self.context.done_reason,
            final_state=self.context.sim_state,
            log=simulator.get_log(),
        )
