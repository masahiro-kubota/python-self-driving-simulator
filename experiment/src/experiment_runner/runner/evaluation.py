"""Experiment executor implementation."""

from typing import TYPE_CHECKING, Any

from core.data import (
    SimulationResult,
)
from core.interfaces.node import Node, SimulationContext
from core.nodes import PhysicsNode
from experiment_runner.interfaces import ExperimentRunner
from experiment_runner.preprocessing.schemas import ResolvedExperimentConfig

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


class EvaluationRunner(ExperimentRunner[ResolvedExperimentConfig, SimulationResult]):
    """Runner for evaluation experiments."""

    def run(self, config: ResolvedExperimentConfig, components: dict[str, Any]) -> SimulationResult:
        """Run evaluation experiment.

        Args:
            config: Experiment configuration
            components: Instantiated components (simulator, ad_component, etc.)

        Returns:
            Simulation result
        """
        simulator = components["simulator"]
        ad_component = components["ad_component"]

        # Run simulation
        max_steps = config.execution.max_steps_per_episode if config.execution else 2000
        sim_rate = config.simulator.rate_hz

        # Create Context & Reset simulator
        initial_state = simulator.reset()
        context = SimulationContext(
            current_time=0.0,
            sim_state=initial_state,
            vehicle_state=initial_state,  # Initialize perceived state
        )

        # Collect Nodes
        nodes = []
        # 1. Physics
        goal_radius = (
            config.execution.goal_radius
            if config.execution and hasattr(config.execution, "goal_radius")
            else 5.0
        )
        nodes.append(PhysicsNode(simulator, rate_hz=sim_rate, goal_radius=goal_radius))

        # 2. ADComponent Nodes
        nodes.extend(ad_component.get_schedulable_nodes())

        executor = SingleProcessExecutor(nodes, context)

        # Run
        duration = max_steps * (1.0 / sim_rate)
        sim_result = executor.run(duration=duration, dt=1.0 / sim_rate)

        return sim_result

    def get_type(self) -> str:
        return "evaluation"
