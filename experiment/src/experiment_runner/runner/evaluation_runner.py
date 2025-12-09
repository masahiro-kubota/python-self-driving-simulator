"""Experiment executor implementation."""

from typing import TYPE_CHECKING, Any

from core.data import (
    SimulationResult,
)
from core.executor import SingleProcessExecutor
from core.interfaces.node import SimulationContext
from core.nodes import PhysicsNode
from experiment_runner.interfaces import ExperimentRunner
from experiment_runner.preprocessing.schemas import ResolvedExperimentConfig

if TYPE_CHECKING:
    pass


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

        # Inject context into nodes
        for node in nodes:
            node.set_context(context)

        # Determine Clock Type
        clock_type = config.execution.clock_type if config.execution else "stepped"

        if clock_type == "stepped":
            from core.clock import SteppedClock

            # SteppedClock manages time advancement
            clock = SteppedClock(start_time=0.0, dt=1.0 / sim_rate)
        else:
            raise ValueError(f"Unsupported clock type: {clock_type}")

        executor = SingleProcessExecutor(nodes, clock)

        # Run
        duration = max_steps * (1.0 / sim_rate)
        executor.run(duration=duration, stop_condition=lambda: context.done)

        return SimulationResult(
            success=context.success,
            reason=context.done_reason,
            final_state=context.sim_state,
            log=simulator.get_log(),
        )

    def get_type(self) -> str:
        return "evaluation"
