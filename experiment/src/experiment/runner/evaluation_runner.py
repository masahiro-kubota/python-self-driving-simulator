"""実験実行の実装モジュール。"""

from typing import TYPE_CHECKING, Any

from core.clock import create_clock
from core.data import SimulationResult
from core.data.frame_data import collect_node_output_fields, create_frame_data_type
from core.executor import SingleProcessExecutor
from core.nodes import PhysicsNode
from experiment.interfaces import ExperimentRunner
from experiment.preprocessing.schemas import ResolvedExperimentConfig
from logger import LoggerNode
from supervisor import SupervisorNode

if TYPE_CHECKING:
    pass


class EvaluationRunner(ExperimentRunner[ResolvedExperimentConfig, SimulationResult]):
    """評価実験用のランナー。"""

    def run(self, config: ResolvedExperimentConfig, components: dict[str, Any]) -> SimulationResult:
        """評価実験を実行します。

        Args:
            config: 実験設定
            components: インスタンス化されたコンポーネント (simulator, ad_componentなど)

        Returns:
            Simulation result
        """
        simulator = components["simulator"]
        ad_component = components["ad_component"]

        # 設定値の取得
        sim_rate = config.simulator.rate_hz
        max_steps = config.execution.max_steps_per_episode if config.execution else 2000
        goal_radius = config.execution.goal_radius if config.execution else 5.0

        # ノードの収集
        nodes = []

        # 1. 物理ノード (PhysicsNode)
        # シミュレータのリセットと初期状態の取得
        _ = simulator.reset()
        nodes.append(PhysicsNode(simulator, sim_rate))

        # 2. ADコンポーネントnode
        nodes.extend(ad_component.get_schedulable_nodes())

        # 3. 評価ノード (SupervisorNode)
        goal_x = getattr(simulator, "goal_x", None)
        goal_y = getattr(simulator, "goal_y", None)
        supervisor = SupervisorNode(
            goal_x=goal_x,
            goal_y=goal_y,
            goal_radius=goal_radius,
            max_steps=max_steps,
            rate_hz=sim_rate,
        )
        nodes.append(supervisor)

        # 4. ロガーノード (LoggerNode)
        logger = LoggerNode(rate_hz=sim_rate)
        nodes.append(logger)

        # FrameDataの構築
        # 1. 全nodeのIO要件を収集
        fields = collect_node_output_fields(nodes)

        # 2. 動的なFrameDataクラスを作成
        DynamicFrameData = create_frame_data_type(fields)  # noqa: N806

        # 3. FrameDataのインスタンス化
        # 初期値はNoneで初期化され、最初のステップで各ノードがデータを埋めることを想定
        frame_data = DynamicFrameData()

        # コンテキストを各ノードに注入
        for node in nodes:
            node.set_frame_data(frame_data)

        # クロックの作成
        clock_type = config.execution.clock_type if config.execution else "stepped"
        clock = create_clock(start_time=0.0, rate_hz=sim_rate, clock_type=clock_type)

        # Executorの作成
        executor = SingleProcessExecutor(nodes, clock)

        # 実験の実行
        duration = max_steps * (1.0 / sim_rate)
        executor.run(duration=duration)

        # 結果の取得
        return SimulationResult(
            success=getattr(frame_data, "success", False),
            reason=getattr(frame_data, "done_reason", "unknown"),
            final_state=getattr(frame_data, "sim_state", None),
            log=logger.get_log(),
        )

    def get_type(self) -> str:
        return "evaluation"
