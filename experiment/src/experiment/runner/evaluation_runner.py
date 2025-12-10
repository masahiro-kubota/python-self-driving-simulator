"""実験実行の実装モジュール。"""

from typing import TYPE_CHECKING

from core.clock import create_clock
from core.data import SimulationResult
from core.data.frame_data import collect_node_output_fields, create_frame_data_type
from core.executor import SingleProcessExecutor
from experiment.interfaces import ExperimentRunner
from experiment.preprocessing.schemas import ResolvedExperimentConfig
from experiment.structures import Experiment
from logger import LoggerNode

if TYPE_CHECKING:
    pass


class EvaluationRunner(ExperimentRunner[ResolvedExperimentConfig, SimulationResult]):
    """評価実験用のランナー。"""

    def run(self, experiment: Experiment) -> SimulationResult:
        """評価実験を実行します。

        Args:
            experiment: 実験定義 (nodes, configを含む)

        Returns:
            Simulation result
        """
        config = experiment.config
        nodes = experiment.nodes

        # 設定値の取得 (Clock作成用)
        sim_rate = config.simulator.rate_hz
        max_steps = config.execution.max_steps_per_episode if config.execution else 2000

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
        # Find LoggerNode to get log
        log = None
        for node in nodes:
            if isinstance(node, LoggerNode):
                log = node.get_log()
                break

        return SimulationResult(
            success=getattr(frame_data, "success", False),
            reason=getattr(frame_data, "done_reason", "unknown"),
            final_state=getattr(frame_data, "sim_state", None),
            log=log,
        )

    def get_type(self) -> str:
        return "evaluation"
