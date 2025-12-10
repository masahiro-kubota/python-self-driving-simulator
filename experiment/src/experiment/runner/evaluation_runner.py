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

        # 設定値の取得
        clock_rate = config.execution.clock_rate_hz if config.execution else 100.0
        duration = config.execution.duration_sec if config.execution else 20.0
        clock_type = config.execution.clock_type if config.execution else "stepped"

        # FrameDataの構築
        # 1. 全nodeのIO要件を収集
        fields = collect_node_output_fields(nodes)

        # 2. 動的なFrameDataクラスを作成
        DynamicFrameData = create_frame_data_type(fields)  # noqa: N806

        # 3. FrameDataのインスタンス化
        # 初期値はNoneで初期化され、最初のステップで各ノードがデータを埋めることを想定
        frame_data = DynamicFrameData()

        # bool型フィールドの初期値をFalseに設定
        # (Noneのままだと、Executorのterminationシグナルチェックで問題が発生する可能性がある)
        for field_name, field_type in fields.items():
            if field_type is bool:
                setattr(frame_data, field_name, False)

        # コンテキストを各ノードに注入
        for node in nodes:
            node.set_frame_data(frame_data)

        # クロックの作成
        clock = create_clock(start_time=0.0, rate_hz=clock_rate, clock_type=clock_type)

        # Executorの作成
        executor = SingleProcessExecutor(nodes, clock)

        # 実験の実行
        executor.run(duration=duration)

        # 終了理由の表示
        reason = getattr(frame_data, "done_reason", "unknown")
        success = getattr(frame_data, "success", False)
        print(f"Simulation completed. Success: {success}, Reason: {reason}")

        # 結果の取得
        # Find LoggerNode to get log
        log = None
        for node in nodes:
            if isinstance(node, LoggerNode):
                log = node.get_log()
                break

        # Inject metadata into log
        if log is not None:
            # Inject vehicle parameters
            sim_params = config.simulator.params
            if "vehicle_params" in sim_params:
                v_params = sim_params["vehicle_params"]
                if hasattr(v_params, "to_dict"):
                    log.metadata.update(v_params.to_dict())
                elif isinstance(v_params, dict):
                    log.metadata.update(v_params)

            # Inject controller type if available
            if config.components.ad_component:
                log.metadata["controller"] = {"type": config.components.ad_component.type}

        return SimulationResult(
            success=getattr(frame_data, "success", False),
            reason=getattr(frame_data, "done_reason", "unknown"),
            final_state=getattr(frame_data, "sim_state", None),
            log=log,
        )

    def get_type(self) -> str:
        return "evaluation"
