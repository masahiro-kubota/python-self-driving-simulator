from typing import Any

from core.data.node_io import NodeIO
from core.data.simulation_context import SimulationContext
from core.interfaces.node import Node
from core.interfaces.processor import ProcessorProtocol


class GenericProcessingNode(Node):
    """汎用的な処理ノード.

    ProcessorとNodeIOを組み合わせて、データフローベースの処理を実行する。
    入力フィールドをContextから読み取り、Processorで処理し、
    出力フィールドにContextに書き込む。
    """

    def __init__(
        self,
        name: str,
        processor: ProcessorProtocol,
        io_spec: NodeIO,
        rate_hz: float,
    ) -> None:
        """Initialize generic processing node.

        Args:
            name: ノード名
            processor: 処理を実行するProcessor
            io_spec: 入出力の定義
            rate_hz: 実行周波数 [Hz]
        """
        super().__init__(name, rate_hz)
        self.processor = processor
        self.io_spec = io_spec

    def on_run(self, context: SimulationContext) -> None:
        """Execute node logic.

        Args:
            context: シミュレーションコンテキスト
        """
        # 入力を収集
        inputs: dict[str, Any] = {}
        for field_name in self.io_spec.inputs:
            value = getattr(context, field_name, None)
            if value is None:
                # 必要なデータがまだない場合はスキップ
                return
            inputs[field_name] = value

        # 処理を実行
        output = self.processor.process(**inputs)

        # 出力を書き込み
        setattr(context, self.io_spec.output, output)
