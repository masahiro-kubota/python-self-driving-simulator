from typing import Any

from core.data.node_io import NodeIO
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

    def on_run(self, _current_time: float) -> bool:
        """Execute node logic.

        Args:
            current_time: Current simulation time

        Returns:
            bool: True if execution was successful
        """
        if self.context is None:
            # Context not ready
            return False

        # 入力を収集
        inputs: dict[str, Any] = {}
        for field_name in self.io_spec.inputs:
            value = getattr(self.context, field_name, None)
            if value is None:
                # 必要なデータがまだない場合はスキップ
                return False
            inputs[field_name] = value

        # 処理を実行
        output = self.processor.process(**inputs)

        # 出力を書き込み
        setattr(self.context, self.io_spec.output, output)
        return True
