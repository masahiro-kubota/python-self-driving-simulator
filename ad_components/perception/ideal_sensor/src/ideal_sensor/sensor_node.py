from core.data import ComponentConfig
from core.data.node_io import NodeIO
from core.interfaces.node import Node, NodeExecutionResult


class IdealSensorConfig(ComponentConfig):
    pass


class IdealSensorNode(Node[IdealSensorConfig]):
    """テスト用のパススルーセンサーノード"""

    def __init__(self, config: IdealSensorConfig, rate_hz: float):
        super().__init__("Sensor", rate_hz, config)

    def get_node_io(self) -> NodeIO:
        from core.data import VehicleState

        return NodeIO(inputs={"sim_state": VehicleState}, outputs={"vehicle_state": VehicleState})

    def on_run(self, _current_time: float) -> NodeExecutionResult:
        if self.frame_data is None:
            return NodeExecutionResult.FAILED

        sim_state = getattr(self.frame_data, "sim_state", None)
        if sim_state:
            self.frame_data.vehicle_state = sim_state

        return NodeExecutionResult.SUCCESS
