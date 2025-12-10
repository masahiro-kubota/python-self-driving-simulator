from typing import Any

from core.data import VehicleState
from core.data.node_io import NodeIO
from core.interfaces.node import Node, NodeConfig


class IdealSensorConfig(NodeConfig):
    pass


class IdealSensorNode(Node[IdealSensorConfig]):
    """理想的なセンサーノード (ノイズなし、遅延なし)."""

    def __init__(self, config: dict, rate_hz: float, vehicle_params: Any | None = None):
        super().__init__("Sensor", rate_hz, config, config_model=IdealSensorConfig)
        _ = vehicle_params

    def get_node_io(self) -> NodeIO:
        return NodeIO(inputs={"sim_state": VehicleState}, outputs={"vehicle_state": VehicleState})

    def on_run(self, _current_time: float) -> bool:
        if self.frame_data is None:
            return False

        sim_state = getattr(self.frame_data, "sim_state", None)
        if sim_state is None:
            return True

        # Pass through
        self.frame_data.vehicle_state = sim_state
        return True
