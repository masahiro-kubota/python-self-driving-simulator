from core.data import ComponentConfig, VehicleParameters
from core.data.node_io import NodeIO
from core.interfaces.node import Node, NodeExecutionResult
from pydantic import Field


class IdealSensorConfig(ComponentConfig):
    vehicle_params: VehicleParameters = Field(..., description="Vehicle parameters")
    vehicle_color: str = Field("#0000FFCC", description="Vehicle color")


class IdealSensorNode(Node[IdealSensorConfig]):
    """テスト用のパススルーセンサーノード"""

    def __init__(self, config: IdealSensorConfig, rate_hz: float):
        super().__init__("Sensor", rate_hz, config)

        from core.data.ros import ColorRGBA
        from logger.visualization.vehicle_visualizer import VehicleVisualizer

        self.vehicle_visualizer = VehicleVisualizer(
            config.vehicle_params, color=ColorRGBA.from_hex(config.vehicle_color)
        )

    def get_node_io(self) -> NodeIO:
        from core.data import VehicleState
        from core.data.ros import MarkerArray

        return NodeIO(
            inputs={"sim_state": VehicleState},
            outputs={
                "vehicle_state": VehicleState,
                "vehicle_marker": MarkerArray,
            },
        )

    def on_run(self, _current_time: float) -> NodeExecutionResult:
        if self.frame_data is None:
            return NodeExecutionResult.FAILED

        sim_state = getattr(self.frame_data, "sim_state", None)
        if sim_state:
            self.frame_data.vehicle_state = sim_state

            # Visualization
            from core.data.ros import MarkerArray

            marker = self.vehicle_visualizer.create_marker(sim_state, _current_time)
            # Directly set the attribute with the topic name expected by Foxglove/Logger
            setattr(self.frame_data, "vehicle/marker", MarkerArray(markers=[marker]))

        return NodeExecutionResult.SUCCESS
