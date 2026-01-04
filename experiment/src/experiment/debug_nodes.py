
import math
from typing import Any

from core.data import ComponentConfig
from core.data.node_io import NodeIO
from core.data.ros import Float32, Time
from core.interfaces.node import Node, NodeExecutionResult
from pydantic import Field


class ControlCmdDiffConfig(ComponentConfig):
    """Configuration for ControlCmdDiffNode."""

    topic_a: str = Field(..., description="First topic name")
    topic_b: str = Field(..., description="Second topic name")
    output_topic: str = Field("control_cmd_diff", description="Output topic name for difference")
    wait_for_both: bool = Field(True, description="Wait for both topics to be available")


class ControlCmdDiffNode(Node[ControlCmdDiffConfig]):
    """Node to publish the difference between two control commands (steering angle)."""

    def __init__(self, config: ControlCmdDiffConfig, rate_hz: float, priority: int):
        super().__init__("ControlCmdDiffNode", rate_hz, config, priority)
        self.topic_a = config.topic_a
        self.topic_b = config.topic_b
        self.output_topic = config.output_topic

        self.last_a_steer = 0.0
        self.last_b_steer = 0.0

    def get_node_io(self) -> NodeIO:
        from core.data.autoware import AckermannControlCommand

        return NodeIO(
            inputs={
                self.topic_a: AckermannControlCommand,
                self.topic_b: AckermannControlCommand,
            },
            outputs={
                self.output_topic: Float32,
            },
        )

    def on_run(self, _current_time: float) -> NodeExecutionResult:
        msg_a = self.subscribe(self.topic_a)
        msg_b = self.subscribe(self.topic_b)

        if self.config.wait_for_both:
            if msg_a is None or msg_b is None:
                return NodeExecutionResult.SKIPPED

        # Update last known values if available
        if msg_a:
            self.last_a_steer = self._get_steer(msg_a)
        
        if msg_b:
            self.last_b_steer = self._get_steer(msg_b)

        # Calculate diff
        diff = self.last_a_steer - self.last_b_steer

        # Publish
        self.publish(
            self.output_topic,
            Float32(data=diff)
        )

        return NodeExecutionResult.SUCCESS

    def _get_steer(self, msg: Any) -> float:
        # Handle both ControlCmd structures if successful, assuming core.data.autoware structure
        # msg.lateral.steering_tire_angle
        if hasattr(msg, "lateral") and hasattr(msg.lateral, "steering_tire_angle"):
            return msg.lateral.steering_tire_angle
        if hasattr(msg, "steering_angle"): # fallback
            return msg.steering_angle
        return 0.0
