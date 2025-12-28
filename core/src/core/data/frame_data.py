"""Frame data definition."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


@dataclass
class FrameData:
    """Base class for frame data.

    Acts as a shared memory between nodes for a single step.
    Specific fields are defined dynamically based on node requirements.
    """


def create_frame_data_type(fields: dict[str, type | str]) -> type[FrameData]:
    """Create a dynamic FrameData class with specified fields.

    Args:
        fields: Dictionary mapping field names to types.

    Returns:
        A new dataclass type inheriting from FrameData.
    """
    from dataclasses import field, make_dataclass

    from core.data.topic_slot import TopicSlot

    # Resolve string types to Any if needed, or keep as is if make_dataclass handles it (it needs types usually)
    # For simplicity, we default to Any if type is string "Any"
    resolved_fields = []
    for name, type_ in fields.items():
        if isinstance(type_, str):
            # Fallback for string types
            type_hint = Any
        else:
            type_hint = type_

        resolved_fields.append((name, TopicSlot[type_hint], field(default_factory=TopicSlot)))

    return make_dataclass("DynamicFrameData", resolved_fields, bases=(FrameData,))


if TYPE_CHECKING:
    from core.interfaces.node import Node


def collect_node_output_fields(nodes: list["Node"]) -> dict[str, type]:
    """Collect all output field types from a list of nodes.

    Args:
        nodes: List of nodes to collect IO from.

    Returns:
        Dictionary mapping field names to types.
    """
    fields: dict[str, type] = {}
    for node in nodes:
        io = node.get_node_io()
        for name, type_ in io.outputs.items():
            if isinstance(type_, str):
                fields[name] = Any  # Resolve string types
            else:
                fields[name] = type_
    return fields
