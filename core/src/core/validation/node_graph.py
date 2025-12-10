"""Node graph validation and visualization."""

from core.interfaces.node import Node


def validate_node_graph(nodes: list[Node]) -> None:
    """Validate that the node graph is consistent and acyclic.

    Args:
        nodes: List of nodes to validate

    Raises:
        ValueError: If graph is invalid
    """
    # 利用可能な出力(初期値として"action"と"sim_state"を含む - 循環のため)
    available_outputs = {"action", "sim_state"}

    for node in nodes:
        io_spec = node.get_node_io()

        # 必要な入力が利用可能かチェック
        for input_field in io_spec.inputs:
            if input_field not in available_outputs:
                raise ValueError(
                    f"Node '{node.name}' requires '{input_field}' "
                    f"but no previous node produces it. "
                    f"Available outputs: {available_outputs}"
                )

        # このノードの出力を追加
        available_outputs.update(io_spec.outputs.keys())

    # actionが最終的に生成されるかチェック
    if "action" not in available_outputs:
        raise ValueError("No node produces 'action' required for simulation")


def visualize_node_graph(nodes: list[Node]) -> str:
    """Visualize node graph in Mermaid format.

    Args:
        nodes: List of nodes

    Returns:
        str: Mermaid graph definition
    """
    lines = ["graph LR"]

    for node in nodes:
        io_spec = node.get_node_io()

        # ノード定義 (名前と周波数を表示)
        lines.append(f'    {node.name}["{node.name}<br/>{node.rate_hz}Hz"]')

        # 入力エッジ
        for input_field in io_spec.inputs:
            lines.append(f"    {input_field} --> {node.name}")

        # 出力エッジ
        for output_name in io_spec.outputs:
            lines.append(f"    {node.name} --> {output_name}")

    return "\n".join(lines)
