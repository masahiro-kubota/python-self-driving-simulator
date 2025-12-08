from dataclasses import dataclass


@dataclass
class NodeIO:
    """ノードの入出力定義."""

    inputs: list[str]
    output: str
