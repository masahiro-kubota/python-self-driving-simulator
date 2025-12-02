"""AD Component Log data structure."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ADComponentLog:
    """ADコンポーネントのログ（柔軟な構造）.

    各ADコンポーネントが出力する任意のログデータを格納します。
    コンポーネントの種類によって異なるデータを含むことができます。

    Attributes:
        component_type: コンポーネントの種類 ("planner", "controller", "e2e"など)
        data: コンポーネント固有のログデータ（辞書形式で柔軟に対応）

    Examples:
        Plannerのログ:
            ADComponentLog(
                component_type="planner",
                data={"trajectory_length": 100, "computation_time_ms": 5.2}
            )

        Controllerのログ:
            ADComponentLog(
                component_type="controller",
                data={"steering": 0.1, "throttle": 0.5, "pid_error": 0.02}
            )

        E2Eモデルのログ:
            ADComponentLog(
                component_type="e2e",
                data={"model_output": ..., "attention_weights": ...}
            )
    """

    component_type: str
    data: dict[str, Any] = field(default_factory=dict)
