"""AD Component configuration data structures."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ADComponentType(Enum):
    """自動運転コンポーネントのタイプ."""

    PERCEPTION = "perception"
    PLANNER = "planner"
    CONTROLLER = "controller"
    END_TO_END = "end_to_end"


@dataclass
class ADComponentSpec:
    """個別のコンポーネント仕様.

    Attributes:
        type: コンポーネントタイプ
        class_path: クラスパス（例: "pure_pursuit.PurePursuitPlanner"）
        params: 初期化パラメータ
    """

    type: ADComponentType
    class_path: str
    params: dict[str, Any]


@dataclass
class ADComponentConfig:
    """自動運転コンポーネント設定.

    E2Eモデルの場合:
        components = [ADComponentSpec(type=END_TO_END, ...)]

    従来型の場合:
        components = [
            ADComponentSpec(type=PLANNER, ...),
            ADComponentSpec(type=CONTROLLER, ...)
        ]

    複数プランナー（アンサンブル等）:
        components = [
            ADComponentSpec(type=PLANNER, class_path="planner1.Planner1", ...),
            ADComponentSpec(type=PLANNER, class_path="planner2.Planner2", ...),
            ADComponentSpec(type=CONTROLLER, ...)
        ]
    """

    components: list[ADComponentSpec]

    def validate(self) -> None:
        """設定の妥当性チェック.

        Raises:
            ValueError: E2Eコンポーネントと他のコンポーネントが混在している場合
        """
        component_types = [c.type for c in self.components]

        # E2Eと他のコンポーネントが混在していないかチェック
        if ADComponentType.END_TO_END in component_types and len(self.components) > 1:
            raise ValueError("END_TO_END component cannot be used with other components")

    def has_end_to_end(self) -> bool:
        """E2Eコンポーネントを含むか.

        Returns:
            E2Eコンポーネントが含まれている場合True
        """
        return any(c.type == ADComponentType.END_TO_END for c in self.components)

    def get_component(self, component_type: ADComponentType) -> ADComponentSpec | None:
        """特定タイプのコンポーネントを取得（最初の1つ）.

        Args:
            component_type: コンポーネントタイプ

        Returns:
            コンポーネント仕様、見つからない場合None
        """
        for c in self.components:
            if c.type == component_type:
                return c
        return None

    def get_components(self, component_type: ADComponentType) -> list[ADComponentSpec]:
        """特定タイプのコンポーネントを全て取得.

        Args:
            component_type: コンポーネントタイプ

        Returns:
            コンポーネント仕様のリスト
        """
        return [c for c in self.components if c.type == component_type]
