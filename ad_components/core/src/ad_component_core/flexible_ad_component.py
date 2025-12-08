import importlib
from typing import Any

from core.data import VehicleParameters
from core.data.node_io import NodeIO
from core.interfaces.ad_components import ADComponent
from core.interfaces.node import Node
from core.nodes import GenericProcessingNode
from core.utils.paths import get_project_root
from core.validation.node_graph import validate_node_graph


class FlexibleADComponent(ADComponent):
    """柔軟なノード構成を持つADComponent.

    YAML設定から動的にノードを構築し、データフローを形成する。
    """

    def __init__(
        self, vehicle_params: VehicleParameters, nodes: list[dict[str, Any]], **_kwargs
    ) -> None:
        """Initialize FlexibleADComponent.

        Args:
            vehicle_params: 車両パラメータ
            nodes: ノード設定のリスト
            **_kwargs: その他のパラメータ (無視)
        """
        super().__init__(vehicle_params)
        self.nodes_list: list[Node] = []

        # 設定からノードを動的に構築
        for node_config in nodes:
            processor = self._create_processor(node_config["processor"], vehicle_params)
            node = GenericProcessingNode(
                name=node_config["name"],
                processor=processor,
                io_spec=NodeIO(**node_config["io"]),
                rate_hz=node_config["rate_hz"],
            )
            self.nodes_list.append(node)

        # ノードグラフを検証
        validate_node_graph(self.nodes_list)  # type: ignore

    def _create_processor(
        self, processor_config: dict[str, Any], vehicle_params: VehicleParameters
    ) -> Any:
        """Create processor from configuration.

        Args:
            processor_config: Processor設定
                             'type': クラスパス (e.g., "pure_pursuit.PurePursuitPlanner")
                             'params': パラメータ辞書
            vehicle_params: 車両パラメータ

        Returns:
            Any: 作成されたProcessor
        """
        import inspect

        processor_type = processor_config["type"]
        params = processor_config.get("params", {}).copy()

        # パスパラメータを解決
        path_keys = {"track_path", "model_path", "scaler_path"}
        workspace_root = get_project_root()
        for key, value in params.items():
            if key in path_keys and isinstance(value, str):
                params[key] = workspace_root / value

        # クラスをインポート
        try:
            module_name, class_name = processor_type.rsplit(".", 1)
        except ValueError as e:
            raise ValueError(
                f"Invalid processor type: {processor_type}. "
                "Must be in 'module.ClassName' format."
            ) from e

        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        # シグネチャを確認してvehicle_paramsが必要かチェック
        sig = inspect.signature(cls.__init__)
        if "vehicle_params" in sig.parameters:
            params["vehicle_params"] = vehicle_params

        # インスタンス化
        return cls(**params)

    def get_schedulable_nodes(self) -> list[Node]:
        """スケジュール可能なノードのリストを返す."""
        return self.nodes_list

    def reset(self) -> None:
        """コンポーネントをリセット."""
        # 必要に応じて各Processorのリセット処理を呼び出す
        # 現時点では未実装
        pass
