"""Base classes for simulators."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from core.data import (
    Action,
    SimulationLog,
    SimulationResult,
    SimulationStep,
    VehicleParameters,
    VehicleState,
)
from core.interfaces import Simulator
from simulator_core.data import SimulationVehicleState

if TYPE_CHECKING:
    from shapely.geometry import Polygon

    from core.data import ADComponentLog
    from core.interfaces import ADComponent


class BaseSimulator(Simulator, ABC):
    """シミュレータの基底クラス.

    共通の初期化処理、ステップ実行、ログ記録を提供します。
    """

    def __init__(
        self,
        vehicle_params: "VehicleParameters | None" = None,
        initial_state: VehicleState | None = None,
        dt: float = 0.1,
        map_path: str | None = None,
        goal_x: float | None = None,
        goal_y: float | None = None,
    ) -> None:
        """初期化.

        Args:
            vehicle_params: 車両パラメータ（Noneの場合はデフォルト値を使用）
            initial_state: 初期車両状態
            dt: シミュレーション時間刻み [s]
            map_path: Lanelet2マップファイルへのパス
            goal_x: ゴール位置のX座標 [m]
            goal_y: ゴール位置のY座標 [m]
        """
        # 後方互換性のため、vehicle_paramsがNoneの場合はデフォルト値を使用
        if vehicle_params is None:
            vehicle_params = VehicleParameters()
        elif isinstance(vehicle_params, dict):
            vehicle_params = VehicleParameters(**vehicle_params)

        self.vehicle_params = vehicle_params
        self.dt = dt
        self.goal_x = goal_x
        self.goal_y = goal_y

        if initial_state is None:
            self.initial_state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0, timestamp=0.0)
        elif isinstance(initial_state, dict):
            self.initial_state = VehicleState(**initial_state)
        else:
            self.initial_state = initial_state

        # 内部状態はSimulationVehicleStateで管理
        self._current_state = SimulationVehicleState.from_vehicle_state(self.initial_state)
        self.current_time = 0.0  # シミュレーション時刻の追跡
        self.log = SimulationLog(steps=[], metadata={})

        # マップの読み込み
        self.map: Any = None  # LaneletMap | None (runtime import)
        if map_path:
            import pathlib

            from simulator_core.lanelet_map import LaneletMap

            self.map = LaneletMap(pathlib.Path(map_path))

    def reset(self) -> VehicleState:
        """シミュレーションをリセット.

        Returns:
            初期車両状態
        """
        self._current_state = SimulationVehicleState.from_vehicle_state(self.initial_state)
        self.current_time = 0.0
        self.log = SimulationLog(steps=[], metadata={})
        return self.initial_state

    def step(self, action: Action) -> tuple[VehicleState, bool, dict[str, Any]]:
        """シミュレーションを1ステップ進める.

        Args:
            action: 実行するアクション

        Returns:
            tuple containing:
                - next_state: Updated vehicle state
                - done: Episode termination flag
                - info: Additional information
        """
        # 1. Update state (Subclass responsibility)
        self._current_state = self._update_state(action)
        self.current_time += self.dt  # 時刻を進める
        self._current_state.timestamp = self.current_time  # 状態のタイムスタンプ更新

        # 2. Convert to VehicleState for external interface
        vehicle_state = self._current_state.to_vehicle_state(action)

        # 3. Map validation (if map is loaded)
        if self.map is not None:
            # Check if vehicle polygon is within drivable area
            # If implementation of _get_vehicle_polygon is missing (e.g. older subclasses),
            # fallback to point check or error?
            # Since we control subclasses, we assume implementation.
            # But for safety, we can wrap try-excerpt? No, let's enforce it.

            try:
                poly = self._get_vehicle_polygon(vehicle_state)
                if not self.map.is_drivable_polygon(poly):
                    vehicle_state.off_track = True
            except NotImplementedError:
                # Fallback to point check if not implemented
                if not self.map.is_drivable(vehicle_state.x, vehicle_state.y):
                    vehicle_state.off_track = True

        # 4. Logging
        step_log = SimulationStep(
            timestamp=self.current_time,
            vehicle_state=vehicle_state,
            action=action,
            ad_component_log=self._create_ad_component_log(),
            info=self._create_info(),
        )
        self.log.steps.append(step_log)

        # 5. Check done
        done = self._is_done()
        info = self._create_info()

        return vehicle_state, done, info

    @abstractmethod
    def _update_state(self, action: Action) -> SimulationVehicleState:
        """車両状態を更新する（サブクラスで実装）.

        Args:
            action: 実行するアクション

        Returns:
            更新後の車両状態（SimulationVehicleState形式）
        """

    def run(
        self,
        ad_component: "ADComponent",
        max_steps: int = 1000,
        goal_threshold: float = 5.0,
        min_elapsed_time: float = 20.0,
    ) -> SimulationResult:
        """シミュレーションを実行.

        Args:
            ad_component: AD component instance (planner + controller)
            max_steps: 最大ステップ数
            goal_threshold: ゴール判定の距離閾値 [m]
            min_elapsed_time: ゴール判定を行う最小経過時間 [s]

        Returns:
            SimulationResult: シミュレーション結果
        """
        # Reset simulator
        current_state = self.reset()

        # Run simulation loop
        for step in range(max_steps):
            # Plan
            target_trajectory = ad_component.planner.plan(None, current_state)

            # Control
            action = ad_component.controller.control(target_trajectory, current_state)

            # Simulate
            next_state, done, _ = self.step(action)

            # Check goal
            if self.goal_x is not None and self.goal_y is not None:
                goal_reached = self._check_goal(
                    next_state,
                    step,
                    goal_threshold,
                    min_elapsed_time,
                )
                if goal_reached:
                    return SimulationResult(
                        success=True,
                        reason="goal_reached",
                        final_state=next_state,
                        log=self.get_log(),
                    )

            # Update state
            current_state = next_state

            # Check done flag
            if done:
                return SimulationResult(
                    success=False,
                    reason="done_flag",
                    final_state=current_state,
                    log=self.get_log(),
                )

        # Max steps reached
        return SimulationResult(
            success=False,
            reason="max_steps",
            final_state=current_state,
            log=self.get_log(),
        )

    def get_log(self) -> SimulationLog:
        """シミュレーションログを取得.

        Returns:
            SimulationLog: シミュレーションログ
        """
        return self.log

    def close(self) -> None:
        """シミュレータを終了."""

    def _create_ad_component_log(self) -> "ADComponentLog":
        """ADコンポーネントログを生成.

        サブクラスでオーバーライドして、必要なログを生成できます。

        Returns:
            ADコンポーネントログ
        """
        from core.data import ADComponentLog

        # デフォルトでは空のログを返す
        return ADComponentLog(component_type="simulator", data={})

    def _create_info(self) -> dict[str, Any]:
        """追加情報を生成.

        Returns:
            追加情報の辞書
        """
        return {}

    def _is_done(self) -> bool:
        """エピソード終了判定.

        Returns:
            終了フラグ
        """
        return False

    def _check_goal(
        self,
        state: VehicleState,
        step: int,
        goal_threshold: float,
        min_elapsed_time: float,
    ) -> bool:
        """ゴール到達判定.

        Args:
            state: 現在の車両状態
            step: 現在のステップ数
            goal_threshold: ゴール判定の距離閾値 [m]
            min_elapsed_time: ゴール判定を行う最小経過時間 [s]

        Returns:
            ゴール到達フラグ
        """
        # Check distance to goal
        assert self.goal_x is not None and self.goal_y is not None
        dist_to_end = ((state.x - self.goal_x) ** 2 + (state.y - self.goal_y) ** 2) ** 0.5

        # Use time threshold to avoid early goal detection
        elapsed_time = step * self.dt
        return dist_to_end < goal_threshold and elapsed_time > min_elapsed_time

    def _get_vehicle_polygon(self, state: VehicleState) -> "Polygon":
        """車両のポリゴンを取得（サブクラスで実装）.

        Args:
            state: 車両状態

        Returns:
            Polygon: 車両のポリゴン
        """
        raise NotImplementedError("Subclasses must implement _get_vehicle_polygon")

    def _create_vehicle_polygon(
        self,
        x: float,
        y: float,
        yaw: float,
        front_edge_dist: float,
        rear_edge_dist: float,
        half_width: float,
    ) -> "Polygon":
        """車両パラメータからポリゴンを生成するヘルパー関数.

        Args:
            x: 基準点X座標
            y: 基準点Y座標
            yaw: ヨー角
            front_edge_dist: 基準点からフロントバンパーまでの距離（正）
            rear_edge_dist: 基準点からリアバンパーまでの距離（負）
            half_width: 車幅の半分

        Returns:
            Polygon: 回転・平行移動したポリゴン
        """
        import math

        from shapely.geometry import Polygon

        # Vehicle frame coordinates (x forward, y left)
        # p1: Front Left
        # p2: Front Right
        # p3: Rear Right
        # p4: Rear Left

        p1 = (front_edge_dist, half_width)
        p2 = (front_edge_dist, -half_width)
        p3 = (rear_edge_dist, -half_width)
        p4 = (rear_edge_dist, half_width)

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        points = []
        for px, py in [p1, p2, p3, p4]:
            # Rotate
            rx = px * cos_yaw - py * sin_yaw
            ry = px * sin_yaw + py * cos_yaw
            # Translate
            tx = rx + x
            ty = ry + y
            points.append((tx, ty))

        return Polygon(points)
