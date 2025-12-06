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
from simulator_core.data import DynamicVehicleState

if TYPE_CHECKING:
    from core.data import ADComponentLog, Trajectory
    from core.interfaces import Controller, Planner


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
    ) -> None:
        """初期化.

        Args:
            vehicle_params: 車両パラメータ（Noneの場合はデフォルト値を使用）
            initial_state: 初期車両状態
            dt: シミュレーション時間刻み [s]
            map_path: Lanelet2マップファイルへのパス
        """
        # 後方互換性のため、vehicle_paramsがNoneの場合はデフォルト値を使用
        if vehicle_params is None:
            vehicle_params = VehicleParameters()
        elif isinstance(vehicle_params, dict):
            vehicle_params = VehicleParameters(**vehicle_params)

        self.vehicle_params = vehicle_params
        self.dt = dt
        self.vehicle_params = vehicle_params
        self.dt = dt

        if initial_state is None:
            self.initial_state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0, timestamp=0.0)
        elif isinstance(initial_state, dict):
            self.initial_state = VehicleState(**initial_state)
        else:
            self.initial_state = initial_state

        # 内部状態はDynamicVehicleStateで管理
        self._current_state = DynamicVehicleState.from_vehicle_state(self.initial_state)
        self.log = SimulationLog()

        # マップの読み込み
        self.map: Any = None  # LaneletMap | None (runtime import)
        if map_path:
            import pathlib

            from simulator_core.map import LaneletMap

            self.map = LaneletMap(pathlib.Path(map_path))

    def reset(self) -> VehicleState:
        """シミュレーションをリセット.

        Returns:
            初期車両状態
        """
        self._current_state = DynamicVehicleState.from_vehicle_state(self.initial_state)
        self.log = SimulationLog()
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

        # 2. Convert to VehicleState for external interface
        vehicle_state = self._current_state.to_vehicle_state(action)

        # 3. Map validation (if map is loaded)
        if self.map is not None and not self.map.is_drivable(vehicle_state.x, vehicle_state.y):
            vehicle_state.off_track = True

        # 4. Logging
        step_log = SimulationStep(
            timestamp=vehicle_state.timestamp or 0.0,
            vehicle_state=vehicle_state,
            action=action,
            ad_component_log=self._create_ad_component_log(vehicle_state),
            info=self._create_info(),
        )
        self.log.add_step(step_log)

        # 5. Check done
        done = self._is_done()
        info = self._create_info()

        return vehicle_state, done, info

    @abstractmethod
    def _update_state(self, action: Action) -> DynamicVehicleState:
        """車両状態を更新する（サブクラスで実装）.

        Args:
            action: 実行するアクション

        Returns:
            更新後の車両状態（DynamicVehicleState形式）
        """
        pass

    def run(
        self,
        planner: "Planner",
        controller: "Controller",
        max_steps: int = 1000,
        reference_trajectory: "Trajectory | None" = None,
        goal_threshold: float = 5.0,
        min_elapsed_time: float = 20.0,
    ) -> SimulationResult:
        """シミュレーションを実行.

        Args:
            planner: プランナー
            controller: コントローラー
            max_steps: 最大ステップ数
            reference_trajectory: 参照軌道（ゴール判定用）
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
            target_trajectory = planner.plan(None, current_state)

            # Control
            action = controller.control(target_trajectory, current_state)

            # Simulate
            next_state, done, _ = self.step(action)

            # Check goal
            if reference_trajectory is not None:
                goal_reached = self._check_goal(
                    next_state,
                    reference_trajectory,
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

    def _create_ad_component_log(self, state: VehicleState) -> "ADComponentLog":
        """ADコンポーネントログを生成.

        サブクラスでオーバーライドして、必要なログを生成できます。

        Args:
            state: 現在の車両状態

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
        reference_trajectory: "Trajectory",
        step: int,
        goal_threshold: float,
        min_elapsed_time: float,
    ) -> bool:
        """ゴール到達判定.

        Args:
            state: 現在の車両状態
            reference_trajectory: 参照軌道
            step: 現在のステップ数
            goal_threshold: ゴール判定の距離閾値 [m]
            min_elapsed_time: ゴール判定を行う最小経過時間 [s]

        Returns:
            ゴール到達フラグ
        """
        # Check distance to goal
        dist_to_end = (
            (state.x - reference_trajectory[-1].x) ** 2
            + (state.y - reference_trajectory[-1].y) ** 2
        ) ** 0.5

        # Use time threshold to avoid early goal detection
        elapsed_time = step * self.dt
        return dist_to_end < goal_threshold and elapsed_time > min_elapsed_time
