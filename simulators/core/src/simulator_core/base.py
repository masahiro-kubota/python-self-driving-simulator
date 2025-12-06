"""Base classes for simulators."""

from abc import ABC
from typing import TYPE_CHECKING, Any

from core.data import (
    Scene,
    SimulationLog,
    SimulationResult,
    VehicleParameters,
    VehicleState,
)
from core.interfaces import Simulator

if TYPE_CHECKING:
    from core.data import ADComponentLog, Trajectory
    from core.interfaces import Controller, Planner


class BaseSimulator(Simulator, ABC):
    """シミュレータの基底クラス.

    共通の初期化処理とヘルパーメソッドを提供します。
    """

    def __init__(
        self,
        vehicle_params: "VehicleParameters | None" = None,
        scene: "Scene | None" = None,
        initial_state: VehicleState | None = None,
        dt: float = 0.1,
    ) -> None:
        """初期化.

        Args:
            vehicle_params: 車両パラメータ（Noneの場合はデフォルト値を使用）
            scene: シミュレーション環境（Noneの場合は空のシーンを使用）
            initial_state: 初期車両状態
            dt: シミュレーション時間刻み [s]
        """
        # 後方互換性のため、vehicle_paramsがNoneの場合はデフォルト値を使用
        if vehicle_params is None:
            vehicle_params = VehicleParameters()

        # 後方互換性のため、sceneがNoneの場合は空のシーンを使用
        if scene is None:
            scene = Scene()

        self.vehicle_params = vehicle_params
        self.scene = scene
        self.dt = dt
        self.initial_state = initial_state or VehicleState(
            x=0.0, y=0.0, yaw=0.0, velocity=0.0, timestamp=0.0
        )
        self._current_state = self.initial_state
        self.log = SimulationLog()

    def reset(self) -> VehicleState:
        """シミュレーションをリセット.

        Returns:
            初期車両状態
        """
        self._current_state = self.initial_state
        self.log = SimulationLog()
        return self._current_state

    def run(
        self,
        planner: "Planner",
        controller: "Controller",
        max_steps: int = 1000,
        reference_trajectory: "Trajectory | None" = None,
    ) -> SimulationResult:
        """シミュレーションを実行.

        Args:
            planner: プランナー
            controller: コントローラー
            max_steps: 最大ステップ数
            reference_trajectory: 参照軌道（ゴール判定用）

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
                goal_reached = self._check_goal(next_state, reference_trajectory, step)
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

    def render(self) -> None:
        """シミュレーションを描画(未実装)."""

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
        self, state: VehicleState, reference_trajectory: "Trajectory", step: int
    ) -> bool:
        """ゴール到達判定.

        Args:
            state: 現在の車両状態
            reference_trajectory: 参照軌道
            step: 現在のステップ数

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
        return dist_to_end < 5.0 and elapsed_time > 20.0
