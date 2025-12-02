"""Base classes for simulators."""

from abc import ABC
from typing import TYPE_CHECKING, Any

from core.data import Observation, SimulationLog, SimulationResult, VehicleState
from core.interfaces import Simulator

if TYPE_CHECKING:
    from core.data import Trajectory
    from core.interfaces import Controller, Planner


class BaseSimulator(Simulator, ABC):
    """シミュレータの基底クラス.

    共通の初期化処理とヘルパーメソッドを提供します。
    """

    def __init__(
        self,
        initial_state: VehicleState | None = None,
        dt: float = 0.1,
    ) -> None:
        """初期化.

        Args:
            initial_state: 初期車両状態
            dt: シミュレーション時間刻み [s]
        """
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
            next_state, observation, done, info = self.step(action)

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

    def _create_observation(self, state: VehicleState) -> Observation:
        """観測データを生成.

        Args:
            state: 現在の車両状態

        Returns:
            観測データ
        """
        # TODO: Implement proper observation generation based on track/obstacles
        return Observation(
            lateral_error=0.0,
            heading_error=0.0,
            velocity=state.velocity,
            target_velocity=0.0,
            timestamp=state.timestamp,
        )

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
