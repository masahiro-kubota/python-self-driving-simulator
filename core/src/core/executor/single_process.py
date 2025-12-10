"""Single process executor for simulation."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from core.interfaces.clock import Clock
from core.interfaces.node import Node

if TYPE_CHECKING:
    pass


class SingleProcessExecutor:
    """Time-based scheduler for single process execution.

    シングルプロセスでシミュレーションを実行するためのスケジューラクラスです。
    シミュレーション時間に基づき、登録された各ノードの実行タイミングを管理します。
    """

    def __init__(self, nodes: list[Node], clock: Clock):
        self.nodes = nodes
        self.clock = clock

    def run(self, duration: float, stop_condition: Callable[[], bool] | None = None) -> None:
        """Run the simulation loop.

        シミュレーションループを実行します。
        指定された期間(duration)だけループを回し、各ステップで実行すべきノードを呼び出します。

        Args:
            duration: 実行期間 [sec]
            stop_condition: 終了条件を判定するコールバック関数(Trueを返すと終了)
        """
        step_count = 0

        # メインループ: 指定時間経過するか、終了条件が満たされるまで継続
        while self.clock.now < duration:
            # Check stop condition
            if stop_condition and stop_condition():
                break

            # Check termination signal from any node via FrameData
            for node in self.nodes:
                if (
                    hasattr(node, "frame_data")
                    and node.frame_data is not None
                    and hasattr(node.frame_data, "termination_signal")
                    and node.frame_data.termination_signal
                ):
                    # Termination requested by a node
                    break
            else:
                # No termination signal, continue execution
                for node in self.nodes:
                    # 各ノードに対して、現在の時刻で実行すべきか(周期が来ているか)を確認
                    if node.should_run(self.clock.now):
                        node.on_run(self.clock.now)
                        node.next_time += node.period

                self.clock.tick()
                step_count += 1
                continue

            # Termination signal detected, break outer loop
            break
