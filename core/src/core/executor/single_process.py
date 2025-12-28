"""Single process executor for simulation."""

import logging
from collections.abc import Callable

from tqdm import tqdm

from core.interfaces.clock import Clock
from core.interfaces.node import Node, NodeExecutionResult

logger = logging.getLogger(__name__)


class SingleProcessExecutor:
    """Time-based scheduler for single process execution.

    シングルプロセスでシミュレーションを実行するためのスケジューラクラスです。
    シミュレーション時間に基づき、登録された各ノードの実行タイミングを管理します。
    """

    def __init__(self, nodes: list[Node], clock: Clock):
        # Sort nodes by priority (lower values execute first)
        # Use stable sort to maintain original order for nodes with same priority
        self.nodes = sorted(nodes, key=lambda node: node.priority)
        self.clock = clock

        # Log the execution order
        logger.info("Node execution order (by priority):")
        for i, node in enumerate(self.nodes, 1):
            logger.info(f"  {i}. {node.name} (priority={node.priority})")

    def run(
        self,
        duration: float,
        stop_condition: Callable[[], bool] | None = None,
        suppress_console_log: bool = True,
    ) -> None:
        """Run the simulation loop.

        シミュレーションループを実行します。
        指定された期間(duration)だけループを回し、各ステップで実行すべきノードを呼び出します。

        Args:
            duration: 実行期間 [sec]
            stop_condition: 終了条件を判定するコールバック関数(Trueを返すと終了)
            suppress_console_log: 実行中にコンソールへのログ出力を抑制するかどうか
        """
        # Initialize all nodes
        for node in self.nodes:
            node.on_init()

        step_count = 0

        # Calculate total steps for progress bar
        clock_rate = self.clock.rate_hz if hasattr(self.clock, "rate_hz") else 100.0
        total_steps = int(duration * clock_rate)

        # Console logging suppression logic
        root_logger = logging.getLogger()
        original_handlers_levels = {}

        if suppress_console_log:
            # Find and modify console handlers
            for handler in root_logger.handlers:
                # Assuming StreamHandler without a file output is likely stdout/stderr
                if isinstance(handler, logging.StreamHandler) and not isinstance(
                    handler, logging.FileHandler
                ):
                    original_handlers_levels[handler] = handler.level
                    # Suppress INFO/DEBUG, allow WARNING/ERROR
                    handler.setLevel(logging.WARNING)

        try:
            # メインループ: 指定時間経過するか、終了条件が満たされるまで継続
            with tqdm(total=total_steps, desc="Simulation", unit="step", ncols=100) as pbar:
                while self.clock.now < duration:
                    # Check stop condition
                    if stop_condition and stop_condition():
                        logger.info("Stop condition met, terminating simulation")
                        break

                    # Check termination signal from any node via FrameData
                    termination_detected = False
                    for node in self.nodes:
                        if (
                            hasattr(node, "frame_data")
                            and node.frame_data is not None
                            and hasattr(node.frame_data, "termination_signal")
                        ):
                            # Access TopicSlot data
                            slot = getattr(node.frame_data, "termination_signal")
                            if hasattr(slot, "data") and slot.data:
                                # Termination requested by a node
                                termination_detected = True
                                break

                    if termination_detected:
                        logger.info("Termination signal detected, ending simulation")
                        break

                    # No termination signal, continue execution
                    for node in self.nodes:
                        # 各ノードに対して、現在の時刻で実行すべきか(周期が来ているか)を確認
                        if node.should_run(self.clock.now):
                            result = node.on_run(self.clock.now)

                            # Handle execution result
                            if result == NodeExecutionResult.FAILED:
                                # Log or handle failure
                                pass
                            elif result == NodeExecutionResult.SKIPPED:
                                # Node skipped execution (e.g., missing inputs)
                                pass
                            # SUCCESS case needs no special handling

                            # Update next execution time
                            node.update_next_time(self.clock.now)

                    self.clock.tick()
                    step_count += 1
                    pbar.update(1)

                    # Update progress bar description with current time
                    if step_count % 100 == 0:
                        pbar.set_postfix({"time": f"{self.clock.now:.1f}s"})
        finally:
            # Restore original handler levels
            if suppress_console_log:
                for handler, level in original_handlers_levels.items():
                    handler.setLevel(level)

            # Shutdown all nodes
            for node in self.nodes:
                node.on_shutdown()
