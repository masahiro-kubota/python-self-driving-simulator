"""Simulation animator implementation."""

from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from core.data import SimulationLog, Trajectory
from tools.visualization.base import BasePlotter


class SimulationAnimator(BasePlotter):
    """Animator for simulation results."""

    def __init__(self, track_trajectory: Trajectory | None = None) -> None:
        """Initialize animator.

        Args:
            track_trajectory: Reference track trajectory
        """
        super().__init__(track_trajectory)
        self.logs: list[tuple[SimulationLog, str]] = []
        self.points: list[Any] = []  # List of plot objects for vehicles

    def add_log(self, log: SimulationLog, label: str = "Vehicle") -> None:
        """Add simulation log to animator.

        Args:
            log: Simulation log
            label: Label for the legend
        """
        self.logs.append((log, label))

    def save_animation(self, filename: str, interval: int = 50) -> None:
        """Save animation to file.

        Args:
            filename: Output filename (e.g. .gif, .mp4)
            interval: Interval between frames [ms]
        """
        if not self.logs:
            print("No logs to animate.")
            return

        # Determine number of frames
        min_steps = min(len(log.steps) for log, _ in self.logs)
        colors = ["b", "r", "g", "c", "m", "y"]

        # Initialize plot objects
        self.points = []
        for i, (_, label) in enumerate(self.logs):
            color = colors[i % len(colors)]
            # Initial point (off-screen or first point)
            point, = self.ax.plot([], [], f"{color}o", label=label, markersize=8)
            self.points.append(point)
            
            # Also plot full trajectory as thin line
            log = self.logs[i][0]
            history_x = [s.vehicle_state.x - self.offset_x for s in log.steps]
            history_y = [s.vehicle_state.y - self.offset_y for s in log.steps]
            self.ax.plot(history_x, history_y, f"{color}-", alpha=0.3)

        self.ax.legend()
        self.ax.set_title("Simulation Animation")

        def init() -> list[Any]:
            for point in self.points:
                point.set_data([], [])
            return self.points

        def update(frame: int) -> list[Any]:
            for i, (log, _) in enumerate(self.logs):
                step = log.steps[frame]
                x = step.vehicle_state.x - self.offset_x
                y = step.vehicle_state.y - self.offset_y
                self.points[i].set_data([x], [y])
                
            self.ax.set_title(f"Simulation Animation (Step {frame}/{min_steps})")
            return self.points

        ani = animation.FuncAnimation(
            self.fig,
            update,
            frames=min_steps,
            init_func=init,
            interval=interval,
            blit=True,
        )

        print(f"Saving animation to {filename}...")
        if filename.endswith(".gif"):
            writer = animation.PillowWriter(fps=1000 // interval)
            ani.save(filename, writer=writer)
        elif filename.endswith(".mp4"):
            # Requires ffmpeg
            try:
                writer = animation.FFMpegWriter(fps=1000 // interval)
                ani.save(filename, writer=writer)
            except Exception as e:
                print(f"Error saving MP4 (ffmpeg might be missing): {e}")
        else:
            print(f"Unsupported format: {filename}")
            
        self.close()
