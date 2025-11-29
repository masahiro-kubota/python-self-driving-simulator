"""Simulation plotter implementation."""

import matplotlib.pyplot as plt
from core.data import SimulationLog, Trajectory
from tools.visualization.base import BasePlotter


class SimulationPlotter(BasePlotter):
    """Plotter for simulation results (static)."""

    def __init__(self, track_trajectory: Trajectory | None = None) -> None:
        """Initialize plotter.

        Args:
            track_trajectory: Reference track trajectory
        """
        super().__init__(track_trajectory)
        self.logs: list[tuple[SimulationLog, str]] = []

    def add_log(self, log: SimulationLog, label: str = "Vehicle") -> None:
        """Add simulation log to plotter.

        Args:
            log: Simulation log
            label: Label for the legend
        """
        self.logs.append((log, label))

    def save(self, filename: str) -> None:
        """Save plot to file.

        Args:
            filename: Output filename
        """
        # Plot vehicle trajectories
        colors = ["b", "r", "g", "c", "m", "y"]
        
        for i, (log, label) in enumerate(self.logs):
            color = colors[i % len(colors)]
            history_x = [s.vehicle_state.x - self.offset_x for s in log.steps]
            history_y = [s.vehicle_state.y - self.offset_y for s in log.steps]
            
            self.ax.plot(history_x, history_y, f"{color}-", label=label)
            
            # Plot start and end
            if history_x:
                self.ax.plot(history_x[0], history_y[0], f"{color}o", markersize=5)
                self.ax.plot(history_x[-1], history_y[-1], f"{color}x", markersize=5)
            
        self.ax.legend()
        self.fig.savefig(filename)
        self.close()

    def show(self) -> None:
        """Show plot interactively."""
        plt.show()
