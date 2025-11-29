"""Base plotter implementation."""

import matplotlib.pyplot as plt
from core.data import Trajectory
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class BasePlotter:
    """Base class for plotters."""

    def __init__(self, track_trajectory: Trajectory | None = None) -> None:
        """Initialize base plotter.

        Args:
            track_trajectory: Reference track trajectory
        """
        self.fig: Figure
        self.ax: Axes
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        self.track_trajectory = track_trajectory
        
        # Calculate offset to center the plot
        self.offset_x = 0.0
        self.offset_y = 0.0
        if self.track_trajectory and len(self.track_trajectory) > 0:
            self.offset_x = self.track_trajectory[0].x
            self.offset_y = self.track_trajectory[0].y
            
        self._plot_track()

    def _plot_track(self) -> None:
        """Plot the reference track."""
        if self.track_trajectory:
            track_x, track_y, _, _ = self.track_trajectory.to_arrays()
            # Apply offset
            track_x = [x - self.offset_x for x in track_x]
            track_y = [y - self.offset_y for y in track_y]
            
            self.ax.plot(track_x, track_y, "k--", label="Track", alpha=0.5)
            self.ax.set_aspect("equal")
            self.ax.grid(True)
            self.ax.set_xlabel(f"X [m] (Offset: {self.offset_x:.2f})")
            self.ax.set_ylabel(f"Y [m] (Offset: {self.offset_y:.2f})")

    def close(self) -> None:
        """Close the figure."""
        plt.close(self.fig)
