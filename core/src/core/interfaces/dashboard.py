"""Dashboard generation interface."""

from abc import ABC, abstractmethod
from pathlib import Path

from core.data.experiment import ExperimentResult


class DashboardGenerator(ABC):
    """Dashboard generation interface.

    This abstract base class defines the interface for generating interactive dashboards
    from experiment results. Implementations should generate HTML dashboards that
    can be viewed in a web browser.
    """

    @abstractmethod
    def generate(
        self,
        result: ExperimentResult,
        output_path: Path,
        osm_path: Path | None = None,
        vehicle_params: dict | None = None,
    ) -> Path:
        """Generate interactive dashboard from experiment result.

        Args:
            result: Experiment result containing simulation results and metadata
            output_path: Path where the generated HTML dashboard will be saved
            osm_path: Optional path to OSM map file for map visualization
            vehicle_params: Optional vehicle parameters dict for visualization

        Returns:
            Path: Path to the generated dashboard file

        Raises:
            FileNotFoundError: If template or required files are not found
            ValueError: If result data is invalid or incomplete
        """
