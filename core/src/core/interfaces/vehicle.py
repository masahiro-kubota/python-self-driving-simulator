"""Vehicle parameters repository interface."""

from abc import ABC, abstractmethod
from pathlib import Path

from core.data.vehicle import VehicleParameters


class VehicleParametersRepository(ABC):
    """Interface for vehicle parameters persistence.

    This interface abstracts the storage and retrieval of vehicle parameters,
    allowing different implementations (YAML, JSON, database, etc.) without
    affecting the code that uses vehicle parameters.
    """

    @abstractmethod
    def load(self, path: Path) -> VehicleParameters:
        """Load vehicle parameters from file.

        Args:
            path: Input file path

        Returns:
            VehicleParameters object
        """
        pass
