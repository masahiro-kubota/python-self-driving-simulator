"""Generic ADComponent implementation."""

import importlib
import inspect
from typing import Any, TypeVar

from core.interfaces import ADComponent, Controller, Planner

T = TypeVar("T")


class GenericADComponent(ADComponent):
    """Generic ADComponent that loads planner and controller from packages."""

    def _create_planner(self, **kwargs: Any) -> Planner:
        """Create planner from package.

        Args:
            **kwargs:
                - planner_package: Package name containing Planner implementation
                - other planner parameters

        Returns:
            Planner instance
        """
        package_name = kwargs.get("planner_package")
        if not package_name:
            raise ValueError("planner_package must be specified")

        return self._create_instance_from_package(Planner, package_name, **kwargs)

    def _create_controller(self, **kwargs: Any) -> Controller:
        """Create controller from package.

        Args:
            **kwargs:
                - controller_package: Package name containing Controller implementation
                - other controller parameters

        Returns:
            Controller instance
        """
        package_name = kwargs.get("controller_package")
        if not package_name:
            raise ValueError("controller_package must be specified")

        return self._create_instance_from_package(Controller, package_name, **kwargs)

    def _create_instance_from_package(
        self, base_cls: type[T], package_name: str, **kwargs: Any
    ) -> T:
        """Create instance of a subclass of base_cls found in package_name."""
        # Load package
        try:
            module = importlib.import_module(package_name)
        except ImportError:
            # Try replacing hyphens with underscores (e.g. pure-pursuit -> pure_pursuit)
            module_name = package_name.replace("-", "_")
            module = importlib.import_module(module_name)

        # Find implementation class
        target_cls = self._find_subclass(module, base_cls)
        if not target_cls:
            raise ValueError(
                f"No {base_cls.__name__} implementation found in package {package_name}"
            )

        # Instantiate with filtered arguments
        return self._instantiate_with_params(target_cls, **kwargs)

    def _find_subclass(self, module: Any, base_cls: type[T]) -> type[T] | None:
        """Find a subclass of base_cls in the module."""
        # Check __all__ first
        if hasattr(module, "__all__"):
            for name in module.__all__:
                obj = getattr(module, name)
                if isinstance(obj, type) and issubclass(obj, base_cls) and obj is not base_cls:
                    return obj

        # Scan module members
        for name, obj in inspect.getmembers(module):
            if isinstance(obj, type) and issubclass(obj, base_cls) and obj is not base_cls:
                return obj
        return None

    def _instantiate_with_params(self, cls: type[T], **kwargs: Any) -> T:
        """Instantiate class with arguments matching its signature."""
        sig = inspect.signature(cls.__init__)
        params = {}

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            # Inject vehicle_params if requested
            if name == "vehicle_params" and self.vehicle_params is not None:
                params[name] = self.vehicle_params
                continue

            # Pass generic kwargs if accepting **kwargs
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                # If the class accepts **kwargs, pass everything
                # Note: this might duplicate parameters if we already extracted some
                remaining_kwargs = {k: v for k, v in kwargs.items() if k not in params}
                params.update(remaining_kwargs)
                break

            # Extract from kwargs if available
            if name in kwargs:
                params[name] = kwargs[name]

        # Instantiate class
        instance = cls(**params)

        # Additional logic for Pure Pursuit track auto-loading
        track_path = kwargs.get("track_path")

        # Check if track_path needs default resolution (only for pure_pursuit module)
        if track_path is None and "pure_pursuit" in cls.__module__:
            try:
                from pathlib import Path

                import pure_pursuit

                pkg_root = Path(pure_pursuit.__file__).parent
                default_track = pkg_root / "data/tracks/raceline_awsim_15km.csv"
                if default_track.exists():
                    track_path = default_track
            except ImportError:
                pass

        if track_path and hasattr(instance, "set_reference_trajectory"):
            try:
                from pathlib import Path

                from planning_utils import load_track_csv  # type: ignore

                track = load_track_csv(Path(track_path))
                instance.set_reference_trajectory(track)
            except (ImportError, ValueError) as e:
                print(f"Warning: Failed to load track: {e}")

        return instance
