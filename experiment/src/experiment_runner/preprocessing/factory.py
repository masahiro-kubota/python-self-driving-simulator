import importlib
import inspect
from importlib import metadata
from typing import Any

from core.data import VehicleParameters
from core.utils import get_project_root


class ComponentFactory:
    """Factory for creating components dynamically."""

    def create(
        self,
        component_type: str,
        params: dict[str, Any],
        vehicle_params: VehicleParameters | None = None,
    ) -> Any:
        """Dynamically instantiate a component.

        Args:
            component_type: Component type in "module.ClassName" format
            params: Component parameters
            vehicle_params: Vehicle parameters to inject if provided

        Returns:
            Instantiated component
        """
        # Resolve special parameters
        resolved_params = {}
        path_keys = {"track_path", "model_path", "scaler_path"}

        for key, value in params.items():
            if key in path_keys and isinstance(value, str):
                # User specified custom path
                resolved_params[key] = get_project_root() / value

            # Recursive instantiation for sub-components (e.g. planner/controller in stack)
            elif isinstance(value, dict) and "type" in value and "params" in value:
                sub_type = value["type"]
                sub_params = value["params"]
                # Recursively instantiate, propagating vehicle_params
                resolved_params[key] = self.create(sub_type, sub_params, vehicle_params)
            else:
                resolved_params[key] = value

        # Inject vehicle_params if provided
        if vehicle_params is not None:
            resolved_params["vehicle_params"] = vehicle_params

        cls = None
        # 1. Try resolving via Entry Points
        if "." not in component_type:
            for group in ["ad_components", "simulator"]:
                # Python 3.10+ usage
                eps = metadata.entry_points(group=group)
                # Filter by name
                matches = [ep for ep in eps if ep.name == component_type]
                if matches:
                    cls = matches[0].load()
                    break

        # 2. Fallback to module path
        if cls is None:
            try:
                module_name, class_name = component_type.rsplit(".", 1)
            except ValueError:
                raise ValueError(
                    f"Invalid component type: {component_type}. "
                    "Must be in 'Entry Point' or 'module.ClassName' format."
                ) from None

            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)

        # Filter arguments based on __init__ signature
        sig = inspect.signature(cls.__init__)
        valid_params = {}

        for param_name in sig.parameters:
            if param_name == "self":
                continue
            if param_name in resolved_params:
                valid_params[param_name] = resolved_params[param_name]

            # Special case: var_keyword (**kwargs)
            if sig.parameters[param_name].kind == inspect.Parameter.VAR_KEYWORD:
                valid_params.update(resolved_params)
                break

        return cls(**valid_params)
