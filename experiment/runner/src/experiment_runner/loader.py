"""Configuration loader for module/system/experiment layers."""

from pathlib import Path
from typing import Any, TypeVar

import yaml

from core.utils import get_project_root
from experiment_runner.config import (
    ExperimentLayerConfig,
    ModuleConfig,
    ResolvedExperimentConfig,
    SystemConfig,
)

T = TypeVar("T")


def _recursive_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries.

    Args:
        base: The base dictionary.
        overrides: The dictionary with overrides.

    Returns:
        Merged dictionary.
    """
    merged = base.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = _recursive_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load YAML file.

    Args:
        path: Path to YAML file.

    Returns:
        Loaded dictionary.
    """
    path = get_project_root() / path
    with open(path) as f:
        return yaml.safe_load(f)


def load_experiment_config(path: Path | str) -> ResolvedExperimentConfig:
    """Load and merge experiment configuration layers.

    Args:
        path: Path to the experiment layer content file (ExperimentLayerConfig).

    Returns:
        Resolved experiment configuration (ResolvedExperimentConfig).
    """
    # 1. Load Experiment Layer
    exp_data = load_yaml(path)
    # Handle the case where the wrapper key "experiment" is used or not
    # The existing files usually have root keys like experiment, components, etc.
    # But new structure expects root keys: experiment, system, overrides.
    # Let's assume the passed YAML follows the new ExperimentLayerConfig structure schema directly
    # or wrapped in an "experiment" key. Pydantic can handle direct mapping.
    # To be safe and flexible, let's look for known fields.

    # Normalize structure:
    # If "experiment" key exists and contains metadata (name, type, etc.), flatten it into root
    # so ExperimentLayerConfig can validate it.
    config_data = exp_data.copy()
    if "experiment" in config_data and isinstance(config_data["experiment"], dict):
        exp_meta = config_data.pop("experiment")
        config_data.update(exp_meta)

    experiment_layer = ExperimentLayerConfig(**config_data)

    # 2. Load System Layer
    system_data = load_yaml(experiment_layer.system)
    # Similar normalization for system?
    # User example 3-2:
    # system:
    #   name: ...
    #   module: ...
    if "system" in system_data and isinstance(system_data["system"], dict):
        system_config_data = system_data["system"]
    else:
        system_config_data = system_data

    system_layer = SystemConfig(**system_config_data)

    # 3. Load Module Layer
    module_data = load_yaml(system_layer.module)
    if "module" in module_data and isinstance(module_data["module"], dict):
        module_config_data = module_data["module"]
    else:
        module_config_data = module_data

    module_layer = ModuleConfig(**module_config_data)

    # 4. Construct Base Configuration from Module
    # The module defines 'components', which maps to ResolvedExperimentConfig.components (mostly)
    # but structure needs alignment.
    # Module: components -> { input:..., perception:..., planning:..., control:..., simulator:... }
    # Resolved: components -> { ad_component: { type:..., params:... } }, simulator: { type:..., params:... }

    # We need to transform the Module Components into the Resolved structure.
    # This might require some mapping logic, as Module groups planning/control under "components"
    # while Resolved separates simulator and wraps planning/control in ad_component?
    #
    # Wait, looking at Config.py:
    # ComponentsConfig has `ad_component`.
    # SimulatorConfig is separate.
    #
    # User example 3-1:
    # module:
    #   components:
    #     perception: ...
    #     planning: ...
    #     control: ...
    #     simulator: ...
    #
    # We need to map `planning` and `control` (and `perception`) into `ad_component`.
    # And `simulator` to `simulator`.

    # Let's start with an empty dict for the resolved config and build it up.

    # --- Components (AD Component) ---
    # If module defines ad_component directly, use it
    # Otherwise fail (no legacy support)

    from importlib import metadata

    from core.utils.param_loader import load_component_defaults

    if "ad_component" in module_layer.components:
        # Direct ad_component definition
        resolved_components = {"ad_component": module_layer.components["ad_component"]}

        # Load defaults for the ad_component type
        ad_comp_type = resolved_components["ad_component"]["type"]
        ad_package = None

        if "." not in ad_comp_type:
            # Entry point lookup
            for group in ["ad_components", "simulators"]:
                eps = metadata.entry_points(group=group)
                matches = [ep for ep in eps if ep.name == ad_comp_type]
                if matches:
                    ad_package = matches[0].value.split(":")[0].split(".")[0]
                    break
        else:
            ad_package = ad_comp_type.split(".")[0]

        ad_defaults = {}
        if ad_package:
            ad_defaults = load_component_defaults(ad_package)

        ad_user_params = resolved_components["ad_component"].get("params", {}) or {}
        resolved_components["ad_component"]["params"] = _recursive_merge(
            ad_defaults, ad_user_params
        )
    else:
        raise ValueError("Module must define 'ad_component' in components")

    # --- Simulator ---
    if "simulator" in module_layer.components:
        resolved_simulator = module_layer.components["simulator"]

        # Load Defaults for Simulator
        sim_type = resolved_simulator["type"]
        sim_package = None

        if "." not in sim_type:
            # Entry point lookup
            for group in ["ad_components", "simulators"]:
                eps = metadata.entry_points(group=group)
                matches = [ep for ep in eps if ep.name == sim_type]
                if matches:
                    sim_package = matches[0].value.split(":")[0].split(".")[0]
                    break
        else:
            sim_package = sim_type.split(".")[0]

        sim_defaults = {}
        if sim_package:
            sim_defaults = load_component_defaults(sim_package)

        sim_user_params = resolved_simulator.get("params", {}) or {}
        resolved_simulator["params"] = _recursive_merge(sim_defaults, sim_user_params)

    else:
        raise ValueError("Module must define 'simulator' component")

    # 5. Apply System Layer Overrides

    # Simulator params injection
    sim_params = resolved_simulator.get("params", {})

    # Inject Vehicle Config
    if system_layer.vehicle:
        # If config_path is present, we keep it as a param "vehicle_config" to be resolved by Runner
        # OR we resolve it here?
        # User plan said: "Resolve vehicle.config_path -> Load vehicle config -> Inject"
        # Since we are in loader, let's leave the path resolution to Runner (as it has the logic for YamlVehicleRepository)
        # OR better: resolve it here if simple, but Runner has the repository logic.
        # Actually, let's follow the plan: "Resolve vehicle.config_path -> Load vehicle config -> Inject"
        # Check if we can import the repository here.
        # `from experiment_runner.yaml_vehicle_repository import YamlVehicleParametersRepository`
        # It seems cleaner to do it here so Runner receives a clean object.
        pass  # Will do below.

    # Inject Scene Config
    if system_layer.scene:
        # Mapping: scene.config_path -> simulator.params.map_path (if using scene config for map)
        # User example: scene: { config_path: ..., use_case: ... }
        # The scene config file likely contains the map path or IS the map metadata?
        # Current logic checks "map_path" in simulator params.
        # Let's support passing the scene config path to simulator, or resolving it.
        # To match user request "Resolve scene.config_path -> Inject into simulator.params.map_path",
        # we need to know what's in default_scene.yaml.
        # But for now let's assume we pass the parameters from system.scene to simulator params
        # or do specific mapping.

        # Specific override for dt, initial_state from system
        pass

    # Apply `simulator_overrides` from System
    if system_layer.simulator_overrides and "params" in system_layer.simulator_overrides:
        sim_params = _recursive_merge(sim_params, system_layer.simulator_overrides["params"])

    # 6. Apply Experiment Layer Overrides
    # `overrides` dict structure is likely matching the Resolved structure?
    # User example 3-3:
    # overrides:
    #   components:
    #     planning:
    #   execution: ...

    # Wait, `components.planning` in logical structure maps to `resolved.components.ad_component.params.planning`.
    # We need to map these overrides intelligently.

    # Let's construct the initial "resolved dict" and then apply overrides.

    base_config = {
        "experiment": {
            "name": experiment_layer.name,
            "type": experiment_layer.type,
            "description": experiment_layer.description,
        },
        "components": resolved_components,
        "simulator": resolved_simulator,
        "logging": {},  # Defaults
        # execution, training etc. come from defaults or overrides
    }

    # Apply System Runtime to Execution?
    # User example: runtime: { mode: "singleprocess" }
    # Maps to ResolvedExperimentConfig.runtime
    if system_layer.runtime:
        base_config["runtime"] = system_layer.runtime

    # --- MERGING OVERRIDES ---

    # Experiment overrides provided by user look like:
    # components:
    #   planning: ...
    #
    # But our internal structure is:
    # components:
    #   ad_component:
    #     params:
    #       planning: ...

    # We need a remapping helper for overrides if they use the "logical" simplified path.
    # Or we assume the user provides overrides in the "Resolved" structure format?
    # User example 3-3:
    # overrides:
    #   components:
    #     planning: ...
    #
    # This strongly suggests we need to remap `components.X` -> `components.ad_component.params.X`

    overrides = experiment_layer.overrides

    # Start merging
    final_dict = base_config

    # 6.1 Handle Component Overrides
    if "components" in overrides:
        comp_ops = overrides.pop("components")

        # Merge ad_component overrides directly
        if "ad_component" in comp_ops:
            final_dict["components"]["ad_component"] = _recursive_merge(
                final_dict["components"]["ad_component"], comp_ops["ad_component"]
            )
        else:
            raise ValueError(
                "Component overrides must be nested under 'ad_component' key. "
                "Legacy top-level component keys are no longer supported."
            )

    # 6.2 Handle other overrides (Execution, Logging, etc.)
    # execution, evaluation, logging can satisfy direct merge if structure matches.
    # User example:
    # execution: { num_episodes: 5 ... } -> matches Resolved
    # evaluation: { metrics: ... } -> matches Resolved
    # logging: ... -> matches Resolved

    final_dict = _recursive_merge(final_dict, overrides)

    # --- RESOLVE PATHS (Vehicle & Scene) ---

    # Resolve vehicle
    if system_layer.vehicle and "config_path" in system_layer.vehicle:
        # Load from file
        v_path = get_project_root() / system_layer.vehicle["config_path"]
        from experiment_runner.yaml_vehicle_repository import YamlVehicleParametersRepository

        repo = YamlVehicleParametersRepository()
        v_params = repo.load(v_path)

        # Inject into simulator params
        sim_params["vehicle_params"] = v_params

    # Resolve scene
    if system_layer.scene and "config_path" in system_layer.scene:
        # Assume config path points to a file that contains 'map_path' or IS the map info
        # For default_scene.yaml, user implies it has 'use_case'.
        # If it's a dedicated scene config, load it.
        s_path = get_project_root() / system_layer.scene["config_path"]
        s_data = load_yaml(s_path)

        # If scene config has "map_path" override?
        # Or if the scene config IS the definition of the scene parameters.
        # User example:
        # system:
        #   scene:
        #     config_path: "experiment/configs/scenes/default_scene.yaml"
        #
        # If default_scene.yaml has:
        # map_path: "..."
        # Then we inject it.

        if "map_path" in s_data:
            # Resolve map path relative to project root if string
            sim_params["map_path"] = str(get_project_root() / s_data["map_path"])
        elif "scene" in s_data and "map_path" in s_data["scene"]:
            sim_params["map_path"] = str(get_project_root() / s_data["scene"]["map_path"])

    # Update simulator params in final dict
    if final_dict["simulator"] is None:
        # Should not accept null simulator
        raise ValueError("Simulator configuration missing")

    final_dict["simulator"]["params"] = sim_params

    # 7. Build Object
    return ResolvedExperimentConfig(**final_dict)
