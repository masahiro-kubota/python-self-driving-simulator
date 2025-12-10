"""Configuration loader for module/system/experiment layers."""

from pathlib import Path
from typing import Any, TypeVar

from core.data import VehicleParameters, VehicleState
from core.utils import get_project_root
from core.utils.config import load_yaml as core_load_yaml
from core.utils.config import merge_configs
from core.utils.node_factory import create_node
from core.validation.node_graph import validate_node_graph
from experiment.preprocessing.schemas import (
    ExperimentLayerConfig,
    ModuleConfig,
    ResolvedExperimentConfig,
    SystemConfig,
)
from experiment.structures import Experiment
from logger import LoggerNode
from supervisor import SupervisorNode

T = TypeVar("T")


def _recursive_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries.

    Args:
        base: The base dictionary.
        overrides: The dictionary with overrides.

    Returns:
        Merged dictionary.

    Note:
        This is a wrapper around core.utils.config.merge_configs for backward compatibility.
    """
    return merge_configs(base, overrides)


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load YAML file relative to project root.

    Args:
        path: Path to YAML file (relative to project root).

    Returns:
        Loaded dictionary.

    Note:
        This is a convenience wrapper that automatically prepends the project root path.
    """
    full_path = get_project_root() / path
    return core_load_yaml(full_path)


def load_experiment_config(path: Path | str) -> ResolvedExperimentConfig:
    """Load and merge experiment configuration layers.

    Args:
        path: Path to the experiment layer content file (ExperimentLayerConfig).

    Returns:
        Resolved experiment configuration (ResolvedExperimentConfig).
    """
    # 1. Load Experiment Layer
    exp_data = load_yaml(path)

    # Normalize structure:
    # Support both flat and nested structures
    # Nested: experiment: {name, type, system, execution, postprocess}
    # Flat: name, type, system, execution, postprocess (at root level)
    config_data = exp_data.copy()

    if "experiment" in config_data and isinstance(config_data["experiment"], dict):
        # Nested structure: flatten it
        exp_content = config_data.pop("experiment")

        # Merge experiment content into root, but preserve root-level overrides
        # Priority: root level > experiment nested level
        for key in [
            "name",
            "type",
            "description",
            "system",
            "execution",
            "postprocess",
            "supervisor",
        ]:
            if key not in config_data and key in exp_content:
                config_data[key] = exp_content[key]

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
            for group in ["ad_components", "simulator"]:
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
    resolved_simulator = None

    # 1. Check System Layer for Simulator definition
    if system_layer.simulator:
        resolved_simulator = system_layer.simulator
    # 2. Check Module Layer (legacy support or default)
    elif "simulator" in module_layer.components:
        resolved_simulator = module_layer.components["simulator"]

    if resolved_simulator is None:
        raise ValueError("Simulator configuration missing (must be in System or Module)")

    # Load Defaults for Simulator
    sim_type = resolved_simulator["type"]
    sim_package = None

    if "." not in sim_type:
        # Entry point lookup
        for group in ["ad_components", "simulator"]:
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

    # 5. Apply System Layer Overrides

    # Simulator params injection
    sim_params = resolved_simulator.get("params", {})

    # Inject Map Path
    if system_layer.map_path:
        # Resolve map path relative to project root
        full_map_path = str(get_project_root() / system_layer.map_path)

        # 1. Inject into Simulator
        sim_params["map_path"] = full_map_path

        # 2. Inject into AD Component (for planning/generic usage)
        if "params" not in resolved_components["ad_component"]:
            resolved_components["ad_component"]["params"] = {}
        resolved_components["ad_component"]["params"]["map_path"] = full_map_path

    # Apply `simulator_overrides` from System
    if system_layer.simulator_overrides and "params" in system_layer.simulator_overrides:
        sim_params = _recursive_merge(sim_params, system_layer.simulator_overrides["params"])

    # 6. Apply Experiment Layer Overrides

    base_config = {
        "experiment": {
            "name": experiment_layer.name,
            "type": experiment_layer.type,
            "description": experiment_layer.description,
        },
        "components": resolved_components,
        "simulator": resolved_simulator,
        "postprocess": {},  # Defaults (will be overridden if experiment_layer.postprocess exists)
    }

    # Apply System Runtime
    if system_layer.runtime:
        base_config["runtime"] = system_layer.runtime

    # --- APPLYING EXPERIMENT LAYER CONFIG ---

    # Start with base config
    final_dict = base_config

    # Apply postprocess config if specified
    if experiment_layer.postprocess:
        final_dict["postprocess"] = experiment_layer.postprocess.model_dump()

    # Apply execution config if specified
    if experiment_layer.execution:
        final_dict["execution"] = experiment_layer.execution.model_dump()

    # --- RESOLVE PATHS (Vehicle) ---

    # Resolve vehicle
    if system_layer.vehicle and "config_path" in system_layer.vehicle:
        # Load from file
        v_path = get_project_root() / system_layer.vehicle["config_path"]
        v_params = VehicleParameters(**load_yaml(v_path))

        # Inject into simulator params
        sim_params["vehicle_params"] = v_params

    # Update simulator params in final dict
    if final_dict["simulator"] is None:
        raise ValueError("Simulator configuration missing")

    final_dict["simulator"]["params"] = sim_params

    # --- Supervisor ---

    # Load defaults from supervisor package
    supervisor_defaults = load_component_defaults("supervisor")

    # Get user overrides from execution config if any (e.g. goal_radius)
    # Note: legacy config might put goal_radius in execution
    # We should merge them into supervisor params

    supervisor_params = supervisor_defaults.copy()

    # Apply user-provided supervisor overrides if any
    if experiment_layer.supervisor:
        supervisor_params = _recursive_merge(supervisor_params, experiment_layer.supervisor)

    final_dict["supervisor"] = {"params": supervisor_params}

    # 7. Build Object

    return ResolvedExperimentConfig(**final_dict)


class DefaultPreprocessor:
    """前処理の具体的な実装

    インターフェースではなく、具体的なクラスとして実装。
    全実験タイプで共通の処理を行う。
    """

    def __init__(self) -> None:
        from experiment.preprocessing.factory import ComponentFactory

        self.component_factory = ComponentFactory()

    def create_experiment(self, config_path: Path) -> Experiment:
        """Create experiment instance from configuration file.

        Args:
            config_path: Path to the experiment configuration file.

        Returns:
            Executable Experiment instance.
        """
        import uuid

        # 1. Load configuration
        config = self.load_config(config_path)

        # 2. Create nodes (Simulator, AD, etc.)
        nodes = self._create_nodes(config)

        # 3. Create Experiment instance
        experiment_id = str(uuid.uuid4())

        return Experiment(
            id=experiment_id,
            type=config.experiment.type,
            config=config,
            nodes=nodes,
        )

    def load_config(self, config_path: Path) -> ResolvedExperimentConfig:
        """YAML設定を読み込み、階層マージしてスキーマに変換"""
        return load_experiment_config(config_path)

    def _create_nodes(self, config: ResolvedExperimentConfig) -> list[Any]:
        """Create experiment nodes (Simulator, AD Component, etc.)."""
        # Dispatch based on experiment type (or currently just default to evaluation structure)
        # In a more advanced setup, this would use a NodeConstructionStrategy

        workspace_root = get_project_root()
        sim_params = config.simulator.params.copy()

        # 1. Load Vehicle Parameters
        vehicle_params = None
        if "vehicle_config" in sim_params:
            config_path = sim_params.pop("vehicle_config")
            full_path = workspace_root / config_path

            if not full_path.exists():
                # Try relative to project if not found? Or assume relative?
                # Usually relative to project root.
                pass

            if not full_path.exists():
                raise FileNotFoundError(f"Vehicle config not found: {full_path}")

            from core.utils.config import load_yaml as load_yaml_file

            vehicle_params = VehicleParameters(**load_yaml_file(full_path))
            sim_params["vehicle_params"] = vehicle_params
        else:
            if "vehicle_params" in sim_params and isinstance(sim_params["vehicle_params"], dict):
                vehicle_params = VehicleParameters(**sim_params["vehicle_params"])
            elif "vehicle_params" in sim_params and isinstance(
                sim_params["vehicle_params"], VehicleParameters
            ):
                vehicle_params = sim_params["vehicle_params"]
            else:
                vehicle_params = VehicleParameters()
                sim_params["vehicle_params"] = vehicle_params

        # 2. Setup AD Nodes
        comp_config = config.components
        # We ignore ad_component.type now (or treat it as metadata)
        # We expect a list of nodes in params
        ad_params = comp_config.ad_component.params

        ad_nodes = []
        if "nodes" in ad_params:
            for node_cfg in ad_params["nodes"]:
                if "type" not in node_cfg:
                    raise ValueError(f"Node config missing 'type': {node_cfg}")

                node = create_node(
                    node_type=node_cfg["type"],
                    rate_hz=node_cfg.get("rate_hz", 10.0),
                    params=node_cfg.get("params", {}),
                    vehicle_params=vehicle_params,
                )
                ad_nodes.append(node)

        if ad_nodes:
            validate_node_graph(ad_nodes)

        # 3. Setup Simulator
        sim_rate = config.simulator.rate_hz

        # Simulator is now a Node, so we instantiate it directly
        from simulator.simulator import Simulator

        # Prepare simulator config
        # Get initial_state from sim_params if available
        initial_state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0, timestamp=0.0)
        if "initial_state" in sim_params:
            initial_state_dict = sim_params.pop("initial_state")
            if isinstance(initial_state_dict, dict):
                initial_state = VehicleState(**initial_state_dict, timestamp=0.0)
            elif isinstance(initial_state_dict, VehicleState):
                initial_state = initial_state_dict

        simulator_config = {
            "vehicle_params": vehicle_params,
            "initial_state": initial_state,
        }

        # Add map_path if available
        if "map_path" in sim_params:
            map_path_str = sim_params["map_path"]
            # Resolve relative to workspace root
            if not map_path_str.startswith("/"):
                map_path_str = str(workspace_root / map_path_str)
            simulator_config["map_path"] = map_path_str

        from simulator.simulator import SimulatorConfig

        simulator_config_model = SimulatorConfig(**simulator_config)
        simulator = Simulator(config=simulator_config_model, rate_hz=sim_rate)

        nodes = []

        # 4. Add Simulator as first node
        nodes.append(simulator)

        # 5. AD Component Nodes
        nodes.extend(ad_nodes)

        # 6. Supervisor Node
        if config.supervisor:
            from supervisor import SupervisorConfig

            supervisor_params = config.supervisor.params
            supervisor_config_model = SupervisorConfig(**supervisor_params)
            supervisor = SupervisorNode(
                config=supervisor_config_model,
                rate_hz=sim_rate,
            )
            nodes.append(supervisor)

        # 7. Logger Node
        # Always add logger for evaluation?
        nodes.append(LoggerNode(rate_hz=sim_rate))

        return nodes
