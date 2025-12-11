"""Configuration loader for module/system/experiment layers."""

import logging
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

logger = logging.getLogger(__name__)

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


def _resolve_defaults(
    user_params: dict[str, Any], default_params: dict[str, Any]
) -> dict[str, Any]:
    """Resolve 'default' values in user parameters using default parameters.

    Args:
        user_params: User-provided parameters.
        default_params: Default parameters.

    Returns:
        Resolved parameters.
    """
    resolved = user_params.copy()
    for key, value in resolved.items():
        if (
            isinstance(value, dict)
            and key in default_params
            and isinstance(default_params[key], dict)
        ):
            resolved[key] = _resolve_defaults(value, default_params[key])
        elif value == "default":
            if key in default_params:
                resolved[key] = default_params[key]
            else:
                # If "default" is specified but no default exists, we can either error or leave it
                # (and let Pydantic fail validation if types don't match).
                # Erroring is explicit safer.
                raise ValueError(f"Parameter '{key}' set to 'default' but no default value found.")
    return resolved


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


def _load_layer_configs(
    path: Path | str,
) -> tuple[ExperimentLayerConfig, SystemConfig, ModuleConfig]:
    """Load and validate all 3 configuration layers.

    Args:
        path: Path to the experiment layer configuration file.

    Returns:
        Tuple of (experiment_layer, system_layer, module_layer).
    """
    # Load Experiment Layer
    exp_data = load_yaml(path)
    if "experiment" not in exp_data:
        raise ValueError(f"Missing 'experiment' key in {path}")
    experiment_layer = ExperimentLayerConfig(**exp_data["experiment"])

    # Load System Layer
    system_data = load_yaml(experiment_layer.system)
    if "system" not in system_data:
        raise ValueError(f"Missing 'system' key in {experiment_layer.system}")
    system_layer = SystemConfig(**system_data["system"])

    # Load Module Layer
    module_data = load_yaml(system_layer.module)
    if "module" not in module_data:
        raise ValueError(f"Missing 'module' key in {system_layer.module}")
    module_layer = ModuleConfig(**module_data["module"])

    return experiment_layer, system_layer, module_layer


def _resolve_simulator_config(
    system_layer: SystemConfig, module_layer: ModuleConfig
) -> dict[str, Any]:
    """Resolve simulator configuration from system/module layers.

    Args:
        system_layer: System configuration.
        module_layer: Module configuration.

    Returns:
        Resolved simulator configuration dict.
    """
    from importlib import metadata

    from core.utils.param_loader import load_component_defaults

    # Check System Layer first, then Module Layer
    resolved_simulator = None
    if system_layer.simulator:
        resolved_simulator = system_layer.simulator
    elif "simulator" in module_layer.components:
        resolved_simulator = module_layer.components["simulator"]

    if resolved_simulator is None:
        raise ValueError("Simulator configuration missing (must be in System or Module)")

    # Load defaults for simulator
    sim_user_params = resolved_simulator.get("params", {}) or {}
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

    resolved_simulator["params"] = _resolve_defaults(sim_user_params, sim_defaults)

    return resolved_simulator


def _inject_system_overrides(
    resolved_simulator: dict[str, Any],
    resolved_components: dict[str, Any],
    system_layer: SystemConfig,
) -> None:
    """Inject system-level overrides into simulator and components.

    Args:
        resolved_simulator: Resolved simulator configuration (modified in-place).
        resolved_components: Resolved components configuration (modified in-place).
        system_layer: System configuration.
    """
    sim_params = resolved_simulator.get("params", {})

    # Inject map path
    if system_layer.map_path:
        full_map_path = str(get_project_root() / system_layer.map_path)

        # Inject into Simulator
        sim_params["map_path"] = full_map_path

        # Inject into AD Component
        if "params" not in resolved_components["ad_component"]:
            resolved_components["ad_component"]["params"] = {}
        resolved_components["ad_component"]["params"]["map_path"] = full_map_path

    # Apply simulator_overrides from System
    if system_layer.simulator_overrides and "params" in system_layer.simulator_overrides:
        sim_params = _recursive_merge(sim_params, system_layer.simulator_overrides["params"])
        resolved_simulator["params"] = sim_params


def _resolve_vehicle_config(system_layer: SystemConfig, sim_params: dict[str, Any]) -> None:
    """Load vehicle configuration and inject into simulator params.

    Args:
        system_layer: System configuration.
        sim_params: Simulator parameters (modified in-place).
    """
    if system_layer.vehicle and "config_path" in system_layer.vehicle:
        v_path = get_project_root() / system_layer.vehicle["config_path"]
        v_params = VehicleParameters(**load_yaml(v_path))
        sim_params["vehicle_params"] = v_params


def _resolve_supervisor_config(experiment_layer: ExperimentLayerConfig) -> dict[str, Any]:
    """Resolve supervisor configuration from system and experiment layers.

    Args:
        experiment_layer: Experiment configuration.

    Returns:
        Resolved supervisor configuration dict.
    """
    from core.utils.param_loader import load_component_defaults

    # Start with system layer supervisor config
    # Note: We need to access the raw dict to get supervisor config
    system_data = load_yaml(experiment_layer.system)
    supervisor_user_params = system_data.get("system", {}).get("supervisor", {})

    # Apply experiment layer supervisor overrides
    if experiment_layer.supervisor:
        supervisor_user_params = _recursive_merge(
            supervisor_user_params, experiment_layer.supervisor
        )

    supervisor_defaults = load_component_defaults("supervisor")
    supervisor_params = _resolve_defaults(supervisor_user_params, supervisor_defaults)

    return {"params": supervisor_params}


def _build_final_config(
    experiment_layer: ExperimentLayerConfig,
    resolved_components: dict[str, Any],
    resolved_simulator: dict[str, Any],
    supervisor_config: dict[str, Any],
) -> ResolvedExperimentConfig:
    """Build final resolved experiment configuration.

    Args:
        experiment_layer: Experiment configuration.
        resolved_components: Resolved components configuration.
        resolved_simulator: Resolved simulator configuration.
        supervisor_config: Resolved supervisor configuration.

    Returns:
        Complete resolved experiment configuration.
    """
    final_dict = {
        "experiment": {
            "name": experiment_layer.name,
            "type": experiment_layer.type,
            "description": experiment_layer.description,
        },
        "components": resolved_components,
        "simulator": resolved_simulator,
        "supervisor": supervisor_config,
        "postprocess": {},
    }

    # Apply postprocess config
    if experiment_layer.postprocess:
        final_dict["postprocess"] = experiment_layer.postprocess.model_dump()

    # Apply execution config
    if experiment_layer.execution:
        final_dict["execution"] = experiment_layer.execution.model_dump()

    return ResolvedExperimentConfig(**final_dict)


def load_experiment_config(path: Path | str) -> ResolvedExperimentConfig:
    """Load and merge experiment configuration layers.

    Args:
        path: Path to the experiment layer content file (ExperimentLayerConfig).

    Returns:
        Resolved experiment configuration (ResolvedExperimentConfig).
    """
    # 1. Load all layers
    experiment_layer, system_layer, module_layer = _load_layer_configs(path)

    # 2. Resolve components
    if "ad_component" not in module_layer.components:
        raise ValueError("Module must define 'ad_component' in components")
    resolved_components = {"ad_component": module_layer.components["ad_component"]}

    # 3. Resolve simulator
    resolved_simulator = _resolve_simulator_config(system_layer, module_layer)

    # 4. Apply system overrides
    _inject_system_overrides(resolved_simulator, resolved_components, system_layer)

    # 5. Resolve vehicle
    sim_params = resolved_simulator["params"]
    _resolve_vehicle_config(system_layer, sim_params)

    # 6. Resolve supervisor
    supervisor_config = _resolve_supervisor_config(experiment_layer)

    # 7. Build final config
    return _build_final_config(
        experiment_layer, resolved_components, resolved_simulator, supervisor_config
    )


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

        # Add obstacles if available
        if "obstacles" in sim_params:
            simulator_config["obstacles"] = sim_params["obstacles"]
            logger.info("Adding %d obstacles to SimulatorConfig", len(sim_params["obstacles"]))

        from simulator.simulator import SimulatorConfig

        logger.info("Creating SimulatorConfig with obstacles: %s", "obstacles" in simulator_config)
        simulator_config_model = SimulatorConfig(**simulator_config)
        logger.info(
            "SimulatorConfig created with %d obstacles", len(simulator_config_model.obstacles)
        )
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
