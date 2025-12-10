from pathlib import Path
from unittest.mock import patch

import pytest

from experiment.preprocessing.loader import _recursive_merge, load_experiment_config
from experiment.preprocessing.schemas import ResolvedExperimentConfig


@pytest.fixture
def mock_project_root():
    with patch("experiment.preprocessing.loader.get_project_root") as mock_root:
        mock_root.return_value = Path("/tmp/mock_root")
        yield mock_root


@pytest.fixture
def mock_load_yaml():
    with patch("experiment.preprocessing.loader.load_yaml") as mock_load:
        yield mock_load


def test_recursive_merge():
    base = {"a": 1, "b": {"c": 2}}
    overrides = {"a": 3, "b": {"d": 4}}
    merged = _recursive_merge(base, overrides)
    assert merged == {"a": 3, "b": {"c": 2, "d": 4}}


def test_load_experiment_config(mock_project_root, mock_load_yaml):
    _ = mock_project_root  # Silence unused warning
    # Mock file contents
    experiment_yaml = {
        "experiment": {
            "name": "test_exp",
            "type": "evaluation",
            "system": "systems/test_system.yaml",
            "overrides": {
                "components": {"ad_component": {"params": {"planning": {"lookahead": 10.0}}}}
            },
        }
    }

    system_yaml = {
        "system": {
            "name": "test_system",
            "module": "modules/test_module.yaml",
            "runtime": {"mode": "singleprocess"},
            "simulator_overrides": {"params": {"dt": 0.05}},
        }
    }

    module_yaml = {
        "module": {
            "name": "test_module",
            "components": {
                "ad_component": {
                    "type": "experiment.ad_components.StandardADComponent",
                    "params": {
                        "planning": {"type": "p", "params": {"lookahead": 5.0}},
                        "control": {"type": "c", "params": {}},
                    },
                },
                "simulator": {"type": "s", "params": {"dt": 0.1}},
            },
        }
    }

    # Configure mock
    def side_effect(path):
        s_path = str(path)
        if "test_system" in s_path:
            return system_yaml
        elif "test_module" in s_path:
            return module_yaml
        else:
            return experiment_yaml

    mock_load_yaml.side_effect = side_effect

    with patch("core.utils.param_loader.load_component_defaults") as mock_defaults:
        mock_defaults.return_value = {"goal_radius": 2.0, "min_elapsed_time": 10.0}

        config = load_experiment_config("experiments/test_exp.yaml")

        assert isinstance(config, ResolvedExperimentConfig)
        assert config.experiment.name == "test_exp"
        # Check runtime injection
        assert config.runtime["mode"] == "singleprocess"
        # Check simulator override injection
        assert config.simulator.params["dt"] == 0.05
        # Check component structure
        assert "planning" in config.components.ad_component.params
        # Check override application (lookahead 10.0 overrides 5.0)
        assert config.components.ad_component.params["planning"]["lookahead"] == 10.0

        # Check supervisor config
        assert config.supervisor is not None
        # Default value from mock
        assert config.supervisor.params["min_elapsed_time"] == 10.0
        # Overridden value if logic implemented correctly (wait, test_exp doesn't override supervisor)
        # Let's check logic: goal_radius might be overridden by execution override?
        # In test_exp overrides: {"components": ...} only.
        pass


def test_supervisor_config_integration(mock_project_root, mock_load_yaml):
    """Test supervisor configuration loading and overrides."""
    _ = mock_project_root

    experiment_yaml = {
        "experiment": {
            "name": "sup_test",
            "type": "evaluation",
            "system": "systems/test.yaml",
            "overrides": {
                "supervisor": {"max_steps": 500}  # Direct override
            },
        }
    }

    system_yaml = {"system": {"name": "sys", "module": "modules/mod.yaml"}}

    module_yaml = {
        "module": {
            "name": "mod",
            "components": {
                "ad_component": {"type": "abc", "params": {}},
                "simulator": {"type": "sim", "params": {}},
            },
        }
    }

    def side_effect(path):
        s_path = str(path)
        if "system" in s_path:
            return system_yaml
        elif "module" in s_path:
            return module_yaml
        return experiment_yaml

    mock_load_yaml.side_effect = side_effect

    with patch("core.utils.param_loader.load_component_defaults") as mock_defaults:
        # Mock default supervisor params
        mock_defaults.return_value = {"goal_radius": 5.0, "min_elapsed_time": 20.0, "goal_x": None}

        config = load_experiment_config("exp.yaml")

        assert config.supervisor is not None
        params = config.supervisor.params

        # Check defaults
        assert params["min_elapsed_time"] == 20.0

        # Check direct override
        assert params["max_steps"] == 500
