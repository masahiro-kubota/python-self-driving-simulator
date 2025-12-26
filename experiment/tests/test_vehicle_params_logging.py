from unittest.mock import patch

from core.utils import get_project_root


@patch("experiment.engine.base.mlflow")
@patch("experiment.engine.evaluator.mlflow")
def test_metadata_injection(_mock_mlflow_eval, _mock_mlflow_base) -> None:
    """Test that vehicle parameters are injected into log metadata."""
    # This test validates that the orchestrator properly validates Hydra config
    # before starting simulation (fail-fast behavior)
    from experiment.core.orchestrator import ExperimentOrchestrator
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    workspace_root = get_project_root()
    config_dir = str(workspace_root / "experiment/conf")

    # Test 1: Verify fail-fast behavior - config validation happens before simulation
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=evaluation",
                "execution.duration_sec=0.1",
            ],
        )

        # Manually resolve hydra interpolations for testing
        OmegaConf.set_struct(cfg, False)
        tmp_path = workspace_root / "tmp"
        tmp_path.mkdir(exist_ok=True)
        cfg["hydra"] = {"runtime": {"output_dir": str(tmp_path)}}

        # Replace ${hydra:runtime.output_dir} in postprocess config
        if "postprocess" in cfg and "mcap" in cfg.postprocess:
            cfg.postprocess.mcap.output_dir = str(tmp_path)

        # Replace ${hydra:runtime.output_dir} in Logger params
        nodes_iter = (
            cfg.system.nodes.values() if isinstance(cfg.system.nodes, dict) else cfg.system.nodes
        )
        for node in nodes_iter:
            node_name = node.get("name") if isinstance(node, dict) else getattr(node, "name", None)
            if node_name == "Logger" and "output_mcap_path" in (
                node.get("params", {}) if isinstance(node, dict) else node.params
            ):
                if isinstance(node, dict):
                    node["params"]["output_mcap_path"] = str(tmp_path)
                else:
                    node.params.output_mcap_path = str(tmp_path)

        OmegaConf.set_struct(cfg, True)

        # This should validate config and fail fast if there are issues
        # (before any simulation starts)
        orchestrator = ExperimentOrchestrator()

        # Verify that config resolution works (fail-fast validation)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        assert cfg_dict is not None
        assert "vehicle" in cfg_dict
        assert "wheelbase" in cfg_dict["vehicle"]

        # Now run the actual experiment
        result = orchestrator.run_from_hydra(cfg)

        # Test 2: Verify metadata injection
        assert len(result.simulation_results) > 0
        log = result.simulation_results[0].log
        assert log is not None
        metadata = log.metadata

        # Check for vehicle params
        assert "wheelbase" in metadata
        assert "width" in metadata
        assert "front_overhang" in metadata
        assert "rear_overhang" in metadata

        # Check values match default vehicle.yaml approx
        assert abs(metadata["wheelbase"] - 1.087) < 1e-3
        assert abs(metadata["width"] - 1.3) < 1e-3
        assert abs(metadata["front_overhang"] - 0.467) < 1e-3
        assert abs(metadata["rear_overhang"] - 0.510) < 1e-3
