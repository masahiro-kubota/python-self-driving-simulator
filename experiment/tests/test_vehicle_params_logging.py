from core.utils import get_project_root


def test_metadata_injection() -> None:
    """Test that vehicle parameters are injected into log metadata."""
    workspace_root = get_project_root()
    config_path = workspace_root / "experiment/configs/experiments/default_experiment.yaml"

    # Load config and override duration to be short
    # We can't easily override config file without writing a new one,
    # but we can modify the object if we use the runner directly.
    # For Orchestrator, we pass the path.

    # Let's use the Orchestrator but expect it to run the full config.
    # To avoid long wait, we can assume the duration in config is short 20s.
    # Or strict test: create a temporary config file with short duration.

    # Creating a temp config
    import yaml

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    # Set short duration
    if "execution" not in config_data:
        config_data["execution"] = {}
    config_data["execution"]["duration_sec"] = 0.1  # 1 step basically

    # Save to tmp
    # We need to make sure paths in config are correct relative to project root
    # Since we run from tmp, relative paths might break if not handled carefully.
    # But loader handles project root relative paths.

    # Actually, let's just use the existing config and run for 1 step by mocking?
    # Or just rely on the existing config not being too long?
    # Pure pursuit config is 20s. That's a bit long for a unit test.

    # Let's instantiate components directly to avoid writing files.

    from experiment.preprocessing.loader import DefaultPreprocessor
    from experiment.runner.evaluation_runner import EvaluationRunner

    preprocessor = DefaultPreprocessor()
    experiment = preprocessor.create_experiment(config_path)

    # Hack: modify duration in the loaded experiment config object
    if experiment.config.execution:
        experiment.config.execution.duration_sec = 0.1

    runner = EvaluationRunner()
    result = runner.run(experiment)

    assert result.log is not None
    metadata = result.log.metadata

    # Check for vehicle params
    assert "wheelbase" in metadata
    assert "width" in metadata
    assert "front_overhang" in metadata
    assert "rear_overhang" in metadata

    # Check values match default_vehicle.yaml approx
    # wheelbase: 1.087
    assert abs(metadata["wheelbase"] - 1.087) < 1e-3
    assert abs(metadata["width"] - 1.3) < 1e-3
    assert abs(metadata["front_overhang"] - 0.467) < 1e-3
    assert abs(metadata["rear_overhang"] - 0.510) < 1e-3
