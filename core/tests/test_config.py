"""Tests for configuration utilities."""

import tempfile
from pathlib import Path

import pytest
from core.utils.config import (
    get_nested_value,
    load_yaml,
    merge_configs,
    save_yaml,
    set_nested_value,
)


class TestLoadSaveYAML:
    """Tests for YAML loading and saving."""

    def test_save_and_load(self) -> None:
        """Test saving and loading YAML file."""
        data = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.yaml"
            save_yaml(data, file_path)
            loaded = load_yaml(file_path)

            assert loaded == data

    def test_load_nonexistent_file(self) -> None:
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_yaml("nonexistent.yaml")

    def test_nested_structure(self) -> None:
        """Test saving and loading nested structure."""
        data = {"level1": {"level2": {"level3": "value"}}}

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "nested.yaml"
            save_yaml(data, file_path)
            loaded = load_yaml(file_path)

            assert loaded == data

    def test_empty_file(self) -> None:
        """Test loading empty YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "empty.yaml"
            file_path.write_text("")
            loaded = load_yaml(file_path)

            assert loaded == {}


class TestMergeConfigs:
    """Tests for config merging."""

    def test_merge_simple(self) -> None:
        """Test merging simple configs."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_configs(base, override)

        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested(self) -> None:
        """Test merging nested configs."""
        base = {"level1": {"a": 1, "b": 2}}
        override = {"level1": {"b": 3, "c": 4}}
        result = merge_configs(base, override)

        assert result == {"level1": {"a": 1, "b": 3, "c": 4}}

    def test_merge_deep_nested(self) -> None:
        """Test merging deeply nested configs."""
        base = {"l1": {"l2": {"a": 1, "b": 2}}}
        override = {"l1": {"l2": {"b": 3}, "l3": 4}}
        result = merge_configs(base, override)

        assert result == {"l1": {"l2": {"a": 1, "b": 3}, "l3": 4}}

    def test_merge_preserves_base(self) -> None:
        """Test that merging doesn't modify base config."""
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        merge_configs(base, override)

        assert base == {"a": 1, "b": 2}  # Unchanged


class TestGetNestedValue:
    """Tests for getting nested values."""

    def test_simple_key(self) -> None:
        """Test getting simple key."""
        config = {"key": "value"}
        result = get_nested_value(config, "key")
        assert result == "value"

    def test_nested_key(self) -> None:
        """Test getting nested key."""
        config = {"level1": {"level2": {"level3": "value"}}}
        result = get_nested_value(config, "level1.level2.level3")
        assert result == "value"

    def test_missing_key_with_default(self) -> None:
        """Test getting missing key returns default."""
        config = {"key": "value"}
        result = get_nested_value(config, "missing", default="default_value")
        assert result == "default_value"

    def test_missing_key_without_default(self) -> None:
        """Test getting missing key returns None."""
        config = {"key": "value"}
        result = get_nested_value(config, "missing")
        assert result is None

    def test_partial_path_exists(self) -> None:
        """Test when partial path exists but not full path."""
        config = {"level1": {"level2": "value"}}
        result = get_nested_value(config, "level1.level2.level3", default="default")
        assert result == "default"

    def test_custom_separator(self) -> None:
        """Test using custom separator."""
        config = {"level1": {"level2": "value"}}
        result = get_nested_value(config, "level1/level2", separator="/")
        assert result == "value"


class TestSetNestedValue:
    """Tests for setting nested values."""

    def test_simple_key(self) -> None:
        """Test setting simple key."""
        config: dict[str, str] = {}
        set_nested_value(config, "key", "value")
        assert config == {"key": "value"}

    def test_nested_key(self) -> None:
        """Test setting nested key."""
        config: dict[str, dict[str, dict[str, str]]] = {}
        set_nested_value(config, "level1.level2.level3", "value")
        assert config == {"level1": {"level2": {"level3": "value"}}}

    def test_overwrite_existing(self) -> None:
        """Test overwriting existing value."""
        config = {"key": "old_value"}
        set_nested_value(config, "key", "new_value")
        assert config == {"key": "new_value"}

    def test_add_to_existing_nested(self) -> None:
        """Test adding to existing nested structure."""
        config = {"level1": {"existing": "value"}}
        set_nested_value(config, "level1.new_key", "new_value")
        assert config == {"level1": {"existing": "value", "new_key": "new_value"}}

    def test_custom_separator(self) -> None:
        """Test using custom separator."""
        config: dict[str, dict[str, str]] = {}
        set_nested_value(config, "level1/level2", "value", separator="/")
        assert config == {"level1": {"level2": "value"}}


class TestIntegration:
    """Integration tests for config utilities."""

    def test_full_workflow(self) -> None:
        """Test complete workflow of config operations."""
        # Create base config
        base_config = {
            "simulator": {"vehicle": {"wheelbase": 2.5, "max_speed": 10.0}},
            "controller": {"type": "pure_pursuit"},
        }

        # Create override config
        override_config = {
            "simulator": {"vehicle": {"max_speed": 15.0}},
            "controller": {"lookahead": 5.0},
        }

        # Merge configs
        merged = merge_configs(base_config, override_config)

        # Get nested values
        wheelbase = get_nested_value(merged, "simulator.vehicle.wheelbase")
        max_speed = get_nested_value(merged, "simulator.vehicle.max_speed")
        lookahead = get_nested_value(merged, "controller.lookahead")

        assert wheelbase == 2.5
        assert max_speed == 15.0
        assert lookahead == 5.0

        # Set new nested value
        set_nested_value(merged, "controller.kp", 1.0)
        kp = get_nested_value(merged, "controller.kp")
        assert kp == 1.0

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "config.yaml"
            save_yaml(merged, file_path)
            loaded = load_yaml(file_path)

            assert loaded == merged
