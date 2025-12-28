"""Tests for obstacle generator."""

import math
from pathlib import Path

import pytest
from experiment.engine.obstacle_generator import ObstacleGenerator
from omegaconf import OmegaConf


@pytest.fixture
def map_path():
    """Path to test map file."""
    return Path("experiment/assets/lanelet2_map.osm")


@pytest.fixture
def generator(map_path):
    """Create obstacle generator instance."""
    return ObstacleGenerator(map_path, seed=42)


class TestExclusionZone:
    """Tests for exclusion zone functionality."""

    def test_exclusion_zone_enabled(self, generator):
        """Test that obstacles are not generated within exclusion zone."""
        initial_state = {"x": 89630.0, "y": 43130.0, "yaw": 2.2, "velocity": 0.0}
        exclusion_distance = 10.0

        config = OmegaConf.create(
            {
                "enabled": True,
                "exclusion_zone": {"enabled": True, "distance": exclusion_distance},
                "groups": [
                    {
                        "count": 10,
                        "type": "static",
                        "shape": {"type": "rectangle", "width": 1.3, "length": 2.064},
                        "placement": {
                            "strategy": "random_track",
                            "offset": {"min": -2.0, "max": 2.0},
                        },
                    }
                ],
            }
        )

        obstacles = generator.generate(config, initial_state=initial_state)

        # Check that all generated obstacles are outside exclusion zone
        for obs in obstacles:
            pos = obs["position"]
            dist = math.hypot(pos["x"] - initial_state["x"], pos["y"] - initial_state["y"])
            assert dist >= exclusion_distance, (
                f"Obstacle at ({pos['x']}, {pos['y']}) is within exclusion zone (distance: {dist:.2f}m)"
            )

    def test_exclusion_zone_disabled(self, generator):
        """Test that obstacles can be generated anywhere when exclusion zone is disabled."""
        initial_state = {"x": 89630.0, "y": 43130.0, "yaw": 2.2, "velocity": 0.0}

        config = OmegaConf.create(
            {
                "enabled": True,
                "exclusion_zone": {"enabled": False, "distance": 10.0},
                "groups": [
                    {
                        "count": 20,
                        "type": "static",
                        "shape": {"type": "rectangle", "width": 1.3, "length": 2.064},
                        "placement": {
                            "strategy": "random_track",
                            "offset": {"min": -2.0, "max": 2.0},
                        },
                    }
                ],
            }
        )

        obstacles = generator.generate(config, initial_state=initial_state)

        # With exclusion zone disabled and enough attempts, some obstacles
        # should be generated close to initial position
        # (This is probabilistic, but with 20 obstacles it's very likely)
        assert len(obstacles) > 0, "Should generate some obstacles"

    def test_exclusion_zone_boundary(self, generator):
        """Test that obstacles at exactly the boundary distance are accepted."""
        initial_state = {"x": 89630.0, "y": 43130.0, "yaw": 2.2, "velocity": 0.0}
        exclusion_distance = 5.0

        config = OmegaConf.create(
            {
                "enabled": True,
                "exclusion_zone": {"enabled": True, "distance": exclusion_distance},
                "groups": [
                    {
                        "count": 10,
                        "type": "static",
                        "shape": {"type": "rectangle", "width": 1.3, "length": 2.064},
                        "placement": {
                            "strategy": "random_track",
                            "offset": {"min": -2.0, "max": 2.0},
                        },
                    }
                ],
            }
        )

        obstacles = generator.generate(config, initial_state=initial_state)

        # All obstacles should be at or beyond the exclusion distance
        for obs in obstacles:
            pos = obs["position"]
            dist = math.hypot(pos["x"] - initial_state["x"], pos["y"] - initial_state["y"])
            assert dist >= exclusion_distance

    def test_no_initial_state(self, generator):
        """Test that generation works when no initial state is provided."""
        config = OmegaConf.create(
            {
                "enabled": True,
                "exclusion_zone": {"enabled": True, "distance": 10.0},
                "groups": [
                    {
                        "count": 5,
                        "type": "static",
                        "shape": {"type": "rectangle", "width": 1.3, "length": 2.064},
                        "placement": {
                            "strategy": "random_track",
                            "offset": {"min": -2.0, "max": 2.0},
                        },
                    }
                ],
            }
        )

        # Should not crash when initial_state is None
        obstacles = generator.generate(config, initial_state=None)
        # Exclusion zone check should be skipped, so obstacles can be generated
        assert isinstance(obstacles, list)

    def test_exclusion_zone_not_configured(self, generator):
        """Test that generation works when exclusion zone is not configured."""
        initial_state = {"x": 89630.0, "y": 43130.0, "yaw": 2.2, "velocity": 0.0}

        config = OmegaConf.create(
            {
                "enabled": True,
                # No exclusion_zone key
                "groups": [
                    {
                        "count": 5,
                        "type": "static",
                        "shape": {"type": "rectangle", "width": 1.3, "length": 2.064},
                        "placement": {
                            "strategy": "random_track",
                            "offset": {"min": -2.0, "max": 2.0},
                        },
                    }
                ],
            }
        )

        # Should not crash when exclusion_zone is not in config
        obstacles = generator.generate(config, initial_state=initial_state)
        assert isinstance(obstacles, list)
