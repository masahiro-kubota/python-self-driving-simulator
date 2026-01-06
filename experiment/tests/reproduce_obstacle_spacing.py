
import pytest
from unittest.mock import MagicMock
from experiment.engine.obstacle_generator import ObstacleGenerator

class MockPoseSampler:
    def __init__(self):
        self.drivable_area = MagicMock()
        self.drivable_area.contains.return_value = True
        self.drivable_area.intersects.return_value = True
        self.validate_pose = MagicMock(return_value=True)
        self.global_centerline = []
        self.total_track_length = 1000.0

@pytest.fixture
def generator():
    # Bypass __init__
    gen = ObstacleGenerator.__new__(ObstacleGenerator)
    gen.pose_sampler = MockPoseSampler()
    gen.drivable_area = gen.pose_sampler.drivable_area
    gen.total_track_length = 1000.0
    gen.exclusion_zone = None
    gen.initial_state = None
    gen.centerlines = [] # Legacy
    return gen

def test_validate_min_distance_rejection(generator):
    """Test that validation fails if obstacles are too close longitudinally."""
    
    # Existing obstacle at dist=100
    existing = [{
        "position": {
            "x": 100.0, "y": 0.0, "yaw": 0.0,
            "centerline_dist": 100.0,
            # centerline_index is missing (None)
        },
        "shape": {"width": 2.0, "length": 4.0}
    }]
    
    # Candidate at dist=105 (Diff=5 < 10)
    candidate = {
        "position": {
            "x": 105.0, "y": 0.0, "yaw": 0.0,
            "centerline_dist": 105.0
        },
        "shape": {"width": 2.0, "length": 4.0}
    }
    
    # Should return False
    valid = generator._validate_placement(
        candidate, existing, min_distance=10.0, require_within_bounds=False
    )
    assert not valid, "Should reject placement within min_distance"

def test_validate_min_distance_acceptance(generator):
    """Test that validation passes if obstacles are far enough."""
    
    existing = [{
        "position": {
            "x": 100.0, "y": 0.0, "yaw": 0.0,
            "centerline_dist": 100.0
        },
        "shape": {"width": 2.0, "length": 4.0}
    }]
    
    # Candidate at dist=111 (Diff=11 > 10)
    candidate = {
        "position": {
            "x": 111.0, "y": 0.0, "yaw": 0.0,
            "centerline_dist": 111.0
        },
        "shape": {"width": 2.0, "length": 4.0}
    }
    
    valid = generator._validate_placement(
        candidate, existing, min_distance=10.0, require_within_bounds=False
    )
    assert valid, "Should accept placement outside min_distance"

def test_validate_same_centerline_index_none(generator):
    """Verify behavior when centerline_index is None for both."""
    existing = [{
        "position": {
            "centerline_dist": 100.0, "centerline_index": None,
            "x": 0, "y":0, "yaw":0 # Dummy
        },
        "shape": {"width": 2, "length": 4}
    }]
    candidate = {
        "position": {
            "centerline_dist": 100.0, "centerline_index": None,
            "x": 0, "y":0, "yaw":0
        },
        "shape": {"width": 2, "length": 4}
    }
    
    # None == None is True, so it should compare distances
    valid = generator._validate_placement(
        candidate, existing, min_distance=10.0
    )
    assert not valid

