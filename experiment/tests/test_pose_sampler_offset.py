
import math
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from shapely.geometry import Polygon

from experiment.engine.pose_sampler import PoseSampler

@pytest.fixture
def pose_sampler():
    with patch.object(PoseSampler, "_load_map") as mock_load_map, \
         patch.object(PoseSampler, "_load_track") as mock_load_track, \
         patch("experiment.engine.pose_sampler.PoseSampler._resolve_path", side_effect=lambda x: x):
        
        # Initialize PoseSampler (mocks configured to do nothing)
        sampler = PoseSampler(Path("dummy_map"), Path("dummy_track"), seed=42)
        
        # Manually verify internal state (should be None/empty from init)
        assert sampler.drivable_area is None
        
        # Setup Drivable Area
        drivable_area = Polygon([(-100, -100), (100, -100), (100, 100), (-100, 100)])
        sampler.drivable_area = drivable_area
        
        # Setup Global Centerline (Straight line along X axis)
        # (x, y, yaw, dist)
        # 0 to 100
        centerline = []
        for i in range(101):
            centerline.append((float(i), 0.0, 0.0, float(i)))
        
        sampler.global_centerline = centerline
        sampler.total_track_length = 100.0
        
        return sampler

def test_lateral_offset_positive(pose_sampler):
    """Test that positive offset shifts to the Left (Y > 0 for Yaw=0)."""
    # Range [2.0, 3.0]
    # At Yaw=0 (East), Left is North (Y+). 
    # Logic: offset_y = base_y + cos(yaw)*offset = 0 + 1*offset = offset.
    # So Y should be positive.
    
    lateral_offset_range = [2.0, 3.0]
    
    for _ in range(20):
        # Result = (x, y, yaw, meta)
        result = pose_sampler.sample_track_pose(
            lateral_offset_range=lateral_offset_range,
            yaw_mode="aligned",
            yaw_offset_range=[0.0, 0.0]
        )
        assert result is not None
        x, y, yaw, meta = result
        
        offset = meta["lateral_offset"]
        assert 2.0 <= offset <= 3.0
        assert 2.0 <= y <= 3.0, f"Expected Y in [2.0, 3.0] for left offset, got {y}"

def test_lateral_offset_negative(pose_sampler):
    """Test that negative offset shifts to the Right (Y < 0 for Yaw=0)."""
    # Range [-3.0, -2.0]
    lateral_offset_range = [-3.0, -2.0]
    
    for _ in range(20):
        result = pose_sampler.sample_track_pose(
            lateral_offset_range=lateral_offset_range,
            yaw_mode="aligned",
            yaw_offset_range=[0.0, 0.0]
        )
        assert result is not None
        x, y, yaw, meta = result
        
        offset = meta["lateral_offset"]
        assert -3.0 <= offset <= -2.0
        assert -3.0 <= y <= -2.0, f"Expected Y in [-3.0, -2.0] for right offset, got {y}"

def test_lateral_offset_distribution_straddle(pose_sampler):
    """Test offset [-1.0, 1.0] produces values on both sides."""
    lateral_offset_range = [-1.0, 1.0]
    
    ys = []
    for _ in range(50):
        result = pose_sampler.sample_track_pose(
            lateral_offset_range=lateral_offset_range
        )
        ys.append(result[1])
    
    assert min(ys) < -0.2
    assert max(ys) > 0.2

