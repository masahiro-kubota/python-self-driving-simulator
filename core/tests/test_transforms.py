"""Tests for coordinate transformation utilities."""

import math

import numpy as np
from core.utils.transforms import (
    global_to_local,
    local_to_global,
    rotation_matrix_2d,
    transform_angle_to_global,
    transform_angle_to_local,
    transformation_matrix_2d,
)


class TestGlobalToLocal:
    """Tests for global_to_local function."""

    def test_no_translation_no_rotation(self) -> None:
        """Test with no translation and no rotation."""
        x, y = global_to_local(1.0, 2.0, 0.0, 0.0, 0.0)
        assert abs(x - 1.0) < 1e-10
        assert abs(y - 2.0) < 1e-10

    def test_translation_only(self) -> None:
        """Test with translation only."""
        x, y = global_to_local(3.0, 4.0, 1.0, 2.0, 0.0)
        assert abs(x - 2.0) < 1e-10
        assert abs(y - 2.0) < 1e-10

    def test_rotation_only(self) -> None:
        """Test with rotation only (90 degrees)."""
        x, y = global_to_local(0.0, 1.0, 0.0, 0.0, math.pi / 2)
        assert abs(x - 1.0) < 1e-10
        assert abs(y - 0.0) < 1e-10

    def test_translation_and_rotation(self) -> None:
        """Test with both translation and rotation."""
        x, y = global_to_local(2.0, 1.0, 1.0, 1.0, math.pi / 2)
        assert abs(x - 0.0) < 1e-10
        assert abs(y - (-1.0)) < 1e-10


class TestLocalToGlobal:
    """Tests for local_to_global function."""

    def test_no_translation_no_rotation(self) -> None:
        """Test with no translation and no rotation."""
        x, y = local_to_global(1.0, 2.0, 0.0, 0.0, 0.0)
        assert abs(x - 1.0) < 1e-10
        assert abs(y - 2.0) < 1e-10

    def test_translation_only(self) -> None:
        """Test with translation only."""
        x, y = local_to_global(2.0, 2.0, 1.0, 2.0, 0.0)
        assert abs(x - 3.0) < 1e-10
        assert abs(y - 4.0) < 1e-10

    def test_rotation_only(self) -> None:
        """Test with rotation only (90 degrees)."""
        x, y = local_to_global(1.0, 0.0, 0.0, 0.0, math.pi / 2)
        assert abs(x - 0.0) < 1e-10
        assert abs(y - 1.0) < 1e-10

    def test_roundtrip(self) -> None:
        """Test roundtrip conversion."""
        origin_x, origin_y, origin_yaw = 5.0, 3.0, 0.7
        global_x, global_y = 10.0, 8.0

        # Global -> Local -> Global
        local_x, local_y = global_to_local(global_x, global_y, origin_x, origin_y, origin_yaw)
        restored_x, restored_y = local_to_global(local_x, local_y, origin_x, origin_y, origin_yaw)

        assert abs(restored_x - global_x) < 1e-10
        assert abs(restored_y - global_y) < 1e-10


class TestAngleTransforms:
    """Tests for angle transformation functions."""

    def test_angle_to_local_no_rotation(self) -> None:
        """Test angle transformation with no rotation."""
        angle = transform_angle_to_local(1.0, 0.0)
        assert abs(angle - 1.0) < 1e-10

    def test_angle_to_local_with_rotation(self) -> None:
        """Test angle transformation with rotation."""
        angle = transform_angle_to_local(math.pi / 2, math.pi / 4)
        assert abs(angle - math.pi / 4) < 1e-10

    def test_angle_to_global_no_rotation(self) -> None:
        """Test angle transformation to global with no rotation."""
        angle = transform_angle_to_global(1.0, 0.0)
        assert abs(angle - 1.0) < 1e-10

    def test_angle_to_global_with_rotation(self) -> None:
        """Test angle transformation to global with rotation."""
        angle = transform_angle_to_global(math.pi / 4, math.pi / 4)
        assert abs(angle - math.pi / 2) < 1e-10

    def test_angle_roundtrip(self) -> None:
        """Test angle transformation roundtrip."""
        origin_yaw = 0.7
        global_angle = 1.5

        local_angle = transform_angle_to_local(global_angle, origin_yaw)
        restored_angle = transform_angle_to_global(local_angle, origin_yaw)

        # Normalize both angles for comparison
        from core.utils.geometry import normalize_angle

        assert abs(normalize_angle(restored_angle - global_angle)) < 1e-10


class TestRotationMatrix2D:
    """Tests for rotation_matrix_2d function."""

    def test_zero_rotation(self) -> None:
        """Test rotation matrix for zero angle."""
        mat = rotation_matrix_2d(0.0)
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(mat, expected)

    def test_90_degree_rotation(self) -> None:
        """Test rotation matrix for 90 degrees."""
        mat = rotation_matrix_2d(math.pi / 2)
        expected = np.array([[0.0, -1.0], [1.0, 0.0]])
        np.testing.assert_array_almost_equal(mat, expected)

    def test_180_degree_rotation(self) -> None:
        """Test rotation matrix for 180 degrees."""
        mat = rotation_matrix_2d(math.pi)
        expected = np.array([[-1.0, 0.0], [0.0, -1.0]])
        np.testing.assert_array_almost_equal(mat, expected)

    def test_rotation_application(self) -> None:
        """Test applying rotation matrix to a vector."""
        mat = rotation_matrix_2d(math.pi / 2)
        vec = np.array([1.0, 0.0])
        result = mat @ vec
        expected = np.array([0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestTransformationMatrix2D:
    """Tests for transformation_matrix_2d function."""

    def test_identity_transformation(self) -> None:
        """Test identity transformation."""
        mat = transformation_matrix_2d(0.0, 0.0, 0.0)
        expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(mat, expected)

    def test_translation_only(self) -> None:
        """Test translation only transformation."""
        mat = transformation_matrix_2d(2.0, 3.0, 0.0)
        point = np.array([1.0, 1.0, 1.0])
        result = mat @ point
        expected = np.array([3.0, 4.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotation_only(self) -> None:
        """Test rotation only transformation."""
        mat = transformation_matrix_2d(0.0, 0.0, math.pi / 2)
        point = np.array([1.0, 0.0, 1.0])
        result = mat @ point
        expected = np.array([0.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_combined_transformation(self) -> None:
        """Test combined translation and rotation."""
        mat = transformation_matrix_2d(1.0, 2.0, math.pi / 2)
        point = np.array([1.0, 0.0, 1.0])
        result = mat @ point
        expected = np.array([1.0, 3.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)
