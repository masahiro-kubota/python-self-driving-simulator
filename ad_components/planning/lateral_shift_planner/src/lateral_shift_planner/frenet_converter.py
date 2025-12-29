import numpy as np
from planning_utils.types import ReferencePath
from scipy.spatial import KDTree


class FrenetConverter:
    """Global <-> Frenet coordinate converter."""

    def __init__(self, ref_path: ReferencePath):
        """Initialize with a reference path.

        Args:
            ref_path: Reference global path
        """
        self.ref_path = ref_path
        self._x, self._y, self._yaw, _ = ref_path.to_arrays()

        # Build KDTree for nearest neighbor search
        self._tree = KDTree(np.column_stack((self._x, self._y)))

        # Pre-calculate s coordinate (cumulative distance)
        self._s = np.zeros(len(ref_path))
        if len(ref_path) > 1:
            dx = np.diff(self._x)
            dy = np.diff(self._y)
            dist = np.sqrt(dx**2 + dy**2)
            self._s[1:] = np.cumsum(dist)

    def global_to_frenet(self, x: float, y: float) -> tuple[float, float]:
        """Convert global (x, y) to Frenet (s, l).

        Args:
            x: Global X coordinate
            y: Global Y coordinate

        Returns:
            Tuple of (s, l)
            s: Longitudinal distance along the path
            l: Lateral distance from the path (positive = left)
        """
        # Find nearest point
        _, idx = self._tree.query([x, y])

        if idx == 0:
            idx_next = 1
        elif idx == len(self.ref_path) - 1:
            idx_next = idx
            idx = idx - 1
        else:
            # Check which neighbor is closer to projection
            # Simple heuristic: check dot product or rely on KDTree giving closest point
            # We need to determine the segment.
            # Vector v_current = P - Path[idx]
            # Vector v_next = Path[idx+1] - Path[idx]
            # If dot(v_current, v_next) > 0, it's likely forward.
            # But KDTree might return idx+1 if it's closer.

            # Let's verify valid segment [idx, idx+1] or [idx-1, idx]
            # We want to project onto the segment.
            # For simplicity, we can look at idx-1, idx, idx+1 and find best projection.
            pass

            # Refined strategy:
            # Consider segment (idx-1, idx) and (idx, idx+1).
            # Project onto both lines, take the one with projection falling within segment (0 <= t <= 1)
            # breaking ties with distance.

            # Since we just want a robust conversion, let's look at nearest locally.
            candidates = []
            if idx > 0:
                candidates.append(idx - 1)
            candidates.append(idx)
            if idx < len(self.ref_path) - 1:
                candidates.append(idx + 1)

            # But basic nearest is usually enough for "s" approximation anchor.
            # Let's stick to the segment that contains the projection.

            # If P is "behind" idx, use (idx-1, idx).
            # If P is "ahead" of idx, use (idx, idx+1).

            # Vector P - Path[idx]
            dx = x - self._x[idx]
            dy = y - self._y[idx]

            # Tangent vector at idx
            tx = np.cos(self._yaw[idx])
            ty = np.sin(self._yaw[idx])

            dot = dx * tx + dy * ty

            if dot >= 0:
                if idx < len(self.ref_path) - 1:
                    idx_next = idx + 1
                else:
                    idx_next = idx
                    idx = idx - 1
            else:
                if idx > 0:
                    idx_next = idx
                    idx = idx - 1
                else:
                    idx_next = 1

        # Segment vector
        dx_seg = self._x[idx_next] - self._x[idx]
        dy_seg = self._y[idx_next] - self._y[idx]
        seg_len = np.sqrt(dx_seg**2 + dy_seg**2)

        # Vector to point
        dx_p = x - self._x[idx]
        dy_p = y - self._y[idx]

        # Project
        if seg_len < 1e-6:
            proj = 0.0
        else:
            proj = (dx_p * dx_seg + dy_p * dy_seg) / seg_len

        s = self._s[idx] + proj

        # Calculate l using cross product (z-component)
        # Vector (dx_seg, dy_seg, 0) x (dx_p, dy_p, 0)
        # cp_z = dx_seg * dy_p - dy_seg * dx_p
        # If we normalize seg vector to tangent t, l = (P-Ref) dot n
        # n = (-sin, cos)
        # tangent (unit) = (dx_seg/len, dy_seg/len)
        # normal (unit) = (-dy_seg/len, dx_seg/len)

        if seg_len < 1e-6:
            # Fallback to yaw
            pass

        # Refined l calculation: use the tangent of the matched point on path
        # But for linear segment approximation:
        # l = distance from line.
        # sign: cross product.
        cross = dx_seg * dy_p - dy_seg * dx_p
        lat = cross / seg_len if seg_len > 1e-6 else 0.0

        return s, lat

    def frenet_to_global(self, s: float, lat: float) -> tuple[float, float]:
        """Convert Frenet (s, l) to Global (x, y).

        Args:
            s: Longitudinal distance
            l: Lateral distance

        Returns:
            Tuple of (x, y)
        """
        # Find segment for s
        # np.searchsorted finds index where s should be inserted to maintain order
        idx = np.searchsorted(self._s, s) - 1

        # Clamp index
        if idx < 0:
            idx = 0
        elif idx >= len(self.ref_path) - 1:
            idx = len(self.ref_path) - 2

        # Interpolate along ref path
        s0 = self._s[idx]
        s1 = self._s[idx + 1]

        if s1 - s0 < 1e-6:
            ratio = 0.0
        else:
            ratio = (s - s0) / (s1 - s0)

        x_ref = self._x[idx] + ratio * (self._x[idx + 1] - self._x[idx])
        y_ref = self._y[idx] + ratio * (self._y[idx + 1] - self._y[idx])

        # Interpolate yaw properly
        yaw0 = self._yaw[idx]
        yaw1 = self._yaw[idx + 1]

        # Handle wraparound
        diff = yaw1 - yaw0
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi

        yaw_ref = yaw0 + ratio * diff

        # Calculate deviation
        # n = (-sin, cos)
        nx = -np.sin(yaw_ref)
        ny = np.cos(yaw_ref)

        x = x_ref + lat * nx
        y = y_ref + lat * ny

        return x, y

    def get_yaw_at_s(self, s: float) -> float:
        """Get interpolated yaw angle at specified s position.

        Args:
            s: Longitudinal distance along the path

        Returns:
            Interpolated yaw angle in radians
        """
        # Find segment for s
        idx = np.searchsorted(self._s, s) - 1

        # Clamp index
        if idx < 0:
            idx = 0
        elif idx >= len(self.ref_path) - 1:
            idx = len(self.ref_path) - 2

        # Interpolate yaw
        s0 = self._s[idx]
        s1 = self._s[idx + 1]
        if s1 - s0 < 1e-6:
            ratio = 0.0
        else:
            ratio = (s - s0) / (s1 - s0)

        yaw0 = self._yaw[idx]
        yaw1 = self._yaw[idx + 1]

        # Handle wraparound
        diff = yaw1 - yaw0
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi

        return yaw0 + ratio * diff
