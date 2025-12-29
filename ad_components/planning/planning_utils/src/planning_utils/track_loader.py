"""Track loader implementation."""

import csv
from pathlib import Path

from scipy.spatial.transform import Rotation

from planning_utils.types import ReferencePath, ReferencePathPoint


def load_track_csv(file_path: str | Path) -> ReferencePath:
    """Load track from CSV file.

    Args:
        file_path: Path to CSV file

    Returns:
        ReferencePath object
    """
    points: list[ReferencePathPoint] = []

    with open(file_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row["x"])
            y = float(row["y"])
            # z = float(row["z"])

            # Quaternion to Yaw
            qx = float(row["x_quat"])
            qy = float(row["y_quat"])
            qz = float(row["z_quat"])
            qw = float(row["w_quat"])

            rot = Rotation.from_quat([qx, qy, qz, qw])
            yaw = rot.as_euler("xyz")[2]  # z-axis rotation

            velocity = float(row["speed"])

            points.append(ReferencePathPoint(x=x, y=y, yaw=yaw, velocity=velocity))

    return ReferencePath(points=points)
