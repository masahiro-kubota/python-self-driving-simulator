import csv
from pathlib import Path

from core.data import Trajectory, TrajectoryPoint


def load_track_csv_simple(path: Path) -> Trajectory:
    """Load track from CSV file.

    Expected format: x, y, yaw, velocity
    """
    points = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header if it exists (try to detect?)
        # Let's assume header exists if first token is string
        f.seek(0)
        first_line = f.readline().strip()
        has_header = False
        try:
            float(first_line.split(",")[0])
        except ValueError:
            has_header = True

        f.seek(0)
        reader = csv.reader(f)
        if has_header:
            next(reader, None)

        for row in reader:
            if len(row) < 2:
                continue
            try:
                x = float(row[0])
                y = float(row[1])
                yaw = float(row[2]) if len(row) > 2 else 0.0
                vel = float(row[3]) if len(row) > 3 else 0.0
                points.append(TrajectoryPoint(x=x, y=y, yaw=yaw, velocity=vel))
            except ValueError:
                continue

    return Trajectory(points=points)
