#!/usr/bin/env python3
"""Test script to generate a dashboard with vehicle rectangle visualization."""

import sys
from pathlib import Path

# Add dashboard/src and dashboard/tests to path
dashboard_root = Path(__file__).parent.parent
sys.path.insert(0, str(dashboard_root / "src"))
sys.path.insert(0, str(dashboard_root / "tests"))

from dashboard.generator import HTMLDashboardGenerator  # noqa: E402
from dummy_data import (  # noqa: E402
    create_experiment_result_from_log,
    generate_circular_trajectory,
)


def main():
    """Generate a test dashboard."""
    # Generate test data with vehicle parameters
    log = generate_circular_trajectory(num_steps=200, radius=50.0, duration=20.0)

    # Create experiment result
    result = create_experiment_result_from_log(
        log,
        experiment_name="Vehicle Rectangle Test",
        experiment_type="visualization_test",
    )

    # Generate dashboard
    generator = HTMLDashboardGenerator()
    output_path = Path("/tmp/test_vehicle_rectangle_dashboard.html")

    print("Generating dashboard with vehicle rectangle visualization...")
    print("Vehicle parameters:")
    print(f"  - Width: {log.metadata['width']} m")
    print(f"  - Length: {log.metadata['length']} m")
    print(f"  - Wheelbase: {log.metadata['wheelbase']} m")
    print(f"  - Front overhang: {log.metadata['front_overhang']} m")
    print(f"  - Rear overhang: {log.metadata['rear_overhang']} m")

    generated_path = generator.generate(result, output_path)
    print("\nDashboard generated successfully!")
    print(f"Open in browser: file://{generated_path}")


if __name__ == "__main__":
    main()
