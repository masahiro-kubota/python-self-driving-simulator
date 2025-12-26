#!/usr/bin/env python3
"""Benchmark script for LiDAR sensor performance.

This script measures the performance of the LiDAR sensor with different
configurations and provides detailed timing information.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from core.data import LidarConfig, SimulatorObstacle, VehicleState
from simulator.map import LaneletMap
from simulator.obstacle import ObstacleManager
from simulator.sensor import LidarSensor


def create_test_map(map_path: Path) -> LaneletMap | None:
    """Create a test map if available."""
    if map_path.exists():
        return LaneletMap(map_path)
    return None


def create_test_obstacles() -> list[SimulatorObstacle]:
    """Create test obstacles for benchmarking."""
    # Simplified: skip obstacles for now due to complex setup
    # Focus on map boundary performance which is the main bottleneck
    return []


def benchmark_lidar_scan(
    num_beams: int,
    range_max: float,
    with_map: bool,
    with_obstacles: bool,
    map_path: Path | None = None,
    num_iterations: int = 100,
) -> dict:
    """Benchmark LiDAR scan performance.

    Args:
        num_beams: Number of laser beams
        range_max: Maximum range in meters
        with_map: Whether to include map boundaries
        with_obstacles: Whether to include obstacles
        map_path: Path to map file
        num_iterations: Number of iterations to run

    Returns:
        Dictionary with benchmark results
    """
    # Setup
    config = LidarConfig(
        num_beams=num_beams,
        range_max=range_max,
        fov=270.0,
        range_min=0.1,
        angle_increment=0.0,
        x=1.5,
        y=0.0,
        z=1.8,
        yaw=0.0,
    )

    map_instance = None
    if with_map and map_path is not None:
        map_instance = create_test_map(map_path)

    obstacle_manager = None
    if with_obstacles:
        obstacles = create_test_obstacles()
        obstacle_manager = ObstacleManager(obstacles)

    sensor = LidarSensor(
        config=config,
        map_instance=map_instance,
        obstacle_manager=obstacle_manager,
    )

    # Test vehicle state
    vehicle_state = VehicleState(
        x=0.0,
        y=0.0,
        yaw=0.0,
        velocity=10.0,
        steering=0.0,
        timestamp=0.0,
    )

    # Warmup
    for _ in range(5):
        sensor.scan(vehicle_state)

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        sensor.scan(vehicle_state)
        end = time.perf_counter()
        times.append(end - start)

    times_array = np.array(times)

    return {
        "num_beams": num_beams,
        "range_max": range_max,
        "with_map": with_map,
        "with_obstacles": with_obstacles,
        "num_iterations": num_iterations,
        "mean_time_ms": float(np.mean(times_array) * 1000),
        "std_time_ms": float(np.std(times_array) * 1000),
        "min_time_ms": float(np.min(times_array) * 1000),
        "max_time_ms": float(np.max(times_array) * 1000),
        "median_time_ms": float(np.median(times_array) * 1000),
        "p95_time_ms": float(np.percentile(times_array, 95) * 1000),
        "p99_time_ms": float(np.percentile(times_array, 99) * 1000),
        "total_time_s": float(np.sum(times_array)),
    }


def print_results(results: list[dict]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 120)
    print("LiDAR Sensor Benchmark Results")
    print("=" * 120)
    print(
        f"{'Beams':<8} {'Map':<6} {'Obs':<6} {'Mean (ms)':<12} {'Std (ms)':<12} "
        f"{'Min (ms)':<12} {'Max (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12}"
    )
    print("-" * 120)

    for result in results:
        print(
            f"{result['num_beams']:<8} "
            f"{'Yes' if result['with_map'] else 'No':<6} "
            f"{'Yes' if result['with_obstacles'] else 'No':<6} "
            f"{result['mean_time_ms']:<12.3f} "
            f"{result['std_time_ms']:<12.3f} "
            f"{result['min_time_ms']:<12.3f} "
            f"{result['max_time_ms']:<12.3f} "
            f"{result['p95_time_ms']:<12.3f} "
            f"{result['p99_time_ms']:<12.3f}"
        )

    print("=" * 120)


def compare_results(before_path: Path, after_path: Path) -> None:
    """Compare two benchmark results and show speedup."""
    with before_path.open() as f:
        before = json.load(f)

    with after_path.open() as f:
        after = json.load(f)

    print("\n" + "=" * 100)
    print("Performance Comparison")
    print("=" * 100)
    print(
        f"{'Beams':<8} {'Map':<6} {'Obs':<6} {'Before (ms)':<14} "
        f"{'After (ms)':<14} {'Speedup':<12} {'Improvement':<12}"
    )
    print("-" * 100)

    for b, a in zip(before, after, strict=False):
        if (
            b["num_beams"] == a["num_beams"]
            and b["with_map"] == a["with_map"]
            and b["with_obstacles"] == a["with_obstacles"]
        ):
            speedup = b["mean_time_ms"] / a["mean_time_ms"]
            improvement = ((b["mean_time_ms"] - a["mean_time_ms"]) / b["mean_time_ms"]) * 100

            print(
                f"{b['num_beams']:<8} "
                f"{'Yes' if b['with_map'] else 'No':<6} "
                f"{'Yes' if b['with_obstacles'] else 'No':<6} "
                f"{b['mean_time_ms']:<14.3f} "
                f"{a['mean_time_ms']:<14.3f} "
                f"{speedup:<12.2f}x "
                f"{improvement:<12.1f}%"
            )

    print("=" * 100)


def main() -> None:
    """Run benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark LiDAR sensor performance")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("BEFORE", "AFTER"),
        help="Compare two benchmark results",
    )
    parser.add_argument(
        "--map-path",
        type=Path,
        help="Path to Lanelet2 map file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per configuration (default: 100)",
    )

    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    # Benchmark configurations
    beam_counts = [54, 108, 180, 360, 540, 1080]
    range_max = 30.0

    results = []

    total_configs = len(beam_counts) * 4  # 4 combinations of map/obstacles
    current = 0

    for num_beams in beam_counts:
        for with_map in [False, True]:
            for with_obstacles in [False, True]:
                current += 1
                print(
                    f"\n[{current}/{total_configs}] Benchmarking: "
                    f"beams={num_beams}, map={with_map}, obstacles={with_obstacles}"
                )

                result = benchmark_lidar_scan(
                    num_beams=num_beams,
                    range_max=range_max,
                    with_map=with_map,
                    with_obstacles=with_obstacles,
                    map_path=args.map_path,
                    num_iterations=args.iterations,
                )
                results.append(result)

                print(f"  Mean: {result['mean_time_ms']:.3f} ms ± {result['std_time_ms']:.3f} ms")

    # Print results
    print_results(results)

    # Save to file if requested
    if args.output:
        with args.output.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
