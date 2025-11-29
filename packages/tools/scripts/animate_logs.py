"""Animation script for simulation logs."""

import argparse
from pathlib import Path

from core.data import SimulationLog, Trajectory, TrajectoryPoint
from tools.visualization.animator import SimulationAnimator


def main() -> None:
    """Generate animation from simulation logs."""
    parser = argparse.ArgumentParser(description="Generate animation from simulation logs")
    parser.add_argument("logs", nargs="+", help="Paths to log files (JSON)")
    parser.add_argument("-o", "--output", required=True, help="Output file path (.gif or .mp4)")
    parser.add_argument("-i", "--interval", type=int, default=50, help="Frame interval [ms]")
    parser.add_argument("-s", "--skip", type=int, default=5, help="Step skip for faster animation")
    parser.add_argument("--no-track", action="store_true", help="Don't plot reference track")
    
    args = parser.parse_args()
    
    # Load logs
    loaded_logs: list[tuple[SimulationLog, str]] = []
    track: Trajectory | None = None
    
    for log_path_str in args.logs:
        log_path = Path(log_path_str)
        if not log_path.exists():
            print(f"Warning: Log file not found: {log_path}")
            continue
            
        print(f"Loading log from {log_path}...")
        log = SimulationLog.load(log_path)
        
        # Use log filename as label
        label = log_path.stem
        loaded_logs.append((log, label))
        
        # Try to extract track from first log's metadata
        if track is None and not args.no_track:
            # If track info is in metadata, we could load it
            # For now, we'll reconstruct track from vehicle states if needed
            # Or just skip track plotting
            pass
    
    if not loaded_logs:
        print("No valid logs found.")
        return
    
    # Create animator
    animator = SimulationAnimator(track_trajectory=track)
    
    for log, label in loaded_logs:
        # Downsample steps
        log.steps = log.steps[::args.skip]
        animator.add_log(log, label)
    
    # Save animation
    animator.save_animation(args.output, interval=args.interval)
    print(f"Animation saved to {args.output}")


if __name__ == "__main__":
    main()
