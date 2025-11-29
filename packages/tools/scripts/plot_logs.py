"""Plot script for simulation logs."""

import argparse
from pathlib import Path

from core.data import SimulationLog
from tools.visualization.plotter import SimulationPlotter


def main() -> None:
    """Generate static plot from simulation logs."""
    parser = argparse.ArgumentParser(description="Generate plot from simulation logs")
    parser.add_argument("logs", nargs="+", help="Paths to log files (JSON)")
    parser.add_argument("-o", "--output", required=True, help="Output file path (.png, .pdf, etc.)")
    parser.add_argument("--no-track", action="store_true", help="Don't plot reference track")
    
    args = parser.parse_args()
    
    # Load logs
    loaded_logs: list[tuple[SimulationLog, str]] = []
    
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
    
    if not loaded_logs:
        print("No valid logs found.")
        return
    
    # Create plotter (no track for now)
    plotter = SimulationPlotter(track_trajectory=None)
    
    for log, label in loaded_logs:
        plotter.add_log(log, label)
    
    # Save plot
    plotter.save(args.output)
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
