"""Convert MCAP file to SimulationLog JSON."""

import argparse
import json
from pathlib import Path

from core.data import Action, SimulationLog, SimulationStep, VehicleState
from mcap.reader import make_reader


def main() -> None:
    """Convert MCAP file to SimulationLog JSON format."""
    parser = argparse.ArgumentParser(description="Convert MCAP to SimulationLog JSON")
    parser.add_argument("mcap_file", help="Path to MCAP file")
    parser.add_argument("-o", "--output", help="Output JSON file (default: same name as MCAP)")

    args = parser.parse_args()

    mcap_path = Path(args.mcap_file)
    if not mcap_path.exists():
        print(f"Error: MCAP file not found: {mcap_path}")
        return

    # Read MCAP
    print(f"Reading MCAP file: {mcap_path}")
    log = SimulationLog(metadata={"source": "mcap", "original_file": str(mcap_path)})

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)

        for schema, channel, message in reader.iter_messages():
            # Decode JSON data
            data = json.loads(message.data.decode())

            # Create SimulationStep
            step = SimulationStep(
                timestamp=data["timestamp"],
                vehicle_state=VehicleState(**data["vehicle_state"]),
                action=Action(**data["action"]),
            )
            log.add_step(step)

    print(f"Loaded {len(log.steps)} steps from MCAP")

    # Save as JSON
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = mcap_path.with_suffix(".json")

    log.save(output_path)
    print(f"Saved SimulationLog to {output_path}")
    print(f"\nYou can now visualize this log with:")
    print(f"  uv run python packages/tools/scripts/plot_logs.py {output_path} -o plot.png")
    print(f"  uv run python packages/tools/scripts/animate_logs.py {output_path} -o animation.gif")


if __name__ == "__main__":
    main()
