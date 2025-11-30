"""Generate interactive HTML dashboard from SimulationLog using React template."""

import argparse
import json
from pathlib import Path
from typing import Any

from core.data import SimulationLog


def generate_dashboard(log: SimulationLog, output_path: str | Path) -> None:
    """Generate interactive HTML dashboard.

    Args:
        log: Simulation log
        output_path: Output HTML file path
    """
    # 1. Prepare data
    data: dict[str, Any] = {
        "metadata": {
            "controller": log.metadata.get("controller", "Unknown Controller"),
            "execution_time": log.metadata.get("execution_time", "Unknown Time"),
            **log.metadata,
        },
        "steps": [],
    }

    for step in log.steps:
        data["steps"].append(
            {
                "timestamp": step.timestamp,
                "x": step.vehicle_state.x,
                "y": step.vehicle_state.y,
                "z": getattr(step.vehicle_state, "z", 0.0),
                "yaw": step.vehicle_state.yaw,
                "velocity": step.vehicle_state.velocity,
                "acceleration": step.action.acceleration,
                "steering": step.action.steering,
            }
        )

    # 2. Load template
    # Assuming the script is in tools/scripts/generate_dashboard.py
    # and the template is in tools/dashboard/dist/index.html
    script_dir = Path(__file__).parent
    template_path = script_dir.parent / "dashboard" / "dist" / "index.html"

    if not template_path.exists():
        print(f"Error: Dashboard template not found at {template_path}")
        print("Please build the dashboard first: cd tools/dashboard && npm run build")
        return

    with open(template_path, encoding="utf-8") as f:
        template_content = f.read()

    # 3. Inject data
    json_data = json.dumps(data)
    injection_script = f"<script>window.SIMULATION_DATA = {json_data};</script>"

    # Inject before </head> or </body>
    if "</head>" in template_content:
        html_content = template_content.replace("</head>", f"{injection_script}</head>")
    else:
        html_content = template_content.replace("</body>", f"{injection_script}</body>")

    # 4. Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Dashboard saved to {output_path}")


def main() -> None:
    """Generate dashboard from command line."""
    parser = argparse.ArgumentParser(description="Generate interactive HTML dashboard")
    parser.add_argument("log_file", help="Path to SimulationLog JSON file")
    parser.add_argument("-o", "--output", required=True, help="Output HTML file path")

    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return

    print(f"Loading log from {log_path}...")
    log = SimulationLog.load(log_path)
    print(f"Loaded {len(log.steps)} steps")

    generate_dashboard(log, args.output)
    print(f"\nOpen {args.output} in a browser to view the dashboard")


if __name__ == "__main__":
    main()
