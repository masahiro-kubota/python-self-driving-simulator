"""Generate interactive HTML dashboard from SimulationLog."""

import argparse
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.data import SimulationLog


def generate_dashboard(log: SimulationLog, output_path: str | Path) -> None:
    """Generate interactive HTML dashboard.

    Args:
        log: Simulation log
        output_path: Output HTML file path
    """
    # Extract data
    timestamps = [s.timestamp for s in log.steps]
    x_coords = [s.vehicle_state.x for s in log.steps]
    y_coords = [s.vehicle_state.y for s in log.steps]
    velocities = [s.vehicle_state.velocity for s in log.steps]
    yaws = [s.vehicle_state.yaw for s in log.steps]
    steerings = [s.action.steering for s in log.steps]
    accelerations = [s.action.acceleration for s in log.steps]

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Trajectory (X-Y)",
            "Velocity vs Time",
            "Yaw vs Time",
            "Steering vs Time",
            "Acceleration vs Time",
            "Speed Profile",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
    )

    # 1. Trajectory
    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="lines+markers",
            name="Trajectory",
            marker=dict(size=3, color=timestamps, colorscale="Viridis", showscale=True),
            line=dict(width=2),
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Time: %{marker.color:.2f}s<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # 2. Velocity
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=velocities,
            mode="lines",
            name="Velocity",
            line=dict(color="blue", width=2),
            hovertemplate="Time: %{x:.2f}s<br>Velocity: %{y:.2f} m/s<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # 3. Yaw
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=yaws,
            mode="lines",
            name="Yaw",
            line=dict(color="green", width=2),
            hovertemplate="Time: %{x:.2f}s<br>Yaw: %{y:.2f} rad<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # 4. Steering
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=steerings,
            mode="lines",
            name="Steering",
            line=dict(color="red", width=2),
            hovertemplate="Time: %{x:.2f}s<br>Steering: %{y:.2f} rad<extra></extra>",
        ),
        row=2,
        col=2,
    )

    # 5. Acceleration
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=accelerations,
            mode="lines",
            name="Acceleration",
            line=dict(color="purple", width=2),
            hovertemplate="Time: %{x:.2f}s<br>Accel: %{y:.2f} m/s²<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # 6. Speed profile (color-coded by speed)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(velocities))),
            y=velocities,
            mode="markers",
            name="Speed Profile",
            marker=dict(
                size=5,
                color=velocities,
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Speed (m/s)", x=1.15),
            ),
            hovertemplate="Step: %{x}<br>Speed: %{y:.2f} m/s<extra></extra>",
        ),
        row=3,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Simulation Dashboard<br><sub>{log.metadata.get('controller', 'Unknown Controller')}</sub>",
            x=0.5,
            xanchor="center",
        ),
        height=1200,
        showlegend=False,
        hovermode="closest",
        template="plotly_white",
    )

    # Update axes labels
    fig.update_xaxes(title_text="X [m]", row=1, col=1)
    fig.update_yaxes(title_text="Y [m]", row=1, col=1)
    fig.update_xaxes(title_text="Time [s]", row=1, col=2)
    fig.update_yaxes(title_text="Velocity [m/s]", row=1, col=2)
    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig.update_yaxes(title_text="Yaw [rad]", row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=2)
    fig.update_yaxes(title_text="Steering [rad]", row=2, col=2)
    fig.update_xaxes(title_text="Time [s]", row=3, col=1)
    fig.update_yaxes(title_text="Acceleration [m/s²]", row=3, col=1)
    fig.update_xaxes(title_text="Step", row=3, col=2)
    fig.update_yaxes(title_text="Speed [m/s]", row=3, col=2)

    # Make trajectory plot square
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)

    # Save to HTML
    fig.write_html(str(output_path), include_plotlyjs="cdn")
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
