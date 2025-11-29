"""MCAP logger for simulation data."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcap.writer import Writer

from core.data import SimulationStep


class MCAPLogger:
    """MCAP format logger for simulation data."""

    def __init__(self, output_path: str | Path) -> None:
        """Initialize MCAP logger.

        Args:
            output_path: Output file path (.mcap)
        """
        self.output_path = Path(output_path)
        self.file: Any = None
        self.writer: Writer | None = None
        self.schema_id: int | None = None
        self.channel_id: int | None = None

    def __enter__(self) -> "MCAPLogger":
        """Open MCAP file for writing."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.output_path, "wb")
        self.writer = Writer(self.file)

        # Register schema
        self.schema_id = self.writer.register_schema(
            name="SimulationStep",
            encoding="json",
            data=json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "number"},
                        "vehicle_state": {"type": "object"},
                        "action": {"type": "object"},
                    },
                }
            ).encode(),
        )

        # Register channel
        self.channel_id = self.writer.register_channel(
            topic="/simulation/step",
            message_encoding="json",
            schema_id=self.schema_id,
        )

        return self

    def log_step(self, step: SimulationStep) -> None:
        """Log a simulation step.

        Args:
            step: Simulation step to log
        """
        if self.writer is None or self.channel_id is None:
            msg = "Logger not initialized. Use 'with MCAPLogger(...) as logger:'"
            raise RuntimeError(msg)

        data = {
            "timestamp": step.timestamp,
            "vehicle_state": asdict(step.vehicle_state),
            "action": asdict(step.action),
        }

        self.writer.add_message(
            channel_id=self.channel_id,
            log_time=int(step.timestamp * 1e9),  # nanoseconds
            data=json.dumps(data).encode(),
            publish_time=int(step.timestamp * 1e9),
        )

    def __exit__(self, *args: Any) -> None:
        """Close MCAP file."""
        if self.writer:
            self.writer.finish()
        if self.file:
            self.file.close()
