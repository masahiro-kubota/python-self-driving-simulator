"""AD component stack for simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.interfaces.ad_components import Controller, Planner


@dataclass
class ADComponentStack:
    """AD component stack for simulation.

    This encapsulates the planner and controller components,
    hiding the internal structure from the simulator interface.

    Attributes:
        planner: Planning component (optional for E2E models)
        controller: Control component (optional for E2E models)
        end_to_end: End-to-end model (optional, for future use)
    """

    planner: Planner | None = None
    controller: Controller | None = None
    end_to_end: Any | None = None  # For E2E models
