"""Core interfaces for the simulation framework."""

from core.data.frame_data import FrameData
from core.data.node_io import NodeIO
from core.interfaces.clock import Clock
from core.interfaces.dashboard import DashboardGenerator
from core.interfaces.experiment import ExperimentLogger
from core.interfaces.node import Node
from core.interfaces.simulator import SimulationLogRepository, Simulator

__all__ = [
    "Clock",
    "DashboardGenerator",
    "ExperimentLogger",
    "FrameData",
    "Node",
    "NodeIO",
    "SimulationLogRepository",
    "Simulator",
]
