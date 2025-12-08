"""Core interfaces for the simulation framework."""

from core.data.node_io import NodeIO
from core.data.simulation_context import SimulationContext
from core.interfaces.ad_components import ADComponent
from core.interfaces.dashboard import DashboardGenerator
from core.interfaces.experiment import ExperimentLogger
from core.interfaces.node import Node
from core.interfaces.processor import ProcessorProtocol
from core.interfaces.simulator import SimulationLogRepository, Simulator

__all__ = [
    "ADComponent",
    "DashboardGenerator",
    "ExperimentLogger",
    "Node",
    "NodeIO",
    "ProcessorProtocol",
    "SimulationContext",
    "SimulationLogRepository",
    "Simulator",
]
