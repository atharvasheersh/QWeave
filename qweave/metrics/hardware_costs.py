"""Declared duration/error costs and directed-coupling validation."""

from dataclasses import dataclass, field
import math

import networkx as nx
from qiskit import QuantumCircuit

from qweave.core.qiskit_adapter import source_operations
from qweave.scheduling import schedule_routed_circuit


DEFAULT_DURATIONS_NS = {
    "h": 35.0, "x": 35.0, "y": 35.0, "z": 1.0,
    "rx": 35.0, "ry": 35.0, "rz": 1.0, "p": 1.0,
    "cx": 300.0, "cz": 260.0, "cp": 300.0, "rzz": 300.0,
    "swap": 900.0, "reset": 1000.0, "measure": 1200.0,
}
SYMMETRIC_TWO_QUBIT = frozenset({"cz", "cp", "rzz"})


@dataclass(frozen=True)
class HardwareCostProfile:
    """Explicit synthetic or calibration-derived hardware assumptions."""

    name: str = "declared_superconducting_proxy_v1"
    calibrated: bool = False
    durations_ns: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DURATIONS_NS))
    one_qubit_error: float = 0.001
    two_qubit_error: float = 0.01
    measurement_error: float = 0.02
    reset_error: float = 0.01
    directed_edge_errors: dict[tuple[int, int], float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("hardware cost profile requires a name")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(value) or value < 0
               for value in self.durations_ns.values()):
            raise ValueError("gate durations must be finite and non-negative")
        probabilities = (self.one_qubit_error, self.two_qubit_error,
                         self.measurement_error, self.reset_error,
                         *self.directed_edge_errors.values())
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(value) or not 0 <= value < 1
               for value in probabilities):
            raise ValueError("hardware error probabilities must lie in [0, 1)")


def validate_coupling_legality(circuit: QuantumCircuit, coupling_graph: nx.Graph) -> bool:
    """Validate ordered gates on directed hardware and ordinary undirected gates."""

    if not isinstance(coupling_graph, nx.Graph) or coupling_graph.is_multigraph():
        raise TypeError("coupling graph must be a simple NetworkX graph")
    if any(type(node) is not int or node < 0 for node in coupling_graph.nodes):
        raise ValueError("physical qubit labels must be non-negative integers")
    for item in source_operations(circuit):
        if any(qubit not in coupling_graph for qubit in item.qubits):
            raise ValueError("operation uses a physical qubit absent from the coupling graph")
        if len(item.qubits) != 2:
            continue
        left, right = item.qubits
        if coupling_graph.is_directed():
            if item.operation.name == "swap":
                legal = (coupling_graph.has_edge(left, right)
                         and coupling_graph.has_edge(right, left))
            elif item.operation.name in SYMMETRIC_TWO_QUBIT:
                legal = coupling_graph.has_edge(left, right) or coupling_graph.has_edge(right, left)
            else:
                legal = coupling_graph.has_edge(left, right)
        else:
            legal = coupling_graph.has_edge(left, right)
        if not legal:
            raise ValueError(
                f"illegal {item.operation.name} direction on physical qubits {(left, right)}")
    return True


def estimate_hardware_cost(circuit: QuantumCircuit, coupling_graph: nx.Graph,
                           profile: HardwareCostProfile | None = None) -> dict:
    """Return stage-separated makespan and independent-error proxy metrics.

    The default profile is declared synthetic data, not device calibration.
    Directed graphs are checked before their undirected resource conflicts are
    passed to the scheduler.
    """

    profile = profile or HardwareCostProfile()
    validate_coupling_legality(circuit, coupling_graph)
    schedule_graph = (coupling_graph.to_undirected()
                      if coupling_graph.is_directed() else coupling_graph)
    schedule = schedule_routed_circuit(circuit, schedule_graph, profile.durations_ns)
    log_success = 0.0
    error_events = 0
    for item in source_operations(circuit):
        name = item.operation.name
        if name == "measure":
            errors = [profile.measurement_error]
        elif name == "reset":
            errors = [profile.reset_error]
        elif len(item.qubits) == 1:
            errors = [profile.one_qubit_error]
        else:
            left, right = item.qubits
            error = profile.directed_edge_errors.get(
                (left, right), profile.directed_edge_errors.get(
                    (right, left), profile.two_qubit_error))
            errors = [error] * (3 if name == "swap" else 1)
        for error in errors:
            log_success += math.log1p(-error)
            error_events += 1
    return {
        "profile": profile.name,
        "calibrated": profile.calibrated,
        "edge_specific": bool(profile.directed_edge_errors),
        "scheduled_makespan_ns": schedule.makespan,
        "scheduled_depth": schedule.depth_after,
        "schedule_depth_delta": schedule.depth_delta,
        "estimated_success_probability": math.exp(log_success),
        "estimated_log_success": log_success,
        "error_events": error_events,
    }
