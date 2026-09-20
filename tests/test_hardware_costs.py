"""Declared hardware assumptions and directed legality stay explicit."""

import networkx as nx
import pytest
from qiskit import QuantumCircuit

from qweave.metrics.hardware_costs import (
    HardwareCostProfile,
    estimate_hardware_cost,
    validate_coupling_legality,
)


def test_directed_cx_requires_ordered_edge_and_swap_requires_reciprocal_edges() -> None:
    directed = nx.DiGraph([(0, 1), (1, 2), (2, 1)])
    forward = QuantumCircuit(3)
    forward.cx(0, 1)
    assert validate_coupling_legality(forward, directed)
    reverse = QuantumCircuit(3)
    reverse.cx(1, 0)
    with pytest.raises(ValueError, match="illegal cx direction"):
        validate_coupling_legality(reverse, directed)
    swap = QuantumCircuit(3)
    swap.swap(0, 1)
    with pytest.raises(ValueError, match="illegal swap direction"):
        validate_coupling_legality(swap, directed)
    reciprocal_swap = QuantumCircuit(3)
    reciprocal_swap.swap(1, 2)
    assert validate_coupling_legality(reciprocal_swap, directed)


def test_declared_durations_and_edge_errors_produce_separate_costs() -> None:
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    graph = nx.DiGraph([(0, 1), (1, 0), (1, 2), (2, 1)])
    profile = HardwareCostProfile(
        name="test_profile",
        durations_ns={"h": 10.0, "cx": 100.0},
        directed_edge_errors={(0, 1): 0.02, (1, 2): 0.03},
    )
    metrics = estimate_hardware_cost(circuit, graph, profile)
    assert metrics["scheduled_makespan_ns"] == 210.0
    assert metrics["calibrated"] is True
    assert metrics["error_events"] == 3
    assert metrics["estimated_success_probability"] == pytest.approx(
        (1 - profile.one_qubit_error) * 0.98 * 0.97)


@pytest.mark.parametrize("kwargs", [
    {"one_qubit_error": 1.0},
    {"measurement_error": -0.1},
    {"durations_ns": {"cx": -1}},
])
def test_invalid_hardware_profile_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        HardwareCostProfile(**kwargs)
