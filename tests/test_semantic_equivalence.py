"""Check the router's unitary semantics across initial and final layouts.

This is a small-instance test oracle for unitary circuits on equal-width
hardware. It is deliberately separate from edge-legality validation.
"""

import networkx as nx
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qweave.routing.fallback_router import route_with_fallback


def _layout_permutation(mapping: dict[int, int]) -> Operator:
    """Move each logical wire from identity position to its physical position."""
    width = len(mapping)
    circuit = QuantumCircuit(width)
    position = {logical: logical for logical in range(width)}
    occupant = {physical: physical for physical in range(width)}
    for logical in range(width):
        source, target = position[logical], mapping[logical]
        if source == target:
            continue
        displaced = occupant[target]
        circuit.swap(source, target)
        position[logical], position[displaced] = target, source
        occupant[source], occupant[target] = displaced, logical
    return Operator(circuit)


def _equivalent_under_layouts(source: QuantumCircuit, compiled: QuantumCircuit,
                              initial: dict[int, int], final: dict[int, int]) -> bool:
    left = Operator(compiled).data @ _layout_permutation(initial).data
    right = _layout_permutation(final).data @ Operator(source).data
    return bool(np.allclose(left, right, atol=1e-10))


@pytest.mark.parametrize("mapping", [
    {0: 0, 1: 1, 2: 2, 3: 3},
    {0: 1, 1: 3, 2: 0, 3: 2},
])
def test_router_preserves_unitary_with_layout_updates(mapping: dict[int, int]) -> None:
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 3)
    circuit.rz(0.3, 3)
    circuit.cx(2, 1)
    circuit.x(0)
    circuit.cx(0, 2)
    routed = route_with_fallback(circuit, nx.path_graph(4), mapping)
    assert _equivalent_under_layouts(circuit, routed.circuit, mapping, routed.final_mapping)


def test_unitary_oracle_detects_incorrect_final_layout() -> None:
    circuit = QuantumCircuit(4)
    circuit.x(0)
    circuit.cx(0, 3)
    initial = {logical: logical for logical in range(4)}
    routed = route_with_fallback(circuit, nx.path_graph(4), initial)
    assert routed.swap_count > 0
    assert _equivalent_under_layouts(circuit, routed.circuit, initial, routed.final_mapping)
    assert not _equivalent_under_layouts(circuit, routed.circuit, initial, initial)
