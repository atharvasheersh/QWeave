from qiskit import QuantumCircuit
import networkx as nx
import pytest

from qweave.core.validation import validate_two_qubit_legality
from qweave.routing.fallback_router import route_with_fallback


def test_adjacent_gate_has_no_swap():
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    result = route_with_fallback(circuit, nx.path_graph(2), {0: 0, 1: 1})
    assert result.swap_count == 0
    validate_two_qubit_legality(result.circuit, nx.path_graph(2))


def test_nonadjacent_gate_is_legal_and_updates_layout():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    graph = nx.path_graph(3)
    result = route_with_fallback(circuit, graph, {0: 0, 1: 1, 2: 2})
    assert result.swap_count == 1
    assert result.final_mapping[0] == 1
    validate_two_qubit_legality(result.circuit, graph)


def test_disconnected_hardware_fails_usefully():
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    graph = nx.Graph([(0, 1)])
    graph.add_node(2)
    with pytest.raises(ValueError, match="no path"):
        route_with_fallback(circuit, graph, {0: 0, 1: 2})
