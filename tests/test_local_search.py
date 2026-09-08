from qiskit import QuantumCircuit
import networkx as nx

from qweave.mapper.local_search import refine_mapping


def test_local_search_only_accepts_strict_improvements():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    result = refine_mapping(circuit, {0: 0, 1: 2, 2: 1}, nx.path_graph(3))
    assert result.final_objective <= result.initial_objective
    assert all(move["objective_after"] < move["objective_before"] for move in result.accepted_moves)
    assert result.iterations == len(result.accepted_moves)

