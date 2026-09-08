from qiskit import QuantumCircuit
import networkx as nx

from qweave.mapper.initial_mapper import initial_mapping


def test_initial_mapping_is_complete_injective_and_deterministic():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    graph = nx.path_graph(3)
    first = initial_mapping(circuit, graph)
    second = initial_mapping(circuit, graph)
    assert first.mapping == second.mapping
    assert set(first.mapping) == {0, 1, 2}
    assert len(set(first.mapping.values())) == 3

