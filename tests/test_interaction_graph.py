from qiskit import QuantumCircuit

from qweave.mapper.interaction_graph import build_interaction_graph


def test_nodes_and_repeated_weights():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.cx(0, 1)
    circuit.h(2)
    graph = build_interaction_graph(circuit)
    assert set(graph.nodes) == {0, 1, 2}
    assert graph[0][1]["weight"] == 2
    assert graph.number_of_edges() == 1


def test_decay_uses_operation_position():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    graph = build_interaction_graph(circuit, decay=0.5)
    assert graph[0][1]["weight"] == 0.5

