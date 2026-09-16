"""Weighted logical interaction graph construction."""

import networkx as nx

from qweave.core.qiskit_adapter import source_operations


def build_interaction_graph(circuit: object, decay: float | None = None) -> nx.Graph:
    """Build an undirected weighted graph from two-qubit circuit operations.

    Without ``decay`` each interaction contributes one. With ``decay`` the
    contribution is ``decay ** gate_index`` where ``gate_index`` is the
    operation's position in ``circuit.data``. The graph always includes all
    logical qubits, including isolated qubits.
    """

    if decay is not None and not 0 < decay <= 1:
        raise ValueError("decay must satisfy 0 < decay <= 1")
    operations = source_operations(circuit)
    graph = nx.Graph()
    graph.add_nodes_from(range(circuit.num_qubits))
    for item in operations:
        if len(item.qubits) != 2:
            continue
        first, second = item.qubits
        contribution = 1.0 if decay is not None else 1
        if decay is not None:
            contribution = decay**item.source_index
        weight = graph.get_edge_data(first, second, {}).get("weight", 0)
        graph.add_edge(first, second, weight=weight + contribution)
    return graph
