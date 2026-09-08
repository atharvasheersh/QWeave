"""Weighted logical interaction graph construction."""

import networkx as nx


def _qubit_index(circuit: object, qubit: object) -> int:
    try:
        return int(circuit.find_bit(qubit).index)
    except AttributeError:
        return int(getattr(qubit, "index", getattr(qubit, "_index")))


def build_interaction_graph(circuit: object, decay: float | None = None) -> nx.Graph:
    """Build an undirected weighted graph from two-qubit circuit operations.

    Without ``decay`` each interaction contributes one. With ``decay`` the
    contribution is ``decay ** gate_index`` where ``gate_index`` is the
    operation's position in ``circuit.data``. The graph always includes all
    logical qubits, including isolated qubits.
    """

    if decay is not None and not 0 < decay <= 1:
        raise ValueError("decay must satisfy 0 < decay <= 1")
    graph = nx.Graph()
    graph.add_nodes_from(range(circuit.num_qubits))
    for gate_index, instruction in enumerate(circuit.data):
        qargs = instruction.qubits if hasattr(instruction, "qubits") else instruction[1]
        if len(qargs) != 2:
            continue
        first, second = (_qubit_index(circuit, qubit) for qubit in qargs)
        contribution = 1.0 if decay is not None else 1
        if decay is not None:
            contribution = decay**gate_index
        weight = graph.get_edge_data(first, second, {}).get("weight", 0)
        graph.add_edge(first, second, weight=weight + contribution)
    return graph
