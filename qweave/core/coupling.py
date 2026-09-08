"""Hardware coupling graph helpers."""

import networkx as nx


def coupling_graph_from_edges(edges: list[tuple[int, int]], num_qubits: int | None = None) -> nx.Graph:
    """Build an undirected hardware graph from physical edge pairs."""

    graph = nx.Graph()
    if num_qubits is not None:
        graph.add_nodes_from(range(num_qubits))
    graph.add_edges_from(edges)
    if not graph.nodes:
        raise ValueError("coupling graph must contain at least one physical qubit")
    return graph


def coupling_graph_from_qiskit(coupling_map: object) -> nx.Graph:
    """Convert a Qiskit ``CouplingMap`` or edge iterable to an undirected graph."""

    if hasattr(coupling_map, "get_edges"):
        edges = list(coupling_map.get_edges())
        size = coupling_map.size() if hasattr(coupling_map, "size") else None
        return coupling_graph_from_edges(edges, size)
    return coupling_graph_from_edges(list(coupling_map))

