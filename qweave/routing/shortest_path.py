"""Deterministic shortest-path helpers."""

import networkx as nx


def deterministic_shortest_path(graph: nx.Graph, source: int, target: int) -> list[int]:
    """Return a shortest path, choosing lexicographically smallest ties."""

    if source not in graph or target not in graph:
        raise ValueError("shortest-path endpoints must be hardware nodes")
    try:
        paths = nx.all_shortest_paths(graph, source, target)
        return min((list(path) for path in paths), key=lambda path: tuple(path))
    except nx.NetworkXNoPath as exc:
        raise ValueError(f"hardware graph has no path between {source} and {target}") from exc

