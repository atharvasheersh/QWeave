"""Deterministic weighted initial logical-to-physical mapper."""

import math
import networkx as nx

from qweave.core.types import MappingResult, stable_node_key
from .interaction_graph import build_interaction_graph


def _distance(graph: nx.Graph, first: object, second: object) -> float:
    try:
        return float(nx.shortest_path_length(graph, first, second))
    except nx.NetworkXNoPath:
        return math.inf


def initial_mapping(circuit: object, coupling_graph: nx.Graph, decay: float | None = None) -> MappingResult:
    """Map logical qubits using interaction strength and hardware centrality.

    The objective is weighted shortest-path distance. Ties use ascending
    logical and physical indices. Complexity is dominated by all-pairs paths.
    """

    interactions = build_interaction_graph(circuit, decay)
    if len(coupling_graph) < circuit.num_qubits:
        raise ValueError("coupling graph has fewer physical than logical qubits")
    closeness = nx.closeness_centrality(coupling_graph)
    physical_order = sorted(
        coupling_graph.nodes,
        key=lambda node: (-closeness[node], -coupling_graph.degree[node], stable_node_key(node)),
    )
    logical_degree = dict(interactions.degree(weight="weight"))
    seed = min(interactions.nodes, key=lambda node: (-logical_degree[node], node))
    mapping = {seed: physical_order[0]}
    trace = [{"logical": seed, "physical": physical_order[0], "reason": "seed"}]
    remaining = set(interactions.nodes) - {seed}
    while remaining:
        logical = min(
            remaining,
            key=lambda node: (
                -sum(interactions[node][mapped].get("weight", 0) for mapped in mapping),
                -logical_degree[node],
                node,
            ),
        )
        candidates = [node for node in physical_order if node not in mapping.values()]
        scored = []
        for physical in candidates:
            cost = sum(
                interactions[logical][mapped_logical].get("weight", 0)
                * _distance(coupling_graph, physical, mapped_physical)
                for mapped_logical, mapped_physical in mapping.items()
                if interactions.has_edge(logical, mapped_logical)
            )
            scored.append((cost, stable_node_key(physical), physical))
        cost, _, physical = min(scored)
        mapping[logical] = physical
        trace.append({"logical": logical, "physical": physical, "cost": cost})
        remaining.remove(logical)
    return MappingResult(mapping=mapping, trace=trace)

