"""Strict-improvement pairwise exchange refinement."""

from dataclasses import dataclass, field
import math
import networkx as nx

from qweave.core.types import Mapping
from qweave.core.validation import validate_mapping
from .interaction_graph import build_interaction_graph


@dataclass
class LocalSearchResult:
    mapping: Mapping
    initial_objective: float
    final_objective: float
    accepted_moves: list[dict] = field(default_factory=list)
    iterations: int = 0


def mapping_objective(mapping: Mapping, interaction_graph: nx.Graph, coupling_graph: nx.Graph) -> float:
    """Compute sum(weight * shortest-path-distance), or infinity if disconnected."""

    total = 0.0
    for first, second, data in interaction_graph.edges(data=True):
        try:
            distance = nx.shortest_path_length(coupling_graph, mapping[first], mapping[second])
        except nx.NetworkXNoPath:
            return math.inf
        total += data.get("weight", 1) * distance
    return total


def refine_mapping(circuit: object, mapping: Mapping, coupling_graph: nx.Graph, max_iterations: int = 100) -> LocalSearchResult:
    """Accept only strictly improving logical-pair physical-location swaps.

    Each iteration chooses the best improving pair, breaking ties by logical
    pair order. Termination is guaranteed by strict descent of the objective.
    """

    interaction_graph = build_interaction_graph(circuit)
    validate_mapping(mapping, circuit, coupling_graph)
    current = dict(mapping)
    initial = mapping_objective(current, interaction_graph, coupling_graph)
    objective = initial
    moves = []
    iterations = 0
    logicals = sorted(current)
    while iterations < max_iterations:
        candidates = []
        for index, first in enumerate(logicals):
            for second in logicals[index + 1 :]:
                trial = dict(current)
                trial[first], trial[second] = trial[second], trial[first]
                value = mapping_objective(trial, interaction_graph, coupling_graph)
                if value < objective:
                    candidates.append((value, first, second, trial))
        if not candidates:
            break
        value, first, second, current = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        moves.append({"logical_pair": [first, second], "objective_before": objective, "objective_after": value})
        objective = value
        iterations += 1
    return LocalSearchResult(current, initial, objective, moves, iterations)

