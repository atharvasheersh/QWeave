"""Simple identity-mapping plus fallback-routing baseline."""

from qweave.core.types import Mapping
from qweave.routing.fallback_router import route_with_fallback


def identity_mapping(circuit: object, coupling_graph: object) -> Mapping:
    """Use logical index as physical index when the hardware permits it."""

    available = set(coupling_graph.nodes)
    mapping = {logical: logical for logical in range(circuit.num_qubits)}
    if not set(mapping.values()).issubset(available):
        raise ValueError("identity mapping requires physical nodes 0..n-1")
    return mapping


def run_basic_baseline(circuit: object, coupling_graph: object) -> object:
    """Route with the identity layout and deterministic fallback router."""

    return route_with_fallback(circuit, coupling_graph, identity_mapping(circuit, coupling_graph))

