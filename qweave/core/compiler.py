"""Small integrated entry point for the deterministic reference path."""

from dataclasses import dataclass

import networkx as nx
from qiskit import QuantumCircuit

from qweave.baselines.basic import identity_mapping
from qweave.mapper.initial_mapper import initial_mapping
from qweave.mapper.local_search import refine_mapping
from qweave.routing.fallback_router import route_with_fallback
from qweave.scheduling import ScheduleResult, schedule_routing_result

from .qiskit_adapter import source_operations, validate_hardware_graph
from .types import Mapping, RoutingResult
from .validation import validate_small_unitary_equivalence


@dataclass
class DeterministicCompileResult:
    method: str
    initial_mapping: Mapping
    routed: RoutingResult
    scheduled: ScheduleResult | None
    semantic_validation: bool | None
    stage_metrics: dict

    @property
    def circuit(self) -> QuantumCircuit:
        return self.scheduled.circuit if self.scheduled is not None else self.routed.circuit


def compile_deterministic(circuit: QuantumCircuit, coupling_graph: nx.Graph,
                          method: str = "weighted", schedule: bool = False,
                          gate_durations: dict[str, float] | None = None) -> DeterministicCompileResult:
    """Compile Basic or weighted mapping through the same validated router.

    SABRE and CP-SAT remain separate reference methods. CP-SAT optimizes only
    the static initial-mapping surrogate, never this routing pipeline.
    """

    if method not in {"basic", "weighted"}:
        raise ValueError("deterministic method must be 'basic' or 'weighted'")
    if gate_durations is not None and not schedule:
        raise ValueError("gate durations require schedule=True")
    source_operations(circuit)
    validate_hardware_graph(coupling_graph, circuit.num_qubits)
    if method == "basic":
        mapping = identity_mapping(circuit, coupling_graph)
    else:
        seed = initial_mapping(circuit, coupling_graph)
        mapping = refine_mapping(circuit, seed.mapping, coupling_graph).mapping
    routed = route_with_fallback(circuit, coupling_graph, mapping)
    semantic = validate_small_unitary_equivalence(circuit, routed, mapping)
    scheduled = (schedule_routing_result(circuit, routed, mapping, coupling_graph, gate_durations)
                 if schedule else None)
    metrics = {"source_depth": circuit.depth(), "routed_depth": routed.circuit.depth(),
               "inserted_routing_swaps": routed.swap_count,
               "semantic_validation": semantic,
               "scheduled_depth": scheduled.depth_after if scheduled else None,
               "schedule_depth_delta": scheduled.depth_delta if scheduled else None,
               "scheduled_makespan": scheduled.makespan if scheduled else None}
    return DeterministicCompileResult(method, dict(mapping), routed, scheduled, semantic, metrics)
