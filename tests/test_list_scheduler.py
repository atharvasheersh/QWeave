"""Scheduling must preserve the fixed routed computation and stage boundary."""

import networkx as nx
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qweave.core.types import RoutingResult
from qweave.core.validation import validate_small_unitary_equivalence
from qweave.routing.fallback_router import route_with_fallback
from qweave.scheduling import schedule_routed_circuit, schedule_routing_result


def test_critical_path_priority_packs_disjoint_work_without_depth_claim() -> None:
    circuit = QuantumCircuit(4, global_phase=0.2)
    circuit.x(0)       # short independent branch appears first
    circuit.h(2)
    circuit.cx(2, 3)   # longer critical branch
    result = schedule_routed_circuit(circuit, nx.path_graph(4))
    assert result.timing[0]["gate_index"] == 1
    assert result.layers == [[1, 0], [2]]
    assert result.depth_before == result.depth_after == 2
    assert result.depth_delta == 0
    assert Operator(result.circuit).equiv(Operator(circuit))
    assert result.circuit.global_phase == circuit.global_phase


def test_gate_durations_set_timed_makespan_without_changing_gate_depth() -> None:
    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.h(1)
    circuit.cx(0, 1)
    result = schedule_routed_circuit(circuit, nx.path_graph(2), {"x": 3, "h": 1, "cx": 2})
    assert result.makespan == 5.0
    assert result.layers == [[0, 1], [2]]
    assert result.depth_delta == 0
    assert Operator(result.circuit).equiv(Operator(circuit))


def test_fixed_route_can_be_scheduled_without_new_swap_or_layout_change() -> None:
    source = QuantumCircuit(4)
    source.h(0)
    source.cx(0, 3)
    source.x(2)
    source.cx(1, 3)
    graph = nx.path_graph(4)
    mapping = {logical: logical for logical in range(4)}
    routed = route_with_fallback(source, graph, mapping)
    scheduled = schedule_routing_result(source, routed, mapping, graph)
    assert scheduled.circuit.count_ops().get("swap", 0) == routed.circuit.count_ops().get("swap", 0)
    assert scheduled.depth_delta == 0
    scheduled_with_layout = RoutingResult(scheduled.circuit, routed.final_mapping, [], routed.swap_count)
    assert validate_small_unitary_equivalence(source, scheduled_with_layout, mapping) is True


def test_scheduler_rejects_illegal_input_and_bad_duration() -> None:
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    with pytest.raises(ValueError, match="illegal"):
        schedule_routed_circuit(circuit, nx.path_graph(3))
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    with pytest.raises(ValueError, match="finite positive"):
        schedule_routed_circuit(circuit, nx.path_graph(2), {"cx": 0})
