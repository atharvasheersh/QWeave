"""Failures that were previously accepted or hidden by edge-only checks."""

from copy import deepcopy

import networkx as nx
import pytest
from qiskit import QuantumCircuit

from qweave.baselines.sabre import run_sabre_baseline
from qweave.core.validation import (
    validate_routing_result,
    validate_small_unitary_equivalence,
    validate_transpiled_small_unitary,
)
from qweave.routing.fallback_router import route_with_fallback


def test_global_phase_and_source_swap_survive_routing() -> None:
    circuit = QuantumCircuit(3, global_phase=0.37)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.swap(1, 2)
    circuit.x(0)
    mapping = {0: 0, 1: 1, 2: 2}
    graph = nx.path_graph(3)
    result = route_with_fallback(circuit, graph, mapping)
    assert result.circuit.global_phase == circuit.global_phase
    assert result.swap_count == 2  # the source SWAP also needs routing, but is not counted as inserted
    assert validate_small_unitary_equivalence(circuit, result, mapping) is True
    assert validate_routing_result(circuit, result, mapping, graph)


def test_replay_rejects_a_forged_final_layout_and_source_gate() -> None:
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    graph = nx.path_graph(3)
    mapping = {0: 0, 1: 1, 2: 2}
    result = route_with_fallback(circuit, graph, mapping)
    bad = deepcopy(result)
    bad.final_mapping = mapping
    with pytest.raises(ValueError, match="final layout"):
        validate_routing_result(circuit, bad, mapping, graph)
    bad = deepcopy(result)
    bad.trace[-1]["physical_qubits"] = [0, 2]
    with pytest.raises(ValueError, match="source gate"):
        validate_routing_result(circuit, bad, mapping, graph)


@pytest.mark.parametrize("operation", ["measure", "reset", "barrier", "ccx"])
def test_unsupported_operations_fail_before_output(operation: str) -> None:
    circuit = QuantumCircuit(3, 1 if operation == "measure" else 0)
    if operation == "measure":
        circuit.measure(0, 0)
    elif operation == "reset":
        circuit.reset(0)
    elif operation == "barrier":
        circuit.barrier(0, 1)
    else:
        circuit.ccx(0, 1, 2)
    with pytest.raises(ValueError, match="classical bits|unsupported source operation"):
        route_with_fallback(circuit, nx.path_graph(3), {0: 0, 1: 1, 2: 2})


def test_directed_hardware_and_invalid_labels_fail_explicitly() -> None:
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    with pytest.raises(TypeError, match="undirected"):
        route_with_fallback(circuit, nx.DiGraph([(0, 1)]), {0: 0, 1: 1})
    with pytest.raises(ValueError, match="non-negative integers"):
        route_with_fallback(circuit, nx.Graph([(0, -1)]), {0: 0, 1: -1})


def test_sabre_adapter_handles_undirected_edges_and_idle_sites() -> None:
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    graph = nx.path_graph(4)
    result = run_sabre_baseline(circuit, graph, seed=3)
    assert result["seed"] == 3
    assert result["circuit"].num_qubits >= 3
    assert result["semantic_validation"] is None  # width differs from source
    with pytest.raises(ValueError, match="contiguous"):
        run_sabre_baseline(circuit, nx.Graph([(0, 1), (1, 3)]), seed=3)


def test_sabre_small_layout_oracle_detects_a_changed_output() -> None:
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 2)
    result = run_sabre_baseline(circuit, nx.path_graph(3), seed=3)
    assert result["semantic_validation"] is True
    bad = result["circuit"].copy()
    bad.x(0)
    with pytest.raises(ValueError, match="unitary disagrees"):
        validate_transpiled_small_unitary(circuit, bad, result["circuit"])
