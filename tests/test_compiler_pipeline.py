"""The integrated core must expose each stage without mixing baselines."""

import networkx as nx
import pytest
from qiskit import QuantumCircuit

from qweave.core.compiler import compile_deterministic
from qweave.core.validation import validate_two_qubit_legality


@pytest.mark.parametrize("method", ["basic", "weighted"])
def test_integrated_path_returns_validated_route_and_separate_schedule(method: str) -> None:
    source = QuantumCircuit(4)
    source.h(0)
    source.cx(0, 3)
    source.x(1)
    source.cx(1, 2)
    graph = nx.path_graph(4)
    result = compile_deterministic(source, graph, method=method, schedule=True)
    assert result.method == method
    assert result.semantic_validation is True
    assert result.scheduled is not None
    assert result.stage_metrics["routed_depth"] == result.routed.circuit.depth()
    assert result.stage_metrics["scheduled_depth"] == result.circuit.depth()
    assert result.stage_metrics["schedule_depth_delta"] == (
        result.circuit.depth() - result.routed.circuit.depth())
    assert validate_two_qubit_legality(result.circuit, graph)


def test_integrated_path_rejects_unknown_method_and_unapplied_durations() -> None:
    source = QuantumCircuit(2)
    source.cx(0, 1)
    graph = nx.path_graph(2)
    with pytest.raises(ValueError, match="basic.*weighted"):
        compile_deterministic(source, graph, method="oracle")
    with pytest.raises(ValueError, match="schedule=True"):
        compile_deterministic(source, graph, gate_durations={"cx": 2})


def test_compile_accepts_reset_and_terminal_measurement_with_symbolic_replay() -> None:
    source = QuantumCircuit(3, 1)
    source.h(0)
    source.cx(0, 2)
    source.reset(1)
    source.measure(2, 0)
    compiled = compile_deterministic(source, nx.path_graph(3), method="weighted")
    assert compiled.stage_metrics["symbolic_route_replay"] is True
    assert compiled.semantic_validation is None
    assert compiled.circuit.num_clbits == 1
