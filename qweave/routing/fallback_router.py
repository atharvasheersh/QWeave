"""Correctness-first deterministic SWAP insertion router."""

from qiskit import QuantumCircuit
from qiskit.circuit.library import SwapGate
import networkx as nx

from qweave.core.types import Mapping, RoutingResult
from qweave.core.validation import validate_mapping, validate_mapping_inverse, validate_two_qubit_legality
from .shortest_path import deterministic_shortest_path


def _index(circuit: QuantumCircuit, bit: object) -> int:
    return int(circuit.find_bit(bit).index)


def route_with_fallback(circuit: QuantumCircuit, coupling_graph: nx.Graph, mapping: Mapping) -> RoutingResult:
    """Route operations in original order using shortest-path SWAPs.

    The source logical qubit is advanced along the selected path until it is
    adjacent to the target. Idle physical nodes remain represented by ``None``
    in the internal inverse layout and require no special handling.
    """

    validate_mapping(mapping, circuit, coupling_graph)
    current = dict(mapping)
    inverse = validate_mapping_inverse(current)
    physical_count = max(coupling_graph.nodes, default=-1) + 1
    if any(not isinstance(node, int) or node < 0 for node in coupling_graph.nodes):
        raise ValueError("physical qubit labels must be non-negative integers")
    output = QuantumCircuit(physical_count, circuit.num_clbits)
    trace: list[dict] = []
    swaps = 0
    for operation, qargs, cargs in circuit.data:
        logicals = [_index(circuit, bit) for bit in qargs]
        if len(logicals) == 2:
            first, second = logicals
            source, target = current[first], current[second]
            if source == target:
                raise ValueError("two logical qubits occupy the same physical qubit")
            if not coupling_graph.has_edge(source, target):
                path = deterministic_shortest_path(coupling_graph, source, target)
                for left, right in zip(path, path[1:-1]):
                    output.append(SwapGate(), [left, right], [])
                    left_logical = inverse.get(left)
                    right_logical = inverse.get(right)
                    if left_logical is not None:
                        current[left_logical] = right
                    if right_logical is not None:
                        current[right_logical] = left
                    inverse[left], inverse[right] = right_logical, left_logical
                    swaps += 1
                    trace.append({"operation": "swap", "edge": [left, right], "mapping": dict(current)})
            output.append(operation, [current[first], current[second]], cargs)
            trace.append({"operation": operation.name, "logical_qubits": logicals, "physical_qubits": [current[first], current[second]]})
        else:
            output.append(operation, [current[logical] for logical in logicals], cargs)
    validate_two_qubit_legality(output, coupling_graph)
    return RoutingResult(output, current, trace, swaps)

