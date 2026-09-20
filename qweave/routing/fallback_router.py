"""Correctness-first deterministic SWAP insertion router."""

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit.library import SwapGate

from qweave.core.qiskit_adapter import physical_output_circuit, source_operations, validate_hardware_graph
from qweave.core.types import Mapping, RoutingResult
from qweave.core.validation import validate_mapping, validate_mapping_inverse, validate_routing_result
from .shortest_path import deterministic_shortest_path


def route_with_fallback(circuit: QuantumCircuit, coupling_graph: nx.Graph, mapping: Mapping) -> RoutingResult:
    """Route operations in original order using shortest-path SWAPs.

    The source logical qubit is advanced along the selected path until it is
    adjacent to the target. Idle physical nodes remain represented by ``None``
    in the internal inverse layout and require no special handling.
    """

    operations = source_operations(circuit)
    physical_count = validate_hardware_graph(coupling_graph, circuit.num_qubits)
    validate_mapping(mapping, circuit, coupling_graph)
    current = dict(mapping)
    inverse = validate_mapping_inverse(current)
    output = physical_output_circuit(circuit, physical_count)
    trace: list[dict] = []
    swaps = 0
    for item in operations:
        operation = item.operation
        logicals = item.qubits
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
                    trace.append({"kind": "inserted_swap", "operation": "swap", "edge": [left, right], "mapping": dict(current)})
            physical = [current[first], current[second]]
            output.append(operation, physical, [output.clbits[index] for index in item.clbits])
        else:
            physical = [current[logical] for logical in logicals]
            output.append(operation, physical, [output.clbits[index] for index in item.clbits])
        trace.append({"kind": "source_gate", "operation": operation.name,
                      "source_index": item.source_index, "logical_qubits": list(logicals),
                      "physical_qubits": physical, "classical_bits": list(item.clbits)})
    result = RoutingResult(output, current, trace, swaps)
    validate_routing_result(circuit, result, mapping, coupling_graph)
    return result
