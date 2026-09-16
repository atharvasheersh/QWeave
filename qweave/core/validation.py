"""Mapping, physical-legality, route-replay, and small-unitary checks."""

import networkx as nx

from .qiskit_adapter import (
    qubit_index,
    source_operations,
    unpack_instruction,
    validate_hardware_graph,
)
from .types import Mapping, RoutingResult


def validate_all_qubits_unique(mapping: Mapping) -> bool:
    """Ensure that no two logical qubits use the same physical qubit."""

    if len(mapping) != len(set(mapping.values())):
        raise ValueError("mapping is not injective: physical qubits are repeated")
    return True


def validate_mapping(mapping: Mapping, logical_qubits: int | object, coupling_graph: nx.Graph) -> bool:
    """Validate completeness, injectivity, and physical-node membership."""

    count = logical_qubits if isinstance(logical_qubits, int) else logical_qubits.num_qubits
    validate_hardware_graph(coupling_graph, count)
    if not isinstance(mapping, dict):
        raise TypeError("mapping must be a logical-to-physical dictionary")
    expected = set(range(count))
    if set(mapping) != expected:
        raise ValueError(f"mapping keys must be exactly {sorted(expected)}")
    if any(type(physical) is not int for physical in mapping.values()):
        raise ValueError("mapping values must be integer physical qubits")
    validate_all_qubits_unique(mapping)
    unknown = set(mapping.values()) - set(coupling_graph.nodes)
    if unknown:
        raise ValueError(f"mapping uses unknown physical qubits: {sorted(unknown)}")
    return True


def validate_mapping_inverse(mapping: Mapping) -> dict[int, int]:
    """Return the inverse mapping after checking it is one-to-one."""

    validate_all_qubits_unique(mapping)
    return {physical: logical for logical, physical in mapping.items()}


def validate_two_qubit_legality(circuit: object, coupling_graph: nx.Graph) -> bool:
    """Reject unsupported physical operations and non-edge two-qubit gates."""

    operations = source_operations(circuit)
    validate_hardware_graph(coupling_graph)
    for item in operations:
        if len(item.qubits) == 2:
            first, second = item.qubits
            if not coupling_graph.has_edge(first, second):
                raise ValueError(f"illegal two-qubit operation on physical qubits {(first, second)}")
        elif item.qubits[0] not in coupling_graph:
            raise ValueError(f"operation uses unknown physical qubit {item.qubits[0]}")
    return True


def validate_routing_result(source: object, result: RoutingResult,
                            initial_mapping: Mapping, coupling_graph: nx.Graph) -> bool:
    """Replay inserted SWAPs and emitted gates against the source and layout."""

    source_gates = source_operations(source)
    validate_mapping(initial_mapping, source, coupling_graph)
    width = validate_hardware_graph(coupling_graph, source.num_qubits)
    if result.circuit.num_qubits != width:
        raise ValueError("routed output width does not match hardware labels")
    if result.circuit.global_phase != source.global_phase:
        raise ValueError("routed output lost source global phase")
    if len(result.trace) != len(result.circuit.data):
        raise ValueError("routing trace must contain one event per emitted operation")
    current = dict(initial_mapping)
    inverse = validate_mapping_inverse(current)
    gate_index = 0
    inserted_swaps = 0
    for event, instruction in zip(result.trace, result.circuit.data):
        operation, qargs, cargs = unpack_instruction(instruction)
        physical = tuple(qubit_index(result.circuit, bit) for bit in qargs)
        if cargs:
            raise ValueError("routed output contains unsupported classical operands")
        if event.get("kind") == "inserted_swap":
            if operation.name != "swap" or len(physical) != 2 or not coupling_graph.has_edge(*physical):
                raise ValueError("routing trace has an invalid inserted SWAP")
            left, right = physical
            left_logical, right_logical = inverse.get(left), inverse.get(right)
            if left_logical is not None:
                current[left_logical] = right
            if right_logical is not None:
                current[right_logical] = left
            inverse[left], inverse[right] = right_logical, left_logical
            inserted_swaps += 1
            if event.get("edge") != [left, right] or event.get("mapping") != current:
                raise ValueError("routing trace SWAP layout is inconsistent")
        elif event.get("kind") == "source_gate":
            if gate_index >= len(source_gates):
                raise ValueError("routed output has an extra source gate")
            source_gate = source_gates[gate_index]
            expected = tuple(current[logical] for logical in source_gate.qubits)
            if (event.get("source_index") != source_gate.source_index
                    or event.get("logical_qubits") != list(source_gate.qubits)
                    or event.get("physical_qubits") != list(physical)
                    or physical != expected or operation.name != source_gate.operation.name
                    or repr(operation.params) != repr(source_gate.operation.params)):
                raise ValueError(f"routed source gate {gate_index} disagrees with circuit or layout")
            gate_index += 1
        else:
            raise ValueError("routing trace has an unknown event kind")
    if gate_index != len(source_gates):
        raise ValueError("routed output omits a source gate")
    if result.final_mapping != current or result.swap_count != inserted_swaps:
        raise ValueError("routed final layout or inserted SWAP count is inconsistent")
    validate_two_qubit_legality(result.circuit, coupling_graph)
    return True


def validate_small_unitary_equivalence(source: object, result: RoutingResult,
                                       initial_mapping: Mapping, max_qubits: int = 6) -> bool | None:
    """Check U_routed P_initial = P_final U_source on equal-width small cases.

    Return ``None`` when the equal-width oracle is outside its declared scope.
    This is a unitary test oracle, not general measurement/dynamic validation.
    """

    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    count = source.num_qubits
    if count > max_qubits or result.circuit.num_qubits != count:
        return None
    if set(initial_mapping.values()) != set(range(count)) or set(result.final_mapping.values()) != set(range(count)):
        return None

    def permutation(mapping: Mapping) -> np.ndarray:
        circuit = QuantumCircuit(count)
        position = {logical: logical for logical in range(count)}
        occupant = {physical: physical for physical in range(count)}
        for logical in range(count):
            left, right = position[logical], mapping[logical]
            if left == right:
                continue
            displaced = occupant[right]
            circuit.swap(left, right)
            position[logical], position[displaced] = right, left
            occupant[left], occupant[right] = displaced, logical
        return Operator(circuit).data

    try:
        left = Operator(result.circuit).data @ permutation(initial_mapping)
        right = permutation(result.final_mapping) @ Operator(source).data
    except Exception as exc:
        raise ValueError("small-unitary oracle cannot evaluate this gate set") from exc
    pivot = np.unravel_index(np.argmax(np.abs(right)), right.shape)
    phase = left[pivot] / right[pivot]
    if not np.allclose(left, phase * right, atol=1e-10):
        raise ValueError("routed unitary disagrees with initial/final layouts")
    return True


def validate_transpiled_small_unitary(source: object, compiled: object,
                                     layout_circuit: object | None = None,
                                     max_qubits: int = 6) -> bool | None:
    """Check Qiskit-transpiled unitary output using its exposed layouts.

    ``layout_circuit`` supplies the original layout when checking a scheduled
    copy, whose operations may be reordered without changing that layout.
    Equal-width circuits of at most ``max_qubits`` are the only tested scope.
    """

    count = source.num_qubits
    if count > max_qubits or compiled.num_qubits != count:
        return None
    layout = getattr(layout_circuit or compiled, "layout", None)
    if layout is None:
        raise ValueError("Qiskit output has no transpiler layout for semantic validation")
    try:
        initial = layout.initial_index_layout(filter_ancillas=True)
        final = layout.final_index_layout(filter_ancillas=True)
    except Exception as exc:
        raise ValueError("Qiskit transpiler layout cannot be interpreted") from exc
    if (len(initial) != count or len(final) != count
            or set(initial) != set(range(count)) or set(final) != set(range(count))):
        raise ValueError("Qiskit transpiler layout is not an equal-width permutation")
    result = RoutingResult(compiled, dict(enumerate(final)), [], 0)
    return validate_small_unitary_equivalence(source, result, dict(enumerate(initial)), max_qubits)
