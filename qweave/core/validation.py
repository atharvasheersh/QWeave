"""Validation utilities for mapping and hardware legality."""

import networkx as nx

from .types import Mapping


def _qubit_index(circuit: object, qubit: object) -> int:
    try:
        return int(circuit.find_bit(qubit).index)
    except AttributeError:
        return int(getattr(qubit, "index", getattr(qubit, "_index")))


def validate_all_qubits_unique(mapping: Mapping) -> bool:
    """Ensure that no two logical qubits use the same physical qubit."""

    if len(mapping) != len(set(mapping.values())):
        raise ValueError("mapping is not injective: physical qubits are repeated")
    return True


def validate_mapping(mapping: Mapping, logical_qubits: int | object, coupling_graph: nx.Graph) -> bool:
    """Validate completeness, injectivity, and physical-node membership."""

    count = logical_qubits if isinstance(logical_qubits, int) else logical_qubits.num_qubits
    expected = set(range(count))
    if set(mapping) != expected:
        raise ValueError(f"mapping keys must be exactly {sorted(expected)}")
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
    """Check that every two-qubit operation uses an undirected hardware edge."""

    for instruction in circuit.data:
        qargs = instruction.qubits if hasattr(instruction, "qubits") else instruction[1]
        if len(qargs) != 2:
            continue
        first, second = (_qubit_index(circuit, qubit) for qubit in qargs)
        if first == second or not coupling_graph.has_edge(first, second):
            raise ValueError(f"illegal two-qubit operation on physical qubits {(first, second)}")
    return True
