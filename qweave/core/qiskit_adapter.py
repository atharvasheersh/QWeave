"""Qiskit input/output compatibility and the deterministic gate contract."""

from dataclasses import dataclass

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit import Gate


@dataclass(frozen=True)
class CircuitOperation:
    source_index: int
    operation: Gate
    qubits: tuple[int, ...]


def qubit_index(circuit: QuantumCircuit, qubit: object) -> int:
    """Use Qiskit's public bit lookup, with the legacy bit-index fallback."""

    try:
        return int(circuit.find_bit(qubit).index)
    except AttributeError:
        index = getattr(qubit, "index", None)
        if index is None:
            index = getattr(qubit, "_index", None)
        if index is None:
            raise TypeError("Qiskit qubit has no usable index")
        return int(index)


def unpack_instruction(instruction: object) -> tuple[object, tuple, tuple]:
    """Handle CircuitInstruction and the older three-tuple representation."""

    if hasattr(instruction, "operation"):
        return instruction.operation, tuple(instruction.qubits), tuple(instruction.clbits)
    operation, qubits, clbits = instruction
    return operation, tuple(qubits), tuple(clbits)


def source_operations(circuit: QuantumCircuit) -> list[CircuitOperation]:
    """Accept only unconditioned one- and two-qubit gates for this core."""

    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("QWeave requires a Qiskit QuantumCircuit")
    if circuit.num_qubits < 1:
        raise ValueError("source circuit must contain at least one qubit")
    if circuit.num_clbits:
        raise ValueError("classical bits and dynamic circuits are not supported")
    operations = []
    for index, instruction in enumerate(circuit.data):
        operation, qubits, clbits = unpack_instruction(instruction)
        if (not isinstance(operation, Gate) or len(qubits) not in (1, 2)
                or clbits or getattr(operation, "condition", None) is not None):
            name = getattr(operation, "name", type(operation).__name__)
            raise ValueError(f"unsupported source operation {index}: {name}; only unconditioned one- and two-qubit gates are supported")
        indices = tuple(qubit_index(circuit, bit) for bit in qubits)
        if len(set(indices)) != len(indices):
            raise ValueError(f"source operation {index} repeats a logical qubit")
        operations.append(CircuitOperation(index, operation, indices))
    return operations


def validate_hardware_graph(graph: nx.Graph, required_qubits: int = 1) -> int:
    """Validate the undirected physical model and return output width."""

    if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
        raise TypeError("hardware must be an undirected simple NetworkX Graph")
    if any(type(node) is not int or node < 0 for node in graph.nodes):
        raise ValueError("physical qubit labels must be non-negative integers")
    if nx.number_of_selfloops(graph):
        raise ValueError("hardware graph must not contain self-loops")
    if len(graph) < required_qubits:
        raise ValueError("hardware graph has fewer physical than logical qubits")
    return max(graph.nodes, default=-1) + 1


def physical_output_circuit(source: QuantumCircuit, width: int) -> QuantumCircuit:
    """Create a physical circuit without losing source global phase."""

    output = QuantumCircuit(width, name=f"qweave_{source.name}")
    output.global_phase = source.global_phase
    output.metadata = dict(source.metadata or {})
    return output
