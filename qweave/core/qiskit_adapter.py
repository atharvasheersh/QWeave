"""Qiskit input/output compatibility and the deterministic gate contract."""

from dataclasses import dataclass

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Instruction, Measure, Reset


@dataclass(frozen=True)
class CircuitOperation:
    source_index: int
    operation: Instruction
    qubits: tuple[int, ...]
    clbits: tuple[int, ...] = ()


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


def clbit_index(circuit: QuantumCircuit, clbit: object) -> int:
    """Return a classical-bit index through Qiskit's public lookup."""

    try:
        return int(circuit.find_bit(clbit).index)
    except AttributeError:
        index = getattr(clbit, "index", getattr(clbit, "_index", None))
        if index is None:
            raise TypeError("Qiskit classical bit has no usable index")
        return int(index)


def unpack_instruction(instruction: object) -> tuple[object, tuple, tuple]:
    """Handle CircuitInstruction and the older three-tuple representation."""

    if hasattr(instruction, "operation"):
        return instruction.operation, tuple(instruction.qubits), tuple(instruction.clbits)
    operation, qubits, clbits = instruction
    return operation, tuple(qubits), tuple(clbits)


def source_operations(circuit: QuantumCircuit) -> list[CircuitOperation]:
    """Parse the declared gate/reset/terminal-measurement contract.

    Unconditioned one- and two-qubit gates and one-qubit resets are accepted.
    Measurements must form a terminal suffix and preserve their classical-bit
    destinations. Dynamic control flow and classically conditioned gates remain
    explicit errors because routing their divergent layouts is not implemented.
    """

    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("QWeave requires a Qiskit QuantumCircuit")
    if circuit.num_qubits < 1:
        raise ValueError("source circuit must contain at least one qubit")
    operations = []
    measurement_started = False
    for index, instruction in enumerate(circuit.data):
        operation, qubits, clbits = unpack_instruction(instruction)
        if getattr(operation, "condition", None) is not None or operation.name in {"if_else", "while_loop", "for_loop", "switch_case"}:
            name = getattr(operation, "name", type(operation).__name__)
            raise ValueError(f"unsupported source operation {index}: {name}; dynamic or classically conditioned operations are not supported")
        is_measure = isinstance(operation, Measure) or operation.name == "measure"
        is_reset = isinstance(operation, Reset) or operation.name == "reset"
        is_gate = isinstance(operation, Gate)
        valid_shape = ((is_gate and len(qubits) in (1, 2) and not clbits)
                       or (is_reset and len(qubits) == 1 and not clbits)
                       or (is_measure and len(qubits) == 1 and len(clbits) == 1))
        if not valid_shape:
            name = getattr(operation, "name", type(operation).__name__)
            raise ValueError(f"unsupported source operation {index}: {name}; expected an unconditioned one-/two-qubit gate, reset, or terminal measurement")
        if measurement_started and not is_measure:
            raise ValueError("measurements must form a terminal suffix")
        measurement_started = measurement_started or is_measure
        indices = tuple(qubit_index(circuit, bit) for bit in qubits)
        if len(set(indices)) != len(indices):
            raise ValueError(f"source operation {index} repeats a logical qubit")
        classical = tuple(clbit_index(circuit, bit) for bit in clbits)
        operations.append(CircuitOperation(index, operation, indices, classical))
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

    output = QuantumCircuit(width, source.num_clbits, name=f"qweave_{source.name}")
    output.global_phase = source.global_phase
    output.metadata = dict(source.metadata or {})
    return output
