"""Frozen benchmark-v2 dataset with named circuits and held-out transfer."""

from hashlib import sha256
import math
import random

import networkx as nx
from qiskit import QuantumCircuit, qasm2


BENCHMARK_VERSION = "2.0"
GENERATION_SEED = 20260921
WIDTHS = (4, 6, 8)
INSTANCES = (0, 1)
TRAIN_FAMILIES = (
    "ghz", "layered_cx", "seeded_interactions",
    "bernstein_vazirani", "qaoa_maxcut",
)
HELD_OUT_FAMILIES = ("qft", "hardware_efficient")


def _ghz(width: int, instance: int) -> QuantumCircuit:
    circuit = QuantumCircuit(width, name=f"ghz_{width}_{instance}")
    root = instance % width
    circuit.h(root)
    order = list(range(root, width)) + list(range(root - 1, -1, -1))
    for left, right in zip(order, order[1:]):
        circuit.cx(left, right)
    return circuit


def _layered(width: int, instance: int) -> QuantumCircuit:
    circuit = QuantumCircuit(width, name=f"layered_{width}_{instance}")
    for layer in range(4 + instance):
        offset = (layer + instance) % 2
        for first in range(offset, width - 1, 2):
            circuit.cx(first, first + 1)
        for qubit in range(width):
            circuit.rz(0.07 * (layer + 1) * (qubit + 1), qubit)
    return circuit


def _seeded(width: int, instance: int) -> QuantumCircuit:
    circuit = QuantumCircuit(width, name=f"seeded_{width}_{instance}")
    rng = random.Random(GENERATION_SEED + 101 * width + instance)
    for qubit in range(width):
        circuit.ry(rng.uniform(-math.pi, math.pi), qubit)
    for _ in range(width * (2 + instance)):
        left, right = rng.sample(range(width), 2)
        circuit.cx(left, right)
    return circuit


def _bernstein_vazirani(width: int, instance: int) -> QuantumCircuit:
    circuit = QuantumCircuit(width, name=f"bv_{width}_{instance}")
    ancilla = width - 1
    secret_rng = random.Random(GENERATION_SEED + 211 * width + instance)
    secret = [secret_rng.randrange(2) for _ in range(width - 1)]
    if not any(secret):
        secret[instance % (width - 1)] = 1
    circuit.x(ancilla)
    circuit.h(range(width))
    for qubit, bit in enumerate(secret):
        if bit:
            circuit.cx(qubit, ancilla)
    circuit.h(range(width - 1))
    return circuit


def _qaoa(width: int, instance: int) -> QuantumCircuit:
    circuit = QuantumCircuit(width, name=f"qaoa_{width}_{instance}")
    gamma, beta = (0.31 + 0.09 * instance, 0.22 + 0.07 * instance)
    circuit.h(range(width))
    for layer in range(2 + instance):
        for qubit in range(width):
            target = (qubit + 1) % width
            circuit.cx(qubit, target)
            circuit.rz(2 * gamma, target)
            circuit.cx(qubit, target)
        for qubit in range(0, width, 2):
            target = (qubit + width // 2) % width
            circuit.cx(qubit, target)
            circuit.rz(gamma, target)
            circuit.cx(qubit, target)
        for qubit in range(width):
            circuit.rx(2 * beta, qubit)
        gamma *= 0.91
        beta *= 1.07
    return circuit


def _qft(width: int, instance: int) -> QuantumCircuit:
    circuit = QuantumCircuit(width, name=f"qft_{width}_{instance}")
    repeats = 1 + instance
    for _ in range(repeats):
        for target in range(width):
            circuit.h(target)
            for control in range(target + 1, width):
                angle = math.pi / (2 ** (control - target))
                circuit.rz(angle / 2, control)
                circuit.cx(control, target)
                circuit.rz(-angle / 2, target)
                circuit.cx(control, target)
                circuit.rz(angle / 2, target)
        for left in range(width // 2):
            right = width - left - 1
            circuit.cx(left, right)
            circuit.cx(right, left)
            circuit.cx(left, right)
    return circuit


def _hardware_efficient(width: int, instance: int) -> QuantumCircuit:
    circuit = QuantumCircuit(width, name=f"hea_{width}_{instance}")
    rng = random.Random(GENERATION_SEED + 307 * width + instance)
    for _ in range(3 + instance):
        for qubit in range(width):
            circuit.ry(rng.uniform(-math.pi, math.pi), qubit)
            circuit.rz(rng.uniform(-math.pi, math.pi), qubit)
        for qubit in range(width):
            circuit.cx(qubit, (qubit + 1) % width)
    return circuit


BUILDERS = {
    "ghz": _ghz,
    "layered_cx": _layered,
    "seeded_interactions": _seeded,
    "bernstein_vazirani": _bernstein_vazirani,
    "qaoa_maxcut": _qaoa,
    "qft": _qft,
    "hardware_efficient": _hardware_efficient,
}


def _topologies(width: int) -> dict[str, nx.Graph]:
    chorded = nx.cycle_graph(width)
    chorded.add_edges_from((index, (index + width // 2) % width)
                           for index in range(width // 2))
    return {
        "path": nx.path_graph(width),
        "ring": nx.cycle_graph(width),
        "grid2": nx.convert_node_labels_to_integers(
            nx.grid_2d_graph(2, width // 2), ordering="sorted"),
        "chorded_ring": chorded,
        "grid2_idle": nx.convert_node_labels_to_integers(
            nx.grid_2d_graph(2, width // 2 + 1), ordering="sorted"),
    }


def _split(family: str, topology: str) -> str | None:
    if family in TRAIN_FAMILIES and topology in {"path", "ring"}:
        return "train"
    if ((family in TRAIN_FAMILIES and topology in {"grid2", "chorded_ring"})
            or (family in HELD_OUT_FAMILIES and topology in {"path", "ring"})):
        return "validation"
    if family in HELD_OUT_FAMILIES and topology in {"grid2", "chorded_ring", "grid2_idle"}:
        return "test"
    return None


def _directed_edges(graph: nx.Graph) -> list[list[int]]:
    edges = []
    for index, (left, right) in enumerate(sorted(tuple(sorted(edge)) for edge in graph.edges)):
        edges.append([left, right] if index % 2 == 0 else [right, left])
        if index % 3 == 0:
            edges.append([right, left] if index % 2 == 0 else [left, right])
    return sorted(edges)


def benchmark_v2_manifest() -> dict:
    """Return the frozen 180-case v2 manifest in deterministic order."""

    cases = []
    for width in WIDTHS:
        for family, builder in BUILDERS.items():
            for instance in INSTANCES:
                circuit = builder(width, instance)
                source_qasm = qasm2.dumps(circuit)
                source_hash = sha256(source_qasm.encode("utf-8")).hexdigest()
                for topology, graph in _topologies(width).items():
                    split = _split(family, topology)
                    if split is None:
                        continue
                    edges = sorted(tuple(sorted(edge)) for edge in graph.edges)
                    cases.append({
                        "case_id": f"{family}_i{instance}_{width}q_{topology}",
                        "split": split,
                        "family": family,
                        "instance": instance,
                        "logical_qubits": width,
                        "physical_qubits": graph.number_of_nodes(),
                        "topology": topology,
                        "hardware_edges": [list(edge) for edge in edges],
                        "directed_hardware_edges": _directed_edges(graph),
                        "edge_error_rates": [
                            {"edge": list(edge), "error": 0.006 + 0.002 * (index % 4)}
                            for index, edge in enumerate(edges)
                        ],
                        "source_qasm2": source_qasm,
                        "source_sha256": source_hash,
                        "generation_seed": GENERATION_SEED,
                    })
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "generation_seed": GENERATION_SEED,
        "split_policy": (
            "Training uses five families on path/ring. Validation covers topology "
            "transfer plus held-out families on path/ring. Test uses QFT and "
            "hardware-efficient families on grid, chorded-ring, and idle-grid hardware."),
        "cases": cases,
    }


def load_v2_case(case: dict) -> tuple[QuantumCircuit, nx.Graph]:
    circuit = qasm2.loads(case["source_qasm2"])
    graph = nx.Graph()
    graph.add_nodes_from(range(case["physical_qubits"]))
    graph.add_edges_from(case["hardware_edges"])
    return circuit, graph


def directed_v2_graph(case: dict) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(case["physical_qubits"]))
    graph.add_edges_from(case["directed_hardware_edges"])
    return graph
