"""Frozen benchmark-v1 circuits, hardware graphs, and learned-policy splits."""

from hashlib import sha256
import random

import networkx as nx
from qiskit import QuantumCircuit, qasm2


BENCHMARK_VERSION = "1.0"
GENERATION_SEED = 20260921
TRAIN_FAMILIES = ("ghz", "layered_cx", "seeded_interactions")
HELD_OUT_FAMILIES = ("star", "distant_pairs")


def _circuits(width: int) -> dict[str, QuantumCircuit]:
    ghz = QuantumCircuit(width, name=f"ghz_{width}")
    ghz.h(0)
    for index in range(width - 1):
        ghz.cx(index, index + 1)

    layered = QuantumCircuit(width, name=f"layered_cx_{width}")
    for offset in (0, 1, 0, 1):
        for first in range(offset, width - 1, 2):
            layered.cx(first, first + 1)

    seeded = QuantumCircuit(width, name=f"seeded_interactions_{width}")
    rng = random.Random(GENERATION_SEED + width)
    seeded.h(0)
    for _ in range(width * 2):
        first, second = rng.sample(range(width), 2)
        seeded.cx(first, second)

    star = QuantumCircuit(width, name=f"star_{width}")
    star.h(0)
    for leaf in range(1, width):
        star.cx(0, leaf)
    for leaf in range(width - 1, 0, -1):
        star.cx(leaf, 0)

    distant = QuantumCircuit(width, name=f"distant_pairs_{width}")
    pairs = [(index, width - index - 1) for index in range(width // 2)]
    for _ in range(2):
        for left, right in pairs:
            distant.h(left)
            distant.cx(left, right)
        for left, right in reversed(pairs):
            distant.cx(right, left)
    return {"ghz": ghz, "layered_cx": layered,
            "seeded_interactions": seeded, "star": star,
            "distant_pairs": distant}


def _topologies(width: int) -> dict[str, nx.Graph]:
    return {
        "path": nx.path_graph(width),
        "ring": nx.cycle_graph(width),
        "grid2": nx.convert_node_labels_to_integers(
            nx.grid_2d_graph(2, width // 2), ordering="sorted"),
    }


def _split(family: str, topology: str) -> str:
    if family in TRAIN_FAMILIES and topology in {"path", "ring"}:
        return "train"
    if family in HELD_OUT_FAMILIES and topology == "grid2":
        return "test"
    return "validation"


def benchmark_manifest() -> dict:
    """Return the immutable 30-case v1 manifest in deterministic order."""

    cases = []
    for width in (4, 6):
        for family, circuit in _circuits(width).items():
            source_qasm = qasm2.dumps(circuit)
            source_hash = sha256(source_qasm.encode("utf-8")).hexdigest()
            for topology, graph in _topologies(width).items():
                cases.append({
                    "case_id": f"{family}_{width}q_{topology}",
                    "split": _split(family, topology),
                    "family": family,
                    "logical_qubits": width,
                    "topology": topology,
                    "hardware_edges": [list(edge) for edge in sorted(
                        tuple(sorted(edge)) for edge in graph.edges)],
                    "source_qasm2": source_qasm,
                    "source_sha256": source_hash,
                    "generation_seed": (GENERATION_SEED + width
                                        if family == "seeded_interactions" else None),
                })
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "generation_seed": GENERATION_SEED,
        "split_policy": (
            "Training uses three families on path/ring. Validation covers their grid2 "
            "transfer and held-out families on path/ring. Test combines the two held-out "
            "families with grid2 and is not used for fitting or selection."),
        "cases": cases,
    }


def load_case(case: dict) -> tuple[QuantumCircuit, nx.Graph]:
    circuit = qasm2.loads(case["source_qasm2"])
    graph = nx.Graph()
    graph.add_nodes_from(range(case["logical_qubits"]))
    graph.add_edges_from(case["hardware_edges"])
    return circuit, graph
