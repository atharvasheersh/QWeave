"""Versioned pilot fixtures; not the final comparative benchmark dataset."""

from hashlib import sha256
import random

import networkx as nx
from qiskit import QuantumCircuit, qasm2


PILOT_VERSION = "0.1"
PILOT_SEED = 20260916


def _circuits(width: int) -> dict[str, QuantumCircuit]:
    ghz = QuantumCircuit(width)
    ghz.h(0)
    for index in range(width - 1):
        ghz.cx(index, index + 1)

    layered = QuantumCircuit(width)
    for offset in (0, 1, 0, 1):
        for first in range(offset, width - 1, 2):
            layered.cx(first, first + 1)

    random_like = QuantumCircuit(width)
    rng = random.Random(PILOT_SEED + width)
    random_like.h(0)
    for _ in range(width * 2):
        first, second = rng.sample(range(width), 2)
        random_like.cx(first, second)
    return {"ghz": ghz, "layered_cx": layered, "seeded_interactions": random_like}


def pilot_manifest() -> dict:
    """Return the same source QASM and edge lists on every invocation."""
    cases = []
    for width in (4, 6, 8):
        topologies = {
            "path": nx.path_graph(width),
            "ring": nx.cycle_graph(width),
            "grid2": nx.convert_node_labels_to_integers(nx.grid_2d_graph(2, width // 2)),
        }
        for family, circuit in _circuits(width).items():
            source_qasm = qasm2.dumps(circuit)
            source_hash = sha256(source_qasm.encode("utf-8")).hexdigest()
            for topology, graph in topologies.items():
                cases.append({
                    "case_id": f"{family}_{width}q_{topology}",
                    "family": family,
                    "logical_qubits": width,
                    "topology": topology,
                    "hardware_edges": [list(edge) for edge in sorted(tuple(sorted(edge)) for edge in graph.edges)],
                    "source_qasm2": source_qasm,
                    "source_sha256": source_hash,
                    "generation_seed": PILOT_SEED + width if family == "seeded_interactions" else None,
                })
    return {"pilot_version": PILOT_VERSION, "cases": cases}
