"""Qiskit SABRE baseline isolated from the deterministic implementation."""

import time
from qiskit import transpile
from qiskit.transpiler import CouplingMap

from qweave.metrics.circuit_metrics import circuit_metrics


def run_sabre_baseline(circuit: object, coupling_graph: object, seed: int = 7) -> dict:
    """Run Qiskit SABRE with a fixed seed and return reproducible metrics."""

    started = time.perf_counter()
    edges = list(coupling_graph.edges)
    compiled = transpile(
        circuit,
        coupling_map=CouplingMap(edges),
        routing_method="sabre",
        optimization_level=0,
        seed_transpiler=seed,
    )
    runtime = time.perf_counter() - started
    result = circuit_metrics(compiled, circuit, runtime)
    result.update({"circuit": compiled, "seed": seed, "method": "qiskit_sabre"})
    return result
