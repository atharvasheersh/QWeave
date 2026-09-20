"""Qiskit SABRE baseline isolated from the deterministic implementation."""

import time
from qiskit import transpile
from qiskit.transpiler import CouplingMap

from qweave.core.qiskit_adapter import source_operations, validate_hardware_graph
from qweave.core.validation import (
    validate_transpiled_small_unitary,
    validate_transpiled_statevector_probes,
    validate_two_qubit_legality,
)
from qweave.metrics.circuit_metrics import circuit_metrics


def run_sabre_baseline(circuit: object, coupling_graph: object, seed: int = 7) -> dict:
    """Run Qiskit SABRE with a fixed seed and return reproducible metrics."""

    operations = source_operations(circuit)
    width = validate_hardware_graph(coupling_graph, circuit.num_qubits)
    if set(coupling_graph.nodes) != set(range(width)):
        raise ValueError("SABRE adapter requires contiguous physical labels 0..p-1")
    if type(seed) is not int or seed < 0:
        raise ValueError("SABRE seed must be a non-negative integer")
    # The deterministic core uses undirected edges. Give Qiskit both CX
    # directions so this reference has the same coupling interpretation.
    edges = sorted({edge for left, right in coupling_graph.edges
                    for edge in ((left, right), (right, left))})
    coupling = CouplingMap(edges)
    for physical in range(width):
        if physical not in coupling.physical_qubits:
            coupling.add_physical_qubit(physical)
    started = time.perf_counter()
    try:
        compiled = transpile(
            circuit,
            coupling_map=coupling,
            routing_method="sabre",
            optimization_level=0,
            seed_transpiler=seed,
        )
    except Exception as exc:
        raise RuntimeError(f"Qiskit SABRE failed on this circuit/hardware: {exc}") from exc
    runtime = time.perf_counter() - started
    validate_two_qubit_legality(compiled, coupling_graph)
    semantic_validation = None
    if all(item.operation.name not in {"measure", "reset"} for item in operations):
        semantic_validation = validate_transpiled_small_unitary(circuit, compiled)
        if semantic_validation is None:
            semantic_validation = validate_transpiled_statevector_probes(circuit, compiled)
    result = circuit_metrics(compiled, circuit, runtime)
    result.update({"circuit": compiled, "seed": seed, "method": "qiskit_sabre",
                   "semantic_validation": semantic_validation})
    return result
