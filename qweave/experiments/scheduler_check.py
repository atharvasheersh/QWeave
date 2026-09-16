"""Narrow fixed-route scheduler check; these are not benchmark results."""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import networkx as nx
import qiskit
from qiskit import qasm2

from qweave.baselines.basic import run_basic_baseline
from qweave.baselines.sabre import run_sabre_baseline
from qweave.core.types import RoutingResult
from qweave.core.validation import validate_small_unitary_equivalence, validate_transpiled_small_unitary
from qweave.mapper.initial_mapper import initial_mapping
from qweave.mapper.interaction_graph import build_interaction_graph
from qweave.mapper.local_search import refine_mapping
from qweave.oracle.cp_sat import solve_initial_mapping
from qweave.routing.fallback_router import route_with_fallback
from qweave.scheduling import schedule_routed_circuit, schedule_routing_result

from .smoke import sample_circuits


def run_scheduler_check(output_dir: str | Path = "results", seed: int = 7) -> Path:
    """Compare depth before/after scheduling for each *fixed* legal route."""

    graph = nx.path_graph(4)
    records = []
    for name, source in sample_circuits().items():
        source_hash = hashlib.sha256(qasm2.dumps(source).encode("utf-8")).hexdigest()
        basic_mapping = {logical: logical for logical in range(source.num_qubits)}
        basic = run_basic_baseline(source, graph)
        first = initial_mapping(source, graph)
        refined = refine_mapping(source, first.mapping, graph)
        weighted = route_with_fallback(source, graph, refined.mapping)
        for method, routed, mapping in (
            ("basic", basic, basic_mapping),
            ("weighted", weighted, refined.mapping),
        ):
            scheduled = schedule_routing_result(source, routed, mapping, graph)
            after = RoutingResult(scheduled.circuit, routed.final_mapping, [], routed.swap_count)
            records.append({
                "circuit_name": name, "source_sha256": source_hash,
                "method": method, "source_depth": source.depth(),
                "routed_depth": scheduled.depth_before,
                "scheduled_depth": scheduled.depth_after,
                "schedule_depth_delta": scheduled.depth_delta,
                "unit_duration_makespan": scheduled.makespan,
                "swaps_before": routed.circuit.count_ops().get("swap", 0),
                "swaps_after": scheduled.circuit.count_ops().get("swap", 0),
                "semantic_before": validate_small_unitary_equivalence(source, routed, mapping),
                "semantic_after": validate_small_unitary_equivalence(source, after, mapping),
            })
        sabre = run_sabre_baseline(source, graph, seed)
        scheduled = schedule_routed_circuit(sabre["circuit"], graph)
        records.append({
            "circuit_name": name, "source_sha256": source_hash,
            "method": "qiskit_sabre", "source_depth": source.depth(),
            "routed_depth": scheduled.depth_before,
            "scheduled_depth": scheduled.depth_after,
            "schedule_depth_delta": scheduled.depth_delta,
            "unit_duration_makespan": scheduled.makespan,
            "swaps_before": sabre["circuit"].count_ops().get("swap", 0),
            "swaps_after": scheduled.circuit.count_ops().get("swap", 0),
            "semantic_before": sabre["semantic_validation"],
            "semantic_after": validate_transpiled_small_unitary(
                source, scheduled.circuit, sabre["circuit"]),
        })
        oracle = solve_initial_mapping(build_interaction_graph(source), graph, 5.0)
        records.append({
            "circuit_name": name, "source_sha256": source_hash,
            "method": "cp_sat_mapping_oracle", "objective": oracle.objective,
            "status": oracle.status, "gap": oracle.gap,
            "routing_or_schedule_depth": None,
        })
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, check=False)
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex}"
    path = output / f"scheduler_check_{run_id}.json"
    payload = {"scope": "three four-qubit smoke circuits on an undirected four-site line; not a benchmark",
               "schema_version": 1, "seed": seed,
               "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
               "working_tree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
               "python_version": sys.version.split()[0], "qiskit_version": qiskit.__version__,
               "topology_edges": sorted([list(edge) for edge in graph.edges]),
               "gate_durations": "unit", "records": records}
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def main() -> None:
    print(run_scheduler_check())


if __name__ == "__main__":
    main()
