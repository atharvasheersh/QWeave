"""Small deterministic experiments producing timestamped JSON and CSV files."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from qiskit import QuantumCircuit
import networkx as nx

from qweave.baselines.basic import run_basic_baseline
from qweave.baselines.sabre import run_sabre_baseline
from qweave.core.validation import validate_two_qubit_legality
from qweave.mapper.initial_mapper import initial_mapping
from qweave.mapper.interaction_graph import build_interaction_graph
from qweave.mapper.local_search import refine_mapping
from qweave.metrics.circuit_metrics import circuit_metrics, timed
from qweave.oracle.cp_sat import solve_initial_mapping
from qweave.routing.fallback_router import route_with_fallback


def sample_circuits() -> dict[str, QuantumCircuit]:
    chain = QuantumCircuit(4)
    for index in range(3):
        chain.cx(index, index + 1)
    star = QuantumCircuit(4)
    for index in range(1, 4):
        star.cx(0, index)
    random_like = QuantumCircuit(4)
    for first, second in [(0, 2), (1, 3), (0, 1), (2, 3), (0, 3)]:
        random_like.cx(first, second)
    return {"cnot_chain": chain, "star": star, "random_like": random_like}


def run_smoke(output_dir: str | Path = "results", seed: int = 7) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    graph = nx.path_graph(4)
    records = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for name, circuit in sample_circuits().items():
        base, base_runtime = timed(run_basic_baseline, circuit, graph)
        validate_two_qubit_legality(base.circuit, graph)
        mapped = initial_mapping(circuit, graph)
        refined = refine_mapping(circuit, mapped.mapping, graph)
        routed, weighted_runtime = timed(route_with_fallback, circuit, graph, refined.mapping)
        validate_two_qubit_legality(routed.circuit, graph)
        sabre = run_sabre_baseline(circuit, graph, seed)
        record = {
            "timestamp": timestamp, "seed": seed, "circuit_name": name,
            "logical_qubits": circuit.num_qubits, "topology": "line4",
            "basic": {**circuit_metrics(base.circuit, circuit, base_runtime), "swap_count": base.swap_count},
            "weighted": {**circuit_metrics(routed.circuit, circuit, weighted_runtime), "swap_count": routed.swap_count, "mapping": refined.mapping},
            "sabre": {key: value for key, value in sabre.items() if key != "circuit"},
            "validation": True,
        }
        if circuit.num_qubits <= 6:
            record["cp_sat"] = solve_initial_mapping(build_interaction_graph(circuit), graph, 5.0).__dict__
        records.append(record)
    run_id = f"{timestamp}_{uuid4().hex}"
    json_path = output / f"smoke_{run_id}.json"
    csv_path = output / f"smoke_{run_id}.csv"
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, default=str)
    rows = [{"circuit_name": item["circuit_name"], "topology": item["topology"], "mapper": mapper, **metrics} for item in records for mapper, metrics in item.items() if mapper in {"basic", "weighted", "sabre"}]
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    return json_path


def main() -> None:
    print(run_smoke())


if __name__ == "__main__":
    main()

