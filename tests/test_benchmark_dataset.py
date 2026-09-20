"""The final benchmark split and harness must remain frozen and auditable."""

import json

from qweave.experiments.benchmark_dataset import benchmark_manifest, load_case
from qweave.experiments.learned_benchmark import run_benchmark


def test_benchmark_manifest_is_deterministic_and_disjoint() -> None:
    first = benchmark_manifest()
    second = benchmark_manifest()
    assert first == second
    assert len(first["cases"]) == 30
    split_ids = {name: {case["case_id"] for case in first["cases"]
                        if case["split"] == name}
                 for name in ("train", "validation", "test")}
    assert {name: len(ids) for name, ids in split_ids.items()} == {
        "train": 12, "validation": 14, "test": 4}
    assert not (split_ids["train"] & split_ids["validation"]
                or split_ids["train"] & split_ids["test"]
                or split_ids["validation"] & split_ids["test"])
    for case in first["cases"]:
        circuit, graph = load_case(case)
        assert circuit.num_qubits == case["logical_qubits"]
        assert graph.number_of_nodes() == case["logical_qubits"]


def test_small_harness_run_has_all_controls_and_valid_outputs(tmp_path) -> None:
    case_id = "star_4q_grid2"
    raw_path = run_benchmark(tmp_path, training_steps=4, policy_seeds=(3,),
                             sabre_seeds=(3,), runtime_repeats=1,
                             test_case_ids={case_id})
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    assert payload["scope"] == "explicit test subset for harness verification"
    assert {row["method"] for row in payload["records"]} == {
        "basic", "weighted", "sabre", "gnn_ppo", "mlp_ppo", "gnn_untrained"}
    assert all(row["case_id"] == case_id and row["status"] == "ok"
               for row in payload["records"])
    assert all(row["semantic_validation"] is True for row in payload["records"])
    assert len(payload["summary"]) == 6
