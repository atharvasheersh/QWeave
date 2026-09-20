"""A tiny subset run protects study separation and artifact schema."""

import json

from qweave.experiments.benchmark_v2_dataset import benchmark_v2_manifest
from qweave.experiments.study_v2 import run_study_v2


def test_subset_study_selects_on_validation_and_writes_checkpoints(tmp_path) -> None:
    manifest = benchmark_v2_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validation = next(case["case_id"] for case in manifest["cases"]
                      if case["split"] == "validation" and case["logical_qubits"] == 4)
    test = next(case["case_id"] for case in manifest["cases"]
                if case["split"] == "test" and case["logical_qubits"] == 4)
    raw = run_study_v2(tmp_path / "runs", manifest_path=manifest_path,
                       selection_steps=4, final_steps=4,
                       policy_seeds=(3,), sabre_seeds=(3,), runtime_repeats=1,
                       bootstrap_resamples=200,
                       validation_case_ids={validation}, test_case_ids={test})
    payload = json.loads(raw.read_text(encoding="utf-8"))
    assert payload["scope"] == "explicit subset harness verification"
    assert {item["candidate"] for item in payload["selection"]} == {
        "standard", "conservative", "aggressive"}
    assert payload["selected_candidate"] in {
        "standard", "conservative", "aggressive"}
    assert {row["method"] for row in payload["records"]} == {
        "gnn_ppo", "mlp_ppo", "gnn_untrained", "basic", "weighted", "sabre"}
    assert all(row["semantic_validation"] is True for row in payload["records"])
    assert len(payload["training"]) == 2
    for item in payload["training"]:
        checkpoint = raw.parent / item["checkpoint"]
        assert checkpoint.exists() and checkpoint.stat().st_size > 0
    assert payload["analysis"]["paired_differences"]
