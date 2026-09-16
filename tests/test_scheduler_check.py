"""Regression checks for stage attribution and immutable check records."""

import json

from qiskit import QuantumCircuit

from qweave.experiments import scheduler_check


def test_fixed_route_record_separates_scheduler_and_mapping_oracle(tmp_path, monkeypatch) -> None:
    source = QuantumCircuit(4)
    source.cx(0, 3)
    monkeypatch.setattr(scheduler_check, "sample_circuits", lambda: {"tiny": source})
    first = scheduler_check.run_scheduler_check(tmp_path, seed=5)
    first_bytes = first.read_bytes()
    second = scheduler_check.run_scheduler_check(tmp_path, seed=5)
    assert first != second
    assert first.read_bytes() == first_bytes
    records = json.loads(first_bytes)["records"]
    stages = [record for record in records if record["method"] != "cp_sat_mapping_oracle"]
    assert {record["method"] for record in stages} == {"basic", "weighted", "qiskit_sabre"}
    assert all(record["schedule_depth_delta"] ==
               record["scheduled_depth"] - record["routed_depth"] for record in stages)
    assert all(record["swaps_before"] == record["swaps_after"] for record in stages)
    assert all(record["semantic_after"] is True for record in stages)
    oracle = next(record for record in records if record["method"] == "cp_sat_mapping_oracle")
    assert oracle["routing_or_schedule_depth"] is None
    assert "scheduled_depth" not in oracle
