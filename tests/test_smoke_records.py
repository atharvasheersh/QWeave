"""Regression check that a new smoke run cannot replace earlier raw data."""

import json

from qweave.experiments import smoke


def test_two_runs_preserve_the_first_json_and_csv(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(smoke, "sample_circuits", lambda: {})
    first = smoke.run_smoke(tmp_path)
    first_json = first.read_bytes()
    first_csv = first.with_suffix(".csv").read_bytes()

    second = smoke.run_smoke(tmp_path)
    assert first != second
    assert first.read_bytes() == first_json
    assert first.with_suffix(".csv").read_bytes() == first_csv
    assert second.with_suffix(".csv").exists()
    assert json.loads(first_json) == []
