"""Statistical summaries must respect within-case seed collapse."""

import pytest

from qweave.experiments.statistics import analyze_records, bootstrap_median_interval, collapse_seeds


def _record(case, family, method, seed, depth, swaps):
    return {"case_id": case, "family": family, "topology": "path",
            "method": method, "seed": seed, "depth": depth,
            "swap_count": swaps, "runtime_seconds": 0.1 + 0.01 * (seed or 0),
            "hardware_makespan_ns": depth * 100.0,
            "estimated_log_success": -0.1 * swaps,
            "fallback_count": 0, "semantic_validation": True,
            "symbolic_validation": True, "directed_legal": method != "candidate"}


def test_seed_collapse_precedes_paired_analysis() -> None:
    records = [
        _record("a", "f1", "candidate", 1, 5, 2),
        _record("a", "f1", "candidate", 2, 9, 4),
        _record("a", "f1", "weighted", None, 8, 3),
        _record("b", "f2", "candidate", 1, 6, 1),
        _record("b", "f2", "candidate", 2, 6, 1),
        _record("b", "f2", "weighted", None, 7, 2),
    ]
    collapsed = collapse_seeds(records)
    candidate_a = next(row for row in collapsed
                       if row["method"] == "candidate" and row["case_id"] == "a")
    assert candidate_a["depth"] == 7
    analysis = analyze_records(records, references=("weighted",), resamples=200, seed=3)
    depth = next(row for row in analysis["paired_differences"]
                 if row["method"] == "candidate" and row["metric"] == "depth")
    assert depth["pairs"] == 2
    assert depth["median"] == -1.0
    candidate = next(row for row in analysis["method_summary"]
                     if row["method"] == "candidate")
    assert candidate["directed_legality_rate"] == 0


def test_unassessed_symbolic_validation_is_not_reported_as_failure() -> None:
    record = _record("a", "f1", "sabre", 1, 5, 2)
    record["symbolic_validation"] = None
    analysis = analyze_records([record], references=(), resamples=200, seed=3)
    summary = analysis["method_summary"][0]
    assert summary["symbolic_assessed_rate"] == 0
    assert summary["symbolic_pass_rate"] is None
    assert analysis["collapsed_records"][0]["symbolic_valid"] is None


def test_ablation_pairs_are_collapsed_by_case() -> None:
    records = []
    for case, gnn, mlp, untrained in (("a", 5, 6, 20), ("b", 7, 7, 18)):
        records.extend([
            _record(case, "f", "gnn_ppo", 1, gnn, 1),
            _record(case, "f", "gnn_ppo", 2, gnn + 2, 1),
            _record(case, "f", "mlp_ppo", 1, mlp, 1),
            _record(case, "f", "gnn_untrained", 1, untrained, 9),
        ])
    analysis = analyze_records(records, references=(), resamples=200, seed=3)
    depth = next(row for row in analysis["ablation_differences"]
                 if row["method"] == "gnn_ppo"
                 and row["reference"] == "mlp_ppo"
                 and row["metric"] == "depth")
    assert depth["pairs"] == 2
    assert depth["median"] == 0.5


def test_bootstrap_is_deterministic_and_validated() -> None:
    first = bootstrap_median_interval([-2, -1, 0, 1], resamples=500, seed=7)
    assert first == bootstrap_median_interval([-2, -1, 0, 1], resamples=500, seed=7)
    with pytest.raises(ValueError):
        bootstrap_median_interval([], resamples=500)
    with pytest.raises(ValueError):
        bootstrap_median_interval([1], resamples=10)
