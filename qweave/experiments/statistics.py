"""Seed-aware descriptive statistics for paired routing experiments."""

from collections import defaultdict

import numpy as np


DEFAULT_METRICS = (
    "depth", "swap_count", "runtime_seconds", "hardware_makespan_ns",
    "estimated_log_success", "fallback_count",
)


def collapse_seeds(records: list[dict], metrics=DEFAULT_METRICS) -> list[dict]:
    """Collapse repeat seeds within each method/case before case analysis."""

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["method"], record["case_id"])].append(record)
    collapsed = []
    for (method, case_id), rows in sorted(grouped.items()):
        output = {"method": method, "case_id": case_id,
                  "family": rows[0]["family"], "topology": rows[0]["topology"],
                  "seed_runs": len(rows)}
        for metric in metrics:
            values = [row[metric] for row in rows if row.get(metric) is not None]
            output[metric] = float(np.median(values)) if values else None
        semantic = [row.get("semantic_validation") for row in rows
                    if row.get("semantic_validation") is not None]
        symbolic = [row.get("symbolic_validation") for row in rows
                    if row.get("symbolic_validation") is not None]
        output["semantic_valid"] = all(value is True for value in semantic) if semantic else None
        output["symbolic_valid"] = all(value is True for value in symbolic) if symbolic else None
        output["directed_legal_rate"] = float(np.mean(
            [bool(row.get("directed_legal")) for row in rows]))
        collapsed.append(output)
    return collapsed


def _quartiles(values: list[float]) -> dict:
    return {"q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75))}


def bootstrap_median_interval(values: list[float], *, resamples: int = 5000,
                              seed: int = 20260921) -> dict:
    if not values:
        raise ValueError("bootstrap requires at least one paired value")
    if type(resamples) is not int or resamples < 100:
        raise ValueError("bootstrap resamples must be an integer of at least 100")
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(resamples, len(data)))
    estimates = np.median(data[indices], axis=1)
    return {"median": float(np.median(data)),
            "ci95_low": float(np.quantile(estimates, 0.025)),
            "ci95_high": float(np.quantile(estimates, 0.975)),
            "pairs": len(data), "resamples": resamples, "seed": seed}


def analyze_records(records: list[dict], *, references=("weighted", "sabre"),
                    resamples: int = 5000, seed: int = 20260921) -> dict:
    """Return family summaries, paired intervals, rates, and counterexamples."""

    collapsed = collapse_seeds(records)
    family_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in collapsed:
        family_groups[(row["method"], row["family"])].append(row)
    family_summary = []
    for (method, family), rows in sorted(family_groups.items()):
        entry = {"method": method, "family": family, "cases": len(rows)}
        for metric in ("depth", "swap_count", "hardware_makespan_ns",
                       "estimated_log_success", "fallback_count"):
            values = [row[metric] for row in rows if row[metric] is not None]
            entry[metric] = _quartiles(values) if values else None
        family_summary.append(entry)

    by_key = {(row["method"], row["case_id"]): row for row in collapsed}
    methods = sorted({row["method"] for row in collapsed})
    paired = []
    counterexamples = []
    for method in methods:
        for reference in references:
            if method == reference:
                continue
            common = sorted(case_id for m, case_id in by_key
                            if m == method and (reference, case_id) in by_key)
            if not common:
                continue
            for metric in ("depth", "swap_count", "hardware_makespan_ns",
                           "estimated_log_success"):
                deltas = [by_key[(method, case_id)][metric]
                          - by_key[(reference, case_id)][metric]
                          for case_id in common]
                paired.append({"method": method, "reference": reference,
                               "metric": metric,
                               **bootstrap_median_interval(
                                   deltas, resamples=resamples, seed=seed)})
            depth_rows = [{"case_id": case_id,
                           "family": by_key[(method, case_id)]["family"],
                           "topology": by_key[(method, case_id)]["topology"],
                           "depth_delta": (by_key[(method, case_id)]["depth"]
                                           - by_key[(reference, case_id)]["depth"]),
                           "swap_delta": (by_key[(method, case_id)]["swap_count"]
                                          - by_key[(reference, case_id)]["swap_count"])}
                          for case_id in common]
            counterexamples.append({"method": method, "reference": reference,
                                    "best": sorted(depth_rows, key=lambda row: (row["depth_delta"], row["case_id"]))[:5],
                                    "worst": sorted(depth_rows, key=lambda row: (-row["depth_delta"], row["case_id"]))[:5]})

    method_summary = []
    for method in methods:
        raw = [row for row in records if row["method"] == method]
        cases = [row for row in collapsed if row["method"] == method]
        runtimes = [row["runtime_seconds"] for row in raw]
        semantic = [row.get("semantic_validation") for row in raw
                    if row.get("semantic_validation") is not None]
        symbolic = [row.get("symbolic_validation") for row in raw
                    if row.get("symbolic_validation") is not None]
        method_summary.append({
            "method": method, "cases": len(cases), "seed_runs": len(raw),
            "depth": _quartiles([row["depth"] for row in cases]),
            "swap_count": _quartiles([row["swap_count"] for row in cases]),
            "runtime_seconds": _quartiles(runtimes),
            "semantic_assessed_rate": len(semantic) / len(raw),
            "semantic_pass_rate": (float(np.mean([value is True for value in semantic]))
                                   if semantic else None),
            "symbolic_assessed_rate": len(symbolic) / len(raw),
            "symbolic_pass_rate": (float(np.mean([value is True for value in symbolic]))
                                   if symbolic else None),
            "directed_legality_rate": float(np.mean([bool(row.get("directed_legal")) for row in raw])),
            "fallback_total": int(sum(row.get("fallback_count", 0) for row in raw)),
        })
    return {"collapsed_records": collapsed, "method_summary": method_summary,
            "family_summary": family_summary, "paired_differences": paired,
            "counterexamples": counterexamples,
            "bootstrap": {"resamples": resamples, "seed": seed}}
