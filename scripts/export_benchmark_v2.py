"""Package the reviewed benchmark-v2 record into tracked research artifacts."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
import statistics

import numpy as np

from qweave.experiments.statistics import analyze_records


METHOD_ORDER = ("basic", "weighted", "sabre", "gnn_ppo", "mlp_ppo", "gnn_untrained")


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _quartiles(values) -> dict:
    values = list(values)
    return {"q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75))}


def _format(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "unassessed"
    return f"{value:.{digits}f}"


def _find(rows: list[dict], method: str, reference: str, metric: str) -> dict:
    return next(row for row in rows if row["method"] == method
                and row["reference"] == reference and row["metric"] == metric)


def export(raw_path: Path, original_raw: Path, destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    analysis = analyze_records(payload["records"], resamples=5000, seed=20260921)
    destination.mkdir(parents=True)
    (destination / "checkpoints").mkdir()
    (destination / "figures").mkdir()
    shutil.copy2(raw_path, destination / "raw.json")
    shutil.copy2(original_raw, destination / "raw_pre_direction_correction.json")
    for checkpoint in sorted((raw_path.parent / "checkpoints").glob("*.pt")):
        shutil.copy2(checkpoint, destination / "checkpoints" / checkpoint.name)

    records = payload["records"]
    direction = []
    for method in METHOD_ORDER:
        rows = [row for row in records if row["method"] == method]
        direction.append({
            "method": method,
            "records": len(rows),
            "pre_synthesis_legality_rate": float(np.mean(
                [row["pre_synthesis_directed_legal"] for row in rows])),
            "post_synthesis_legality_rate": float(np.mean(
                [row["directed_legal"] for row in rows])),
            "depth_delta": _quartiles(
                row["directed_synthesis_depth_delta"] for row in rows),
            "operation_delta": _quartiles(
                row["directed_synthesis_operation_delta"] for row in rows),
        })
    unit_deltas = [row["unit_schedule_depth_delta"] for row in records]
    summary = {
        "benchmark_version": payload["benchmark_version"],
        "schema_version": "2.1-reviewed",
        "source_created_utc": payload["created_utc"],
        "manifest_sha256": payload["manifest_sha256"],
        "raw_sha256": _hash(raw_path),
        "raw_pre_direction_correction_sha256": _hash(original_raw),
        "configuration": payload["configuration"],
        "environment": payload["environment"],
        "selection": payload["selection"],
        "selected_candidate": payload["selected_candidate"],
        "training": payload["training"],
        "test_records": len(records),
        "test_cases": len({row["case_id"] for row in records}),
        "analysis": analysis,
        "unit_scheduling": {
            "all_depth_deltas_zero": all(value == 0 for value in unit_deltas),
            "depth_delta": _quartiles(unit_deltas),
        },
        "directed_synthesis": direction,
        "interpretation_limits": payload["interpretation_limits"],
        "correction": payload["directed_hardware_recomputation"],
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (destination / "learning_curves.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "method", "seed", "steps", "episodes", "mean_recent_return",
            "mean_recent_loss", "fallback_total"))
        writer.writeheader()
        for training in payload["training"]:
            for point in training["training_curve"]:
                writer.writerow({"method": training["method"], "seed": training["seed"], **point})

    method_summary = {row["method"]: row for row in analysis["method_summary"]}
    with (destination / "method_summary.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "cases", "seed_runs", "median_depth", "median_swaps",
                         "median_runtime_seconds", "semantic_pass_rate",
                         "directed_legality_rate", "fallback_total"))
        for method in METHOD_ORDER:
            row = method_summary[method]
            writer.writerow((method, row["cases"], row["seed_runs"], row["depth"]["median"],
                             row["swap_count"]["median"], row["runtime_seconds"]["median"],
                             row["semantic_pass_rate"], row["directed_legality_rate"],
                             row["fallback_total"]))

    paired = analysis["paired_differences"]
    ablation = analysis["ablation_differences"]
    report = [
        "# Benchmark v2 reviewed report", "",
        "Benchmark v2.0 was frozen before evaluation. The aggressive validation-only candidate "
        "was selected (score 19.5; conservative and standard each scored 24.0), then GNN-PPO "
        "and the no-message-passing PPO control were trained for 4,096 environment steps at "
        "seeds 7, 17, 29, 41, and 53. The untouched test split contains 36 cases from QFT and "
        "hardware-efficient families at widths 4, 6, and 8 on grid, chorded-ring, and idle-site "
        "hardware. The final dataset has 792 method/seed/case records.", "",
        "## Main result", "",
        "All 792 outputs passed the applicable numerical semantic check. All QWeave records "
        "also passed symbolic route replay; SABRE symbolic replay is unassessed because Qiskit "
        "does not expose QWeave's event trace. After explicit reverse-CNOT and SWAP lowering, "
        "all 792 hardware-cost circuits passed directed-coupler legality.", "",
        "The study does not support a superiority claim for GNN-PPO. Its paired median depth "
        "difference versus weighted routing was 0 gates with a 95% bootstrap interval [0, 0], "
        "and versus SABRE was 0 [0, 5]. The GNN and no-message-passing PPO control were also "
        "indistinguishable in paired median depth and SWAP count: both differences were 0 "
        "with intervals [0, 0]. PPO training did matter relative to the untrained graph control, "
        "but message passing did not show a benefit at this scale. Weighted routing matched or "
        "improved the aggregate learned results without fallbacks and was substantially faster.", "",
        "## Method-level test summary", "",
        "Seeds are collapsed within each case for depth and SWAP quartiles. Runtime quartiles use "
        "all measured method/seed records.", "",
        "| Method | Cases | Depth median [IQR] | SWAP median [IQR] | Runtime median [IQR] s | Semantic | Directed | Fallbacks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"basic": "Basic", "weighted": "Weighted", "sabre": "SABRE",
              "gnn_ppo": "GNN-PPO", "mlp_ppo": "No-message PPO",
              "gnn_untrained": "Untrained GNN"}
    for method in METHOD_ORDER:
        row = method_summary[method]
        report.append(
            f"| {labels[method]} | {row['cases']} | "
            f"{_format(row['depth']['median'], 1)} [{_format(row['depth']['q25'], 1)}, {_format(row['depth']['q75'], 1)}] | "
            f"{_format(row['swap_count']['median'], 1)} [{_format(row['swap_count']['q25'], 1)}, {_format(row['swap_count']['q75'], 1)}] | "
            f"{_format(row['runtime_seconds']['median'], 4)} [{_format(row['runtime_seconds']['q25'], 4)}, {_format(row['runtime_seconds']['q75'], 4)}] | "
            f"{_format(row['semantic_pass_rate'], 2)} | {_format(row['directed_legality_rate'], 2)} | "
            f"{row['fallback_total']} |")
    report += ["", "## Paired comparisons", "",
               "A negative depth/SWAP/makespan difference favors the first method. A positive "
               "estimated-log-success difference favors the first method.", "",
               "| Comparison | Metric | Paired median | 95% bootstrap interval | Cases |",
               "|---|---|---:|---:|---:|"]
    comparisons = [
        (paired, "gnn_ppo", "weighted", "depth"),
        (paired, "gnn_ppo", "sabre", "depth"),
        (paired, "gnn_ppo", "weighted", "swap_count"),
        (paired, "gnn_ppo", "weighted", "hardware_makespan_ns"),
        (ablation, "gnn_ppo", "mlp_ppo", "depth"),
        (ablation, "gnn_ppo", "mlp_ppo", "swap_count"),
        (ablation, "gnn_ppo", "gnn_untrained", "depth"),
        (ablation, "gnn_ppo", "gnn_untrained", "swap_count"),
    ]
    for rows, method, reference, metric in comparisons:
        row = _find(rows, method, reference, metric)
        report.append(
            f"| {labels[method]} - {labels[reference]} | {metric} | "
            f"{_format(row['median'], 2)} | [{_format(row['ci95_low'], 2)}, "
            f"{_format(row['ci95_high'], 2)}] | {row['pairs']} |")
    report += ["", "## Family behavior and counterexamples", "",
               "On hardware-efficient circuits, weighted, GNN-PPO, and no-message PPO all had "
               "median depth 27 and median SWAP count 0. On QFT, weighted routing had median "
               "depth 71.5 and 16.5 SWAPs, GNN-PPO had 83.5 and 41, and no-message PPO had "
               "90 and 31.5. The strongest GNN-PPO depth improvement over weighted routing was "
               "8 gates on `qft_i1_6q_chorded_ring`; the largest regression was 89 gates and "
               "130 additional SWAPs on `qft_i1_8q_grid2`.", "",
               "The trained GNN incurred 4,543 test fallbacks, compared with 742 for the "
               "no-message PPO control and zero for deterministic methods. Seed 29 was a clear "
               "instability case: its GNN training curve ended with 212 fallbacks and a recent "
               "mean return near -103. Learning curves and checkpoints are included with this report.", "",
               "## Scheduling and directed costs", "",
               "Every unit-duration fixed-route scheduling depth delta was zero. This is a "
               "within-route result and does not imply routing improvement. Before direction "
               "synthesis, most routed circuits were illegal on the one-way couplers. The reviewed "
               "cost pass therefore lowers reverse CNOTs by Hadamard conjugation and decomposes "
               "SWAPs into three direction-lowered CNOTs. All lowered circuits pass ordered-arc "
               "validation. Median added depth was 32 for weighted routing, 34 for SABRE, 37.5 "
               "for GNN-PPO, 28.5 for no-message PPO, 57 for Basic, and 441 for the untrained GNN.", "",
               "Gate durations and edge-error values are declared synthetic proxies. They are "
               "not calibration data and cannot establish hardware fidelity. The corrected paired "
               "GNN-PPO minus weighted makespan difference was 0 ns with interval [0, 1113], "
               "so the proxy-cost analysis also gives no learned advantage.", "",
               "## Reproducibility", "",
               f"- Frozen manifest SHA-256: `{payload['manifest_sha256']}`",
               f"- Corrected raw SHA-256: `{_hash(raw_path)}`",
               f"- Pre-correction raw SHA-256: `{_hash(original_raw)}`",
               f"- Training/evaluation commit: `{payload['environment']['git']['commit']}` (clean)",
               f"- Directed-cost correction commit: `{payload['directed_hardware_recomputation']['postprocess_git']['commit']}` (clean)",
               "- Python 3.11.15; Qiskit 2.5.2; 5,000 bootstrap resamples; bootstrap seed 20260921.", "",
               "The source raw record is retained because the initial audit found that its "
               "duration calculation scheduled pre-synthesis circuits on an undirected graph. "
               "No routing, policy, split, or non-runtime route metric changed during correction.", "",
               "## Claim boundary", "",
               "This synthetic, finite study supports three bounded conclusions: the tested "
               "outputs passed the declared semantic checks; PPO training strongly improved on "
               "the untrained policy; and the tested GNN message passing did not improve paired "
               "median outcomes over the no-message control. It does not support a state-of-the-art, "
               "hardware-fidelity, optimal-routing, or general superiority claim.", ""]
    report_path = destination / "REPORT.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    manifest_lines = []
    for file in sorted(path for path in destination.rglob("*") if path.is_file()):
        if file.name == "SHA256SUMS.txt":
            continue
        manifest_lines.append(f"{_hash(file)}  {file.relative_to(destination).as_posix()}")
    (destination / "SHA256SUMS.txt").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("original_raw", type=Path)
    parser.add_argument("--destination", type=Path,
                        default=Path("results/benchmark_v2"))
    args = parser.parse_args()
    print(export(args.raw, args.original_raw, args.destination))


if __name__ == "__main__":
    main()
