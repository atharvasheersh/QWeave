"""Recompute benchmark-v2 hardware fields after directed-gate synthesis."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import torch

from qweave.baselines.sabre import run_sabre_baseline
from qweave.core.compiler import compile_deterministic
from qweave.learning import GraphActorCritic, load_checkpoint
from qweave.metrics.hardware_costs import (
    estimate_hardware_cost,
    synthesize_directed_circuit,
    validate_coupling_legality,
)

from .benchmark_v2_dataset import directed_v2_graph, load_v2_case
from .statistics import analyze_records
from .study_v2 import _git_metadata, _policy_episode, _profile


def _model(method: str, seed: int, run_dir: Path,
           cache: dict[tuple[str, int], GraphActorCritic]) -> GraphActorCritic:
    key = (method, seed)
    if key not in cache:
        if method in {"gnn_ppo", "mlp_ppo"}:
            checkpoint = run_dir / "checkpoints" / f"{method}_seed{seed}.pt"
            cache[key] = load_checkpoint(str(checkpoint))[0]
        elif method == "gnn_untrained":
            torch.manual_seed(seed)
            model = GraphActorCritic(hidden=64, message_passing_steps=2)
            model.eval()
            cache[key] = model
        else:
            raise ValueError(f"method {method} does not use a policy model")
    return cache[key]


def _compiled_circuit(record: dict, case: dict, run_dir: Path,
                      cache: dict[tuple[str, int], GraphActorCritic]):
    method = record["method"]
    source, graph = load_v2_case(case)
    if method in {"gnn_ppo", "mlp_ppo", "gnn_untrained"}:
        seed = int(record["seed"])
        model = _model(method, seed, run_dir, cache)
        return _policy_episode(case, model, seed)[4].circuit
    if method in {"basic", "weighted"}:
        return compile_deterministic(
            source, graph, method=method, numerical_validation=False).routed.circuit
    if method == "sabre":
        return run_sabre_baseline(
            source, graph, int(record["seed"]), numerical_validation=False)["circuit"]
    raise ValueError(f"unsupported study method {method}")


def _direction_fields(compiled, case: dict) -> dict:
    directed = directed_v2_graph(case)
    try:
        validate_coupling_legality(compiled, directed)
        pre_legal, pre_error = True, None
    except ValueError as exc:
        pre_legal, pre_error = False, str(exc)
    lowered = synthesize_directed_circuit(compiled, directed)
    validate_coupling_legality(lowered, directed)
    cost = estimate_hardware_cost(lowered, directed, _profile(case))
    return {
        "hardware_makespan_ns": cost["scheduled_makespan_ns"],
        "hardware_scheduled_depth": cost["scheduled_depth"],
        "hardware_schedule_depth_delta": cost["schedule_depth_delta"],
        "estimated_success_probability": cost["estimated_success_probability"],
        "estimated_log_success": cost["estimated_log_success"],
        "hardware_profile": cost["profile"],
        "hardware_profile_calibrated": cost["calibrated"],
        "pre_synthesis_directed_legal": pre_legal,
        "pre_synthesis_directed_error": pre_error,
        "directed_synthesis_depth": lowered.depth(),
        "directed_synthesis_depth_delta": lowered.depth() - compiled.depth(),
        "directed_synthesis_operation_delta": len(lowered.data) - len(compiled.data),
        "directed_legal": True,
        "directed_error": None,
    }


def recompute_directed_hardware(raw_path: str | Path, *,
                                manifest_path: str | Path =
                                "benchmarks/benchmark_v2_manifest.json",
                                bootstrap_resamples: int = 5000) -> Path:
    """Replay saved policies and baselines, then write corrected immutable JSON."""

    raw_path = Path(raw_path)
    run_dir = raw_path.parent
    output_path = run_dir / "raw_directed.json"
    summary_path = run_dir / "summary_directed.json"
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("directed recomputation output already exists")
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in manifest["cases"]}
    cache: dict[tuple[str, int], GraphActorCritic] = {}
    total = len(payload["records"])
    for index, record in enumerate(payload["records"], start=1):
        case = cases[record["case_id"]]
        compiled = _compiled_circuit(record, case, run_dir, cache)
        record.update(_direction_fields(compiled, case))
        if index == 1 or index % 25 == 0 or index == total:
            print(f"directed hardware {index}/{total}", flush=True)
    payload["schema_version"] = "2.1"
    payload["analysis"] = analyze_records(
        payload["records"], resamples=bootstrap_resamples, seed=20260921)
    payload["directed_hardware_recomputation"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_raw": raw_path.name,
        "source_raw_sha256": sha256(raw_path.read_bytes()).hexdigest(),
        "postprocess_git": _git_metadata(),
        "method": (
            "Reverse CNOTs use H-H conjugation around the available CNOT; "
            "SWAPs use three direction-lowered CNOTs. Costs and schedules are "
            "computed only after the resulting circuit passes directed legality."),
    }
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    summary = {key: payload[key] for key in (
        "schema_version", "benchmark_version", "created_utc", "scope",
        "configuration", "environment", "selected_candidate", "analysis",
        "interpretation_limits", "directed_hardware_recomputation")}
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return output_path
