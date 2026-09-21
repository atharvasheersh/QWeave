"""Frozen benchmark-v2 training, ablation, cost, and analysis study."""

from datetime import datetime, timezone
import importlib.metadata
from hashlib import sha256
import json
import platform
from pathlib import Path
import subprocess
import time
from uuid import uuid4

import numpy as np
import torch

from qweave.baselines.sabre import run_sabre_baseline
from qweave.core.compiler import compile_deterministic
from qweave.core.validation import (
    validate_small_unitary_equivalence,
    validate_statevector_probes,
)
from qweave.learning import (
    GraphActorCritic,
    PPOConfig,
    QubitRoutingEnv,
    edge_index_from_env,
    save_checkpoint,
    train_ppo,
)
from qweave.mapper.initial_mapper import initial_mapping
from qweave.mapper.local_search import refine_mapping
from qweave.metrics.circuit_metrics import circuit_metrics, timed
from qweave.metrics.hardware_costs import (
    HardwareCostProfile,
    estimate_hardware_cost,
    synthesize_directed_circuit,
    validate_coupling_legality,
)
from qweave.scheduling import schedule_routed_circuit, schedule_routing_result

from .benchmark_v2_dataset import (
    BENCHMARK_VERSION,
    benchmark_v2_manifest,
    directed_v2_graph,
    load_v2_case,
)
from .statistics import analyze_records


SCHEMA_VERSION = "2.0"
POLICY_SEEDS = (7, 17, 29, 41, 53)
SABRE_SEEDS = (7, 17, 29, 41, 53)
SELECTION_SEED = 101
SELECTION_STEPS = 1024
FINAL_STEPS = 4096
SELECTION_CANDIDATES = {
    "aggressive": {"learning_rate": 5e-4, "entropy_coefficient": 0.005},
    "conservative": {"learning_rate": 1e-4, "entropy_coefficient": 0.02},
    "standard": {"learning_rate": 3e-4, "entropy_coefficient": 0.01},
}


def _git_metadata() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _versions() -> dict:
    packages = ("qiskit", "networkx", "numpy", "gymnasium", "torch", "ortools")
    return {name: importlib.metadata.version(name) for name in packages}


def _mapping(case: dict) -> dict[int, int]:
    circuit, graph = load_v2_case(case)
    seeded = initial_mapping(circuit, graph)
    return refine_mapping(circuit, seeded.mapping, graph).mapping


def _factory(case: dict):
    frozen = json.loads(json.dumps(case))
    mapping = _mapping(frozen)

    def factory() -> QubitRoutingEnv:
        circuit, graph = load_v2_case(frozen)
        return QubitRoutingEnv(circuit, graph, dict(mapping))

    return factory


def _profile(case: dict) -> HardwareCostProfile:
    errors = {}
    for item in case["edge_error_rates"]:
        left, right = item["edge"]
        errors[(left, right)] = item["error"]
        errors[(right, left)] = item["error"] + 0.001
    return HardwareCostProfile(directed_edge_errors=errors, calibrated=False)


def _directed_legal(circuit, case: dict) -> tuple[bool, str | None]:
    try:
        validate_coupling_legality(circuit, directed_v2_graph(case))
        return True, None
    except ValueError as exc:
        return False, str(exc)


def _semantic(source, routed, mapping) -> bool:
    result = validate_small_unitary_equivalence(source, routed, mapping)
    if result is None:
        result = validate_statevector_probes(source, routed, mapping, probes=2)
    return result is True


def _common_record(case: dict, method: str, seed: int | None, source,
                   compiled, runtime: float, runtime_samples: list[float],
                   semantic: bool, symbolic: bool | None,
                   fallback_count: int) -> dict:
    directed_graph = directed_v2_graph(case)
    pre_synthesis_legal, pre_synthesis_error = _directed_legal(compiled, case)
    hardware_circuit = synthesize_directed_circuit(compiled, directed_graph)
    cost = estimate_hardware_cost(hardware_circuit, directed_graph, _profile(case))
    directed_legal, directed_error = _directed_legal(hardware_circuit, case)
    return {
        "case_id": case["case_id"], "family": case["family"],
        "instance": case["instance"], "topology": case["topology"],
        "logical_qubits": case["logical_qubits"],
        "physical_qubits": case["physical_qubits"],
        "method": method, "seed": seed,
        **circuit_metrics(compiled, source, runtime),
        "runtime_samples_seconds": runtime_samples,
        "hardware_makespan_ns": cost["scheduled_makespan_ns"],
        "hardware_scheduled_depth": cost["scheduled_depth"],
        "hardware_schedule_depth_delta": cost["schedule_depth_delta"],
        "estimated_success_probability": cost["estimated_success_probability"],
        "estimated_log_success": cost["estimated_log_success"],
        "hardware_profile": cost["profile"],
        "hardware_profile_calibrated": cost["calibrated"],
        "pre_synthesis_directed_legal": pre_synthesis_legal,
        "pre_synthesis_directed_error": pre_synthesis_error,
        "directed_synthesis_depth": hardware_circuit.depth(),
        "directed_synthesis_depth_delta": hardware_circuit.depth() - compiled.depth(),
        "directed_synthesis_operation_delta": len(hardware_circuit.data) - len(compiled.data),
        "semantic_validation": semantic,
        "symbolic_validation": symbolic,
        "directed_legal": directed_legal,
        "directed_error": directed_error,
        "fallback_count": fallback_count,
        "status": "ok",
    }


def _policy_episode(case: dict, model: GraphActorCritic, seed: int):
    source, graph = load_v2_case(case)
    mapping = _mapping(case)
    env = QubitRoutingEnv(source, graph, mapping)
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    started = time.perf_counter()
    while not (terminated or truncated):
        action, _, _ = model.act(observation, edge_index_from_env(env), deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
    runtime = time.perf_counter() - started
    return source, graph, mapping, env, env.routing_result(), runtime


def _policy_record(case: dict, model: GraphActorCritic, method: str, seed: int,
                   runtime_repeats: int) -> dict:
    runs = [_policy_episode(case, model, seed) for _ in range(runtime_repeats + 1)]
    source, _, mapping, env, routed, _ = runs[-1]
    samples = [item[5] for item in runs[1:]]
    record = _common_record(
        case, method, seed, source, routed.circuit, float(np.median(samples)),
        samples, _semantic(source, routed, mapping), True, env.fallback_count)
    unit_schedule = schedule_routing_result(source, routed, mapping, load_v2_case(case)[1])
    record.update({"unit_scheduled_depth": unit_schedule.depth_after,
                   "unit_schedule_depth_delta": unit_schedule.depth_delta,
                   "unit_scheduled_swap_count": circuit_metrics(unit_schedule.circuit)["swap_count"]})
    return record


def _deterministic_record(case: dict, method: str, runtime_repeats: int) -> dict:
    source, graph = load_v2_case(case)
    validated = compile_deterministic(source, graph, method=method, schedule=True)
    timing = [timed(compile_deterministic, source, graph, method, False, None, False)[1]
              for _ in range(runtime_repeats + 1)]
    samples = timing[1:]
    record = _common_record(
        case, method, None, source, validated.routed.circuit,
        float(np.median(samples)), samples,
        validated.semantic_validation is True, True, 0)
    record.update({"unit_scheduled_depth": validated.scheduled.depth_after,
                   "unit_schedule_depth_delta": validated.scheduled.depth_delta,
                   "unit_scheduled_swap_count": circuit_metrics(validated.scheduled.circuit)["swap_count"]})
    return record


def _sabre_record(case: dict, seed: int, runtime_repeats: int) -> dict:
    source, graph = load_v2_case(case)
    validated = run_sabre_baseline(source, graph, seed, numerical_validation=True)
    timing_runs = [run_sabre_baseline(source, graph, seed, numerical_validation=False)
                   for _ in range(runtime_repeats + 1)]
    samples = [item["runtime_seconds"] for item in timing_runs[1:]]
    compiled = validated["circuit"]
    schedule = schedule_routed_circuit(compiled, graph)
    record = _common_record(
        case, "sabre", seed, source, compiled, float(np.median(samples)), samples,
        validated["semantic_validation"] is True, None, 0)
    record.update({"unit_scheduled_depth": schedule.depth_after,
                   "unit_schedule_depth_delta": schedule.depth_delta,
                   "unit_scheduled_swap_count": circuit_metrics(schedule.circuit)["swap_count"]})
    return record


def _compact_training(method: str, seed: int, metrics: dict,
                      checkpoint: str | None = None) -> dict:
    returns = metrics["episode_returns"]
    return {"method": method, "seed": seed, "config": metrics["config"],
            "steps": metrics["steps"], "episodes": metrics["episodes"],
            "return_median": float(np.median(returns)) if returns else None,
            "return_mean": float(np.mean(returns)) if returns else None,
            "fallback_total": int(sum(metrics["fallback_counts"])),
            "training_curve": metrics["training_curve"],
            "checkpoint": checkpoint}


def _validation_score(model: GraphActorCritic, cases: list[dict]) -> tuple[float, dict]:
    values, depths, swaps, fallbacks = [], [], [], []
    for case in cases:
        source, _, _, env, routed, _ = _policy_episode(case, model, SELECTION_SEED)
        depth, swap, fallback = routed.circuit.depth(), routed.swap_count, env.fallback_count
        values.append(depth + 0.5 * swap + 5 * fallback)
        depths.append(depth)
        swaps.append(swap)
        fallbacks.append(fallback)
    return float(np.median(values)), {
        "cases": len(cases), "median_depth": float(np.median(depths)),
        "median_swaps": float(np.median(swaps)),
        "fallback_total": int(sum(fallbacks)),
    }


def _base_config(steps: int, seed: int, candidate: dict,
                 message_passing_steps: int = 2) -> PPOConfig:
    return PPOConfig(total_steps=steps, rollout_steps=min(128, steps),
                     update_epochs=4, minibatch_size=min(32, steps),
                     learning_rate=candidate["learning_rate"],
                     entropy_coefficient=candidate["entropy_coefficient"],
                     seed=seed, hidden=64,
                     message_passing_steps=message_passing_steps)


def run_study_v2(output_root: str | Path = "run_artifacts", *,
                 manifest_path: str | Path = "benchmarks/benchmark_v2_manifest.json",
                 selection_steps: int = SELECTION_STEPS,
                 final_steps: int = FINAL_STEPS,
                 policy_seeds: tuple[int, ...] = POLICY_SEEDS,
                 sabre_seeds: tuple[int, ...] = SABRE_SEEDS,
                 runtime_repeats: int = 3,
                 bootstrap_resamples: int = 5000,
                 validation_case_ids: set[str] | None = None,
                 test_case_ids: set[str] | None = None) -> Path:
    """Run selection, final training, ablations, test evaluation, and analysis."""

    frozen_manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    generated_manifest = benchmark_v2_manifest()
    if frozen_manifest != generated_manifest:
        raise ValueError("generated benchmark v2 does not match the frozen manifest")
    train_cases = [case for case in frozen_manifest["cases"] if case["split"] == "train"]
    validation_cases = [case for case in frozen_manifest["cases"] if case["split"] == "validation"]
    test_cases = [case for case in frozen_manifest["cases"] if case["split"] == "test"]
    if validation_case_ids is not None:
        validation_cases = [case for case in validation_cases
                            if case["case_id"] in validation_case_ids]
    if test_case_ids is not None:
        test_cases = [case for case in test_cases if case["case_id"] in test_case_ids]
    if not validation_cases or not test_cases:
        raise ValueError("study requires at least one validation and test case")
    if type(runtime_repeats) is not int or runtime_repeats < 1:
        raise ValueError("runtime_repeats must be a positive integer")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex
    run_dir = Path(output_root) / f"study_v{BENCHMARK_VERSION}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    factories = [_factory(case) for case in train_cases]

    selection = []
    for name, candidate in sorted(SELECTION_CANDIDATES.items()):
        print(f"selection {name}", flush=True)
        config = _base_config(selection_steps, SELECTION_SEED, candidate)
        model, metrics = train_ppo(factories, config)
        score, validation = _validation_score(model, validation_cases)
        selection.append({"candidate": name, "score": score,
                          "parameters": candidate, "validation": validation,
                          "training": _compact_training(name, SELECTION_SEED, metrics)})
    selected = min(selection, key=lambda item: (item["score"], item["candidate"]))
    selected_name = selected["candidate"]
    selected_parameters = SELECTION_CANDIDATES[selected_name]

    records, training = [], []
    for seed in policy_seeds:
        for method, message_steps in (("gnn_ppo", 2), ("mlp_ppo", 0)):
            print(f"final training {method} seed={seed}", flush=True)
            config = _base_config(final_steps, seed, selected_parameters, message_steps)
            model, metrics = train_ppo(factories, config)
            checkpoint = checkpoint_dir / f"{method}_seed{seed}.pt"
            save_checkpoint(str(checkpoint), model, config,
                            {"benchmark_version": BENCHMARK_VERSION,
                             "selected_candidate": selected_name,
                             "seed": seed, "method": method})
            training.append(_compact_training(method, seed, metrics,
                                              str(checkpoint.relative_to(run_dir))))
            for case in test_cases:
                records.append(_policy_record(case, model, method, seed, runtime_repeats))
        torch.manual_seed(seed)
        untrained = GraphActorCritic(hidden=64, message_passing_steps=2)
        for case in test_cases:
            records.append(_policy_record(case, untrained, "gnn_untrained",
                                          seed, runtime_repeats))
    for case in test_cases:
        print(f"baselines {case['case_id']}", flush=True)
        records.append(_deterministic_record(case, "basic", runtime_repeats))
        records.append(_deterministic_record(case, "weighted", runtime_repeats))
        records.extend(_sabre_record(case, seed, runtime_repeats)
                       for seed in sabre_seeds)

    analysis = analyze_records(records, resamples=bootstrap_resamples,
                               seed=20260921)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": ("frozen full v2 study" if test_case_ids is None
                  else "explicit subset harness verification"),
        "configuration": {
            "selection_steps": selection_steps, "final_steps": final_steps,
            "selection_seed": SELECTION_SEED,
            "policy_seeds": list(policy_seeds), "sabre_seeds": list(sabre_seeds),
            "runtime_warmups": 1, "runtime_repeats": runtime_repeats,
            "bootstrap_resamples": bootstrap_resamples,
            "test_case_ids": [case["case_id"] for case in test_cases],
        },
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform(),
                        "processor": platform.processor(),
                        "packages": _versions(), "git": _git_metadata()},
        "selection": selection, "selected_candidate": selected_name,
        "training": training, "records": records, "analysis": analysis,
        "manifest_sha256": sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "interpretation_limits": [
            "The dataset is synthetic and does not establish production or SOTA performance.",
            "The declared hardware profile and edge errors are proxies, not device calibration.",
            "Dynamic classical control remains outside the routed contract.",
            "Bootstrap intervals summarize this frozen suite and do not imply a broader population.",
        ],
    }
    raw_path = run_dir / "raw.json"
    with raw_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    summary = {key: payload[key] for key in (
        "schema_version", "benchmark_version", "created_utc", "scope",
        "configuration", "environment", "selected_candidate", "analysis",
        "interpretation_limits")}
    with (run_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return raw_path
