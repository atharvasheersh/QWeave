"""Paired benchmark of deterministic, SABRE, and learned routing paths."""

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
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
from qweave.core.validation import validate_small_unitary_equivalence
from qweave.learning import GraphActorCritic, PPOConfig, QubitRoutingEnv, edge_index_from_env, train_ppo
from qweave.mapper.initial_mapper import initial_mapping
from qweave.mapper.local_search import refine_mapping
from qweave.metrics.circuit_metrics import circuit_metrics, timed
from qweave.scheduling import schedule_routed_circuit, schedule_routing_result

from .benchmark_dataset import BENCHMARK_VERSION, benchmark_manifest, load_case


SCHEMA_VERSION = "1.0"


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


def _weighted_mapping(case: dict) -> dict[int, int]:
    circuit, graph = load_case(case)
    seed = initial_mapping(circuit, graph)
    return refine_mapping(circuit, seed.mapping, graph).mapping


def _environment_factory(case: dict):
    frozen = json.loads(json.dumps(case))

    def factory() -> QubitRoutingEnv:
        circuit, graph = load_case(frozen)
        return QubitRoutingEnv(circuit, graph, _weighted_mapping(frozen))

    return factory


def _schedule_fields(schedule) -> dict:
    return {"scheduled_depth": schedule.depth_after,
            "schedule_depth_delta": schedule.depth_delta,
            "scheduled_makespan": schedule.makespan,
            "scheduled_swap_count": circuit_metrics(schedule.circuit)["swap_count"]}


def _deterministic_record(case: dict, method: str, runtime_repeats: int) -> dict:
    circuit, graph = load_case(case)
    runs = [timed(compile_deterministic, circuit, graph, method, True)
            for _ in range(runtime_repeats + 1)]
    compiled = runs[-1][0]
    samples = [item[1] for item in runs[1:]]
    runtime = float(np.median(samples))
    return {"case_id": case["case_id"], "family": case["family"],
            "topology": case["topology"], "method": method, "seed": None,
            **circuit_metrics(compiled.routed.circuit, circuit, runtime),
            "runtime_samples_seconds": samples,
            **_schedule_fields(compiled.scheduled),
            "semantic_validation": compiled.semantic_validation,
            "fallback_count": 0, "status": "ok"}


def _sabre_record(case: dict, seed: int, runtime_repeats: int) -> dict:
    circuit, graph = load_case(case)
    runs = [run_sabre_baseline(circuit, graph, seed)
            for _ in range(runtime_repeats + 1)]
    result = runs[-1]
    samples = [item["runtime_seconds"] for item in runs[1:]]
    result["runtime_seconds"] = float(np.median(samples))
    schedule = schedule_routed_circuit(result["circuit"], graph)
    metrics = {key: value for key, value in result.items()
               if key not in {"circuit", "method", "seed"}}
    return {"case_id": case["case_id"], "family": case["family"],
            "topology": case["topology"], "method": "sabre", "seed": seed,
            **metrics, "runtime_samples_seconds": samples,
            **_schedule_fields(schedule), "fallback_count": 0,
            "status": "ok"}


def _policy_episode(case: dict, model: GraphActorCritic, seed: int):
    circuit, graph = load_case(case)
    mapping = _weighted_mapping(case)
    env = QubitRoutingEnv(circuit, graph, mapping)
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    started = time.perf_counter()
    while not (terminated or truncated):
        action, _, _ = model.act(observation, edge_index_from_env(env), deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
    runtime = time.perf_counter() - started
    return circuit, graph, mapping, env, env.routing_result(), runtime, terminated, truncated


def _policy_record(case: dict, model: GraphActorCritic, method: str, seed: int,
                   runtime_repeats: int) -> dict:
    runs = [_policy_episode(case, model, seed) for _ in range(runtime_repeats + 1)]
    circuit, graph, mapping, env, routed, _, terminated, truncated = runs[-1]
    samples = [item[5] for item in runs[1:]]
    runtime = float(np.median(samples))
    schedule = schedule_routing_result(circuit, routed, mapping, graph)
    semantic = validate_small_unitary_equivalence(circuit, routed, mapping)
    return {"case_id": case["case_id"], "family": case["family"],
            "topology": case["topology"], "method": method, "seed": seed,
            **circuit_metrics(routed.circuit, circuit, runtime),
            "runtime_samples_seconds": samples,
            **_schedule_fields(schedule), "semantic_validation": semantic,
            "fallback_count": env.fallback_count, "terminated": terminated,
            "truncated": truncated, "status": "ok"}


def summarize(records: list[dict]) -> list[dict]:
    """Aggregate seeds within a case before aggregating across test cases."""

    per_case = defaultdict(list)
    for record in records:
        per_case[(record["method"], record["case_id"])].append(record)
    collapsed = defaultdict(list)
    for (method, _), rows in per_case.items():
        collapsed[method].append({
            metric: float(np.median([row[metric] for row in rows]))
            for metric in ("depth", "swap_count", "runtime_seconds",
                           "schedule_depth_delta", "fallback_count")
        })
    output = []
    for method, rows in sorted(collapsed.items()):
        method_records = [row for row in records if row["method"] == method]
        output.append({"method": method, "test_cases": len(rows),
                       "seed_runs": len(method_records),
                       **{f"median_{metric}": float(np.median([row[metric] for row in rows]))
                          for metric in rows[0]},
                       "semantic_valid_rate": sum(row["semantic_validation"] is True
                                                  for row in method_records) / len(method_records)})
    return output


def run_benchmark(output_root: str | Path, *, training_steps: int = 256,
                  policy_seeds: tuple[int, ...] = (7, 17, 29),
                  sabre_seeds: tuple[int, ...] = (3, 7, 11, 17, 29),
                  runtime_repeats: int = 3,
                  test_case_ids: set[str] | None = None) -> Path:
    """Train controls and evaluate untouched test cases into a new run directory."""

    manifest = benchmark_manifest()
    train_cases = [case for case in manifest["cases"] if case["split"] == "train"]
    test_cases = [case for case in manifest["cases"] if case["split"] == "test"]
    if test_case_ids is not None:
        test_cases = [case for case in test_cases if case["case_id"] in test_case_ids]
    if not test_cases:
        raise ValueError("at least one frozen test case must be selected")
    if type(runtime_repeats) is not int or runtime_repeats < 1:
        raise ValueError("runtime_repeats must be a positive integer")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex
    run_dir = Path(output_root) / f"benchmark_v{BENCHMARK_VERSION}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    factories = [_environment_factory(case) for case in train_cases]
    records = []
    training = []
    for seed in policy_seeds:
        for label, steps in (("gnn_ppo", 2), ("mlp_ppo", 0)):
            config = PPOConfig(total_steps=training_steps,
                               rollout_steps=min(64, training_steps),
                               update_epochs=2, minibatch_size=min(32, training_steps),
                               seed=seed, hidden=32, message_passing_steps=steps)
            model, metrics = train_ppo(factories, config)
            training.append({"method": label, "seed": seed, **metrics})
            for case in test_cases:
                records.append(_policy_record(case, model, label, seed, runtime_repeats))
        torch.manual_seed(seed)
        untrained = GraphActorCritic(hidden=32, message_passing_steps=2)
        for case in test_cases:
            records.append(_policy_record(case, untrained, "gnn_untrained", seed,
                                          runtime_repeats))
    for case in test_cases:
        records.extend((_deterministic_record(case, "basic", runtime_repeats),
                        _deterministic_record(case, "weighted", runtime_repeats)))
        records.extend(_sabre_record(case, seed, runtime_repeats) for seed in sabre_seeds)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "scope": ("frozen full test split" if test_case_ids is None
                  else "explicit test subset for harness verification"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {"training_steps": training_steps,
                          "policy_seeds": list(policy_seeds),
                          "sabre_seeds": list(sabre_seeds),
                          "runtime_warmups": 1,
                          "runtime_repeats": runtime_repeats,
                          "test_case_ids": [case["case_id"] for case in test_cases]},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "packages": _versions(),
                        "git": _git_metadata()},
        "manifest": manifest,
        "training": training,
        "records": records,
        "summary": summarize(records),
        "interpretation_limits": [
            "This is a small synthetic benchmark and is not a SOTA comparison.",
            "Policy seeds are training replicates; seed rows for one case are collapsed before case aggregation.",
            "The no-message-passing and untrained controls isolate architecture and PPO training only within this setup.",
            "Scheduling preserves fixed-route wire order, so unit-duration depth deltas may be zero.",
        ],
    }
    output = run_dir / "raw.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with (run_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump({"configuration": payload["configuration"],
                   "summary": payload["summary"],
                   "interpretation_limits": payload["interpretation_limits"]}, handle, indent=2)
    return output
