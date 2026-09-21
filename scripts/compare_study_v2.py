"""Compare benchmark-v2 reruns while excluding environment-sensitive timing."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


VOLATILE_RECORD_FIELDS = {"runtime_seconds", "runtime_samples_seconds"}
TRAINING_FIELDS = {
    "method",
    "seed",
    "steps",
    "episodes",
    "return_median",
    "return_mean",
    "fallback_total",
    "training_curve",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_key(record: dict[str, Any]) -> tuple[str, str, int | None]:
    return record["case_id"], record["method"], record.get("seed")


def _stable_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items()
            if key not in VOLATILE_RECORD_FIELDS}


def _stable_training(item: dict[str, Any]) -> dict[str, Any]:
    return {key: item.get(key) for key in sorted(TRAINING_FIELDS)}


def compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_records = {_record_key(item): _stable_record(item)
                         for item in reference["records"]}
    candidate_records = {_record_key(item): _stable_record(item)
                         for item in candidate["records"]}
    all_keys = sorted(set(reference_records) | set(candidate_records),
                      key=lambda item: (item[0], item[1], -1 if item[2] is None else item[2]))
    record_mismatches = []
    for key in all_keys:
        left = reference_records.get(key)
        right = candidate_records.get(key)
        if left != right:
            changed = sorted(set((left or {})) | set((right or {})))
            changed = [field for field in changed
                       if (left or {}).get(field) != (right or {}).get(field)]
            record_mismatches.append({"key": list(key), "fields": changed})

    reference_training = {
        (item["method"], item["seed"]): _stable_training(item)
        for item in reference["training"]
    }
    candidate_training = {
        (item["method"], item["seed"]): _stable_training(item)
        for item in candidate["training"]
    }
    training_mismatches = []
    for key in sorted(set(reference_training) | set(candidate_training)):
        left = reference_training.get(key)
        right = candidate_training.get(key)
        if left != right:
            changed = sorted(set((left or {})) | set((right or {})))
            changed = [field for field in changed
                       if (left or {}).get(field) != (right or {}).get(field)]
            training_mismatches.append({"key": list(key), "fields": changed})

    candidate_statuses = Counter(item.get("status") for item in candidate["records"])
    return {
        "reference_schema": reference.get("schema_version"),
        "candidate_schema": candidate.get("schema_version"),
        "benchmark_version_match": reference.get("benchmark_version")
        == candidate.get("benchmark_version"),
        "manifest_sha256_match": reference.get("manifest_sha256")
        == candidate.get("manifest_sha256"),
        "selected_candidate_match": reference.get("selected_candidate")
        == candidate.get("selected_candidate"),
        "selection_match": reference.get("selection") == candidate.get("selection"),
        "reference_records": len(reference_records),
        "candidate_records": len(candidate_records),
        "candidate_statuses": dict(sorted(candidate_statuses.items(), key=lambda item: str(item[0]))),
        "non_runtime_record_mismatch_count": len(record_mismatches),
        "non_runtime_record_mismatches": record_mismatches,
        "training_mismatch_count": len(training_mismatches),
        "training_mismatches": training_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(_load(args.reference), _load(args.candidate))
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
