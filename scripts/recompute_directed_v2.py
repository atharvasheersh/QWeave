"""Recompute benchmark-v2 costs after directed-gate synthesis."""

import argparse

from qweave.experiments.recompute_directed import recompute_directed_hardware


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("--manifest", default="benchmarks/benchmark_v2_manifest.json")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()
    print(recompute_directed_hardware(
        args.raw, manifest_path=args.manifest,
        bootstrap_resamples=args.bootstrap_resamples))


if __name__ == "__main__":
    main()
