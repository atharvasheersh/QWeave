"""Run the frozen benchmark-v2 training and analysis protocol."""

import argparse

from qweave.experiments.study_v2 import run_study_v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="run_artifacts")
    parser.add_argument("--manifest", default="benchmarks/benchmark_v2_manifest.json")
    parser.add_argument("--selection-steps", type=int, default=1024)
    parser.add_argument("--final-steps", type=int, default=4096)
    parser.add_argument("--runtime-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()
    print(run_study_v2(args.output, manifest_path=args.manifest,
                       selection_steps=args.selection_steps,
                       final_steps=args.final_steps,
                       runtime_repeats=args.runtime_repeats,
                       bootstrap_resamples=args.bootstrap_resamples))


if __name__ == "__main__":
    main()
