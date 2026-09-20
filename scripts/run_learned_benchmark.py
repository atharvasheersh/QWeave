"""Run the frozen benchmark-v1 learned-routing comparison."""

import argparse

from qweave.experiments.learned_benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="run_artifacts")
    parser.add_argument("--training-steps", type=int, default=256)
    parser.add_argument("--policy-seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument("--sabre-seeds", nargs="+", type=int, default=[3, 7, 11, 17, 29])
    parser.add_argument("--runtime-repeats", type=int, default=3)
    arguments = parser.parse_args()
    output = run_benchmark(arguments.output, training_steps=arguments.training_steps,
                           policy_seeds=tuple(arguments.policy_seeds),
                           sabre_seeds=tuple(arguments.sabre_seeds),
                           runtime_repeats=arguments.runtime_repeats)
    print(output)


if __name__ == "__main__":
    main()
