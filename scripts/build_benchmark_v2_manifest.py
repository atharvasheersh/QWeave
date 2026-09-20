"""Write the frozen benchmark-v2 manifest without overwriting prior output."""

import argparse
import json
from pathlib import Path

from qweave.experiments.benchmark_v2_dataset import benchmark_v2_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/benchmark_v2_manifest.json")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(benchmark_v2_manifest(), handle, indent=2)
    print(path)


if __name__ == "__main__":
    main()
