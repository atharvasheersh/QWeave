"""Write a new exclusive copy of the frozen benchmark-v1 manifest."""

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from qweave.experiments.benchmark_dataset import benchmark_manifest


def main() -> None:
    output = Path("run_artifacts")
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output / f"benchmark_manifest_v1_{stamp}_{uuid4().hex}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(benchmark_manifest(), handle, indent=2)
    print(path)


if __name__ == "__main__":
    main()
