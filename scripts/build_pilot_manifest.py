"""Write a new, never-overwritten pilot-fixture manifest."""

import json
from pathlib import Path
from uuid import uuid4

from qweave.experiments.pilot_dataset import pilot_manifest


def main() -> None:
    output = Path("run_artifacts/pilot_manifest")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"pilot_{uuid4().hex}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(pilot_manifest(), handle, indent=2)
    print(path)


if __name__ == "__main__":
    main()
