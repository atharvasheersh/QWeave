# QWeave

QWeave studies hardware-aware qubit allocation, deterministic routing, and depth minimisation for NISQ quantum compilation. This repository is the early Week-1 deterministic core; learned routing will be integrated separately.

## Problem

Logical circuits often require interactions between qubits that are not adjacent on hardware. Mapping and routing choose a physical layout and insert SWAPs while preserving the circuit's behaviour. The primary mapping convention is `mapping[logical_qubit] = physical_qubit`.

## Architecture

```text
Qiskit circuit -> interaction graph -> weighted initial mapper -> strict local search
       |                                                        |
       +-> basic identity baseline / SABRE baseline             v
                                                deterministic fallback router
                                                           |
                                              legality validation + metrics
                                                           |
                                                JSON/CSV experiment results
```

## Installation and quick start

Requires Python 3.11 or newer.

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[test]"
pytest -q
python scripts/run_smoke.py
python scripts/run_baselines.py
```

The scripts create timestamped JSON and CSV files in `results/` and never overwrite an existing result. No IBM Quantum credentials or hardware are required.

## Implemented algorithms

`build_interaction_graph` counts two-qubit interactions, optionally using `decay ** gate_index`. The weighted mapper seeds the highest interaction-degree logical qubit on a high-centrality hardware node, then greedily minimises weighted distance to assigned neighbours. Ties are deterministic. Local search evaluates all pairwise exchanges and accepts only strict objective decreases, so it terminates after finite descent or the configured iteration limit.

The fallback router processes operations in order, advances one logical qubit along a deterministic shortest path with hardware-edge SWAPs, updates the layout after every SWAP, and validates the final circuit. The Basic baseline uses identity mapping plus this router. The SABRE baseline delegates to Qiskit with a fixed transpiler seed.

## Objective and metrics

For interaction edge `(i, j)`, the mapping objective is `weight(i,j) * shortest_path_distance(mapping[i], mapping[j])`, summed over all interaction edges. Metrics include depth, total operations, two-qubit operations, SWAPs, runtime, absolute depth overhead, and zero-safe relative overhead.

## CP-SAT oracle scope

The CP-SAT implementation is an exact initial-mapping oracle for small instances (limited to eight logical qubits). It does not claim to solve optimal routing and is intentionally unsuitable for production-scale compilation.

## Testing and reproducibility

The pytest suite covers graph weights and decay, deterministic/injective mapping, strict local-search descent, shortest-path tie-breaking, routing legality and disconnected hardware, validation failures, SABRE smoke execution, and a known tiny oracle optimum. Randomness is explicit and seeded. Results include configuration, seed, timestamp, metrics, and validation status. Experimental superiority and novelty are not claimed before measured results exist.

For the next comparative study, see [the experiment protocol](docs/EXPERIMENT_PROTOCOL.md), [validation scope](docs/VALIDATION_SCOPE.md), and [local baseline audit](docs/BASELINE_AUDIT.md). The internal [manuscript draft](paper/manuscript.tex) describes the deterministic method and planned evaluation; its main results are pending.

The earlier [research blueprint](docs/blueprint/README.md) is archived with its PDF and editable sources. It is planning material; the implemented status is described here and in the baseline audit.

## Week-1 status and future integration

Implemented: weighted interaction graph, initial mapping, local search, deterministic fallback routing, Basic and SABRE baselines, legality checks, metrics, CP-SAT mapping oracle, tests, and reproducible smoke comparisons. Deferred: PPO, GNNs, Gymnasium environments, neural routing, learned policies, and full exact routing formulations.

Team roles are documented in `AGENTS.md`: Atharva Sheersh Pandey owns integration and Qiskit adapters; Shrivardhini N owns formal model and scheduler; Diptesh Das owns deterministic optimisation, routing, baselines, and oracle; Haridasu Sreedhar owns RL/GNN/PPO work.

## Research disclaimer

This is research code. Validate compiled circuits and inspect raw results before drawing conclusions. The deterministic fallback is a correctness/reference implementation, not a claim of superiority over SABRE.

