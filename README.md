# QWeave

QWeave studies hardware-aware qubit allocation, deterministic routing, and depth minimisation for NISQ quantum compilation. The deterministic core and a separate dependency-preserving scheduler are implemented; learned routing will be integrated separately.

## Problem

Logical circuits often require interactions between qubits that are not adjacent on hardware. Mapping and routing choose a physical layout and insert SWAPs while preserving the circuit's behaviour. The primary mapping convention is `mapping[logical_qubit] = physical_qubit`.

## Architecture

```text
Qiskit circuit -> interaction graph -> weighted initial mapper -> strict local search
       |                                                        |
       +-> basic identity baseline / SABRE baseline             v
                                                deterministic fallback router
                                                           |
                                    route replay + legality validation + metrics
                                                           |
                                          depth-aware list scheduling (separate)
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
python scripts/run_scheduler_check.py
```

The scripts create timestamped JSON and CSV files in `results/` and never overwrite an existing result. No IBM Quantum credentials or hardware are required.

## Implemented algorithms

`build_interaction_graph` counts two-qubit interactions, optionally using `decay ** gate_index`. The weighted mapper seeds the highest interaction-degree logical qubit on a high-centrality hardware node, then greedily minimises weighted distance to assigned neighbours. Ties are deterministic. Local search evaluates all pairwise exchanges and accepts only strict objective decreases, so it terminates after finite descent or the configured iteration limit.

The fallback router processes operations in order, advances one logical qubit along a deterministic shortest path with hardware-edge SWAPs, and updates the layout after every inserted SWAP. It preserves source global phase and returns a trace with one event per emitted operation; validation replays that trace against the source gates and final layout before edge-legality checks. The Qiskit adapter accepts only unconditioned one- and two-qubit gates without classical bits. It rejects measurements, resets, barriers, three-plus-qubit gates, dynamic operations, directed/multigraph hardware, and invalid physical labels. This is a declared gate-family limit, not a general semantic proof. A small equal-width unitary-layout oracle is available separately.

The Basic baseline uses identity mapping plus this router. The SABRE adapter delegates to Qiskit with a fixed seed, symmetric coupling edges for the same undirected graph interpretation, and validates physical edge legality. On equal-width unitary circuits of at most six qubits, it also interprets Qiskit's initial/final transpiler layouts for the same small unitary check. Wider and idle-site cases return an unassessed semantic status; a general validator is still needed before comparative paper results.

The list scheduler operates only after a legal route is fixed. It builds dependencies from each physical qubit's gate order, ranks ready gates by remaining critical path, and records earliest start/finish times and layers. Optional positive gate durations produce a timed makespan. It never reroutes or swaps gates across a shared wire. Its `depth_delta` is `scheduled_depth - routed_depth` for the same circuit; under unit durations this is normally zero because Qiskit already computes parallel dependency depth.

`qweave.core.compiler.compile_deterministic(circuit, graph, method="basic" | "weighted", schedule=False)` is the integrated reference entry point. It returns the routing result, initial layout, small-unitary status (or `None` outside scope), and separate source/routed/scheduled stage metrics. SABRE and CP-SAT remain separate comparison methods; the latter is never a routing result.

## Objective and metrics

For interaction edge `(i, j)`, the mapping objective is `weight(i,j) * shortest_path_distance(mapping[i], mapping[j])`, summed over all interaction edges. Metrics include depth, total operations, two-qubit operations, SWAPs, runtime, absolute depth overhead, and zero-safe relative overhead.

## CP-SAT oracle scope

The CP-SAT implementation is an exact initial-mapping oracle for small instances (limited to eight logical qubits). It does not claim to solve optimal routing and is intentionally unsuitable for production-scale compilation.

## Testing and reproducibility

The pytest suite covers graph weights and decay, deterministic/injective mapping, strict local-search descent, shortest-path tie-breaking, global-phase preservation, unsupported-input failures, route replay and layout consistency, small unitary equivalence for QWeave and equal-width SABRE output, scheduling order/timing, SABRE smoke execution, and a known tiny mapping-oracle optimum. Randomness is explicit and seeded. Results include configuration, seed, timestamp, metrics, and validation status. Experimental superiority and novelty are not claimed before measured results exist.

`scripts/run_scheduler_check.py` writes a new immutable JSON record for three four-qubit smoke circuits on a four-site line. It reports Basic, weighted, and SABRE routed versus scheduled depth on each *fixed* compiled circuit. CP-SAT contributes only its small initial-mapping objective/status. This check is not a frozen benchmark or evidence of general improvement.

For the next comparative study, see [the experiment protocol](docs/EXPERIMENT_PROTOCOL.md), [validation scope](docs/VALIDATION_SCOPE.md), and [local baseline audit](docs/BASELINE_AUDIT.md). The internal [manuscript draft](paper/manuscript.tex) describes the deterministic method and planned evaluation; its main results are pending.

The earlier [research blueprint](docs/blueprint/README.md) is archived with its PDF and editable sources. It is planning material; the implemented status is described here and in the baseline audit.

## Week-1 status and future integration

Implemented: weighted interaction graph, initial mapping, local search, deterministic fallback routing, Basic and SABRE baselines, route-replay and legality checks, small unitary-layout checks, depth-aware list scheduling, metrics, CP-SAT mapping oracle, tests, and reproducible smoke comparisons. Deferred: PPO, GNNs, Gymnasium environments, neural routing, learned policies, and full exact routing formulations.

Team roles are documented in `AGENTS.md`: Atharva Sheersh Pandey owns integration and Qiskit adapters; Shrivardhini N owns formal model and scheduler; Diptesh Das owns deterministic optimisation, routing, baselines, and oracle; Haridasu Sreedhar owns RL/GNN/PPO work.

## Research disclaimer

This is research code. Validate compiled circuits and inspect raw results before drawing conclusions. The deterministic fallback is a correctness/reference implementation, not a claim of superiority over SABRE.

