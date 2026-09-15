# QWeave experiment protocol (version 0.1, draft)

The September 8 smoke records are sanity checks on three four-qubit CX-only
circuits and one line topology. They are not the benchmark dataset or evidence
of general superiority. This protocol fixes decisions needed to construct a
credible dataset before any learned routing or depth-scheduling comparison.

## Questions and methods

RQ1: How does weighted initial mapping plus strict pairwise refinement and the
deterministic router compare with identity-plus-router and Qiskit SABRE on
legal circuits? RQ2 (future): What changes when scheduling is added? RQ3
(future): What changes when a learned policy is added, including validity and
inference/training cost? Do not combine RQ2/RQ3 results with RQ1 in a single
unqualified claim.

Run the same source circuit and hardware topology through: Basic (identity
layout plus deterministic router), weighted mapping plus refinement and that
router, and Qiskit SABRE at `optimization_level=0` with explicit transpiler
seed. CP-SAT is an **initial-mapping objective oracle**, limited to at most
eight logical qubits; it is neither an exact routing baseline nor a claim of
optimal compiled depth. Record `OPTIMAL` versus `FEASIBLE`, bound/gap, and
timeout. The current mapper/refiner use un-decayed integer interaction counts
in smoke runs; use that same objective for the oracle comparison.

## Dataset specification to freeze before measurement

Create a versioned, deterministic circuit generator with three distinct
families: structured CX/Clifford circuits (chains, stars, GHZ and layered
patterns), random two-qubit interaction circuits, and small named algorithm
circuits after their gate decomposition is fixed. Start at 4, 6, and 8 logical
qubits with fixed depths; expand only after the semantic acceptance gate in
`VALIDATION_SCOPE.md` passes. Use connected path, ring, and small-grid graphs
with explicit zero-based edge lists. Record graph size, edges, source circuit
QASM or a stable serialized representation, source hash, family, generator
version, and generation seed. Keep a held-out family/topology split for any
later learned policy; never tune on the final test split.

Before running a comparative study, check that all methods accept the same
gate family and coupling interpretation. Qiskit's directed coupling/basis
translation and device calibrations are outside this undirected study. If
compilation changes a gate basis, normalize or report the difference; do not
silently compare unlike circuits. Compare equivalent logical computations,
not merely circuits with the same adjacency pattern.

## Measurements and analysis

For each `(circuit, topology, method, seed)` store source and compiled depths,
absolute and zero-safe relative depth overhead, explicit SWAP count,
two-qubit-operation count, total operation count, runtime, output width,
initial/final mapping where available, validation outcomes, version metadata,
and error/timeout status. Source depth is an unconstrained reference, **not**
an optimal legal depth. Preserve every raw JSON/CSV record in an immutable
run directory; a fresh run gets a new ID and cannot overwrite prior records.

Use at least five explicit transpiler seeds for SABRE and repeat runtime
measurements after a warmup on a fixed machine. Deterministic methods use a
single algorithmic seed but still receive repeated timed runs. Pair methods
on exactly the same circuit/topology instance. Report per-family medians and
interquartile ranges for depth/SWAP, paired differences and bootstrap 95%
intervals over instances, validity rates, and runtime distribution. Show
individual counterexamples. Do not interpret repeated runs of the same tiny
instance as independent circuit evidence or claim significance from them.

## Reproduction manifest and release gate

Record Git commit SHA, Python/Qiskit/NetworkX/OR-Tools versions, OS/CPU, graph
and circuit hashes, seeds, warmup/repeat counts, method parameters, elapsed
time definition, and data/schema version. CI runs pytest and a smoke check;
its output is a build artifact, not a paper result. Freeze the dataset and
protocol revision before the comparative run. Release a results table only
after the semantic acceptance gate, raw-record audit, and independent rerun
pass. No experimental novelty or SOTA claim precedes that gate.
