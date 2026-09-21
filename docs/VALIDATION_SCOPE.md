# Validation contract (reviewed 2026-09-21)

This contract distinguishes structural, numerical, and hardware-cost checks.
No single boolean is presented as a general proof for arbitrary quantum
programs.

## Deterministic interface

- `mapping[logical_qubit] = physical_qubit`. It must contain exactly the
  logical indices, map injectively, and use declared hardware nodes.
- The Qiskit adapter accepts unconditioned one- and two-qubit gates, one-qubit
  resets, and a terminal measurement suffix with preserved classical targets.
  Barriers, three-plus-qubit gates, dynamic control flow, and classically
  conditioned operations are explicit errors.
- The mapping/router hardware input is an undirected simple NetworkX graph
  with non-negative integer labels. Idle physical sites are supported.
- The router processes source operations in order, inserts only edge-local
  SWAPs, updates its layout after each SWAP, and returns a physical circuit,
  final mapping, trace, and inserted-SWAP count.

## Structural route replay

`validate_routing_result` replays the one-event-per-output-operation trace. It
checks source operation order and parameters, physical and classical targets,
mapping evolution, inserted SWAP edges, final mapping, output width, global
phase, and physical adjacency. A trace mismatch is a hard failure.

Reset and terminal-measurement circuits are accepted through this structural
contract. Their numerical semantics are intentionally reported as unassessed.
Dynamic branch-dependent layouts remain outside the routed contract.

## Numerical semantic oracles

Equal-width unitary cases of at most six qubits use an exact operator check:

`U_routed P_initial = P_final U_source`

up to global phase. Wider unitary circuits and circuits with idle physical
sites use deterministic layout-aware statevector probes when the physical
width is at most 12. The probe oracle is reproducible and catches tested
semantic mismatches, but it is probabilistic rather than a complete proof.

SABRE uses Qiskit's exposed initial and final layouts for the same exact or
probe-based comparison. Its semantic status is assessed independently from
QWeave's symbolic trace because SABRE does not expose that trace format.

## Directed hardware and costs

Routing comparisons use a shared undirected adjacency interpretation. Hardware
cost evaluation then applies an explicit direction-lowering pass:

- a legal forward CNOT is retained;
- a reverse CNOT is conjugated by Hadamards on both qubits around the available
  forward CNOT;
- a SWAP is decomposed into three CNOTs, each lowered independently;
- symmetric CZ, CP, and RZZ gates may use either orientation of a connected
  coupler;
- other ordered gates require their declared arc.

The lowered circuit must pass ordered-coupler legality before scheduling or
error-proxy calculation. The benchmark records pre-synthesis legality,
post-synthesis legality, added operations, and added depth separately. Declared
durations and edge errors are synthetic assumptions, not calibration data.

## Scheduling contract

The list scheduler accepts a fixed legal physical circuit and preserves each
physical wire's operation order. It reports unit-layer depth and a positive-
duration makespan without changing mapping or routing. Scheduling differences
are reported only against the same circuit. A zero unit-duration delta is
expected when Qiskit's dependency depth already exposes all safe parallelism.

## Benchmark-v2 acceptance gate

For every record used in the reviewed report:

1. Preserve the frozen source QASM, split, graph edges, and manifest hash.
2. Require symbolic replay for QWeave routes and applicable numerical semantic
   validation for every compiler path.
3. Keep failures and fallback counts in the dataset; never score an invalid
   output as a depth or SWAP win.
4. Collapse seeds within each case before paired comparisons.
5. Report family-level distributions, 5,000-resample paired bootstrap
   intervals, runtime quartiles, fallback totals, and counterexamples.
6. Lower to the declared directed couplers before accepting duration/error
   costs, and retain the pre-lowering legality result.

Benchmark v2 met these checks for its 792 test records after the documented
schema-2.1 directed-cost correction. The result supports only the declared
synthetic scope. It does not establish optimal routing, calibrated hardware
fidelity, production readiness, or state-of-the-art performance.
