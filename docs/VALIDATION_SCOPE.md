# Validation contract (draft, 2026-09-16)

This contract distinguishes the checks the code performs now from the checks
required before a result can support a paper claim.

## Current deterministic interface

- `mapping[logical_qubit] = physical_qubit`. It must contain exactly the logical
  indices `0..n-1`, map injectively, and use hardware nodes.
- The router consumes a Qiskit circuit and an undirected NetworkX hardware
  graph. For routing, physical labels must be non-negative integers; experiments
  here use connected graphs labeled contiguously `0..p-1`.
- The router processes operations in source order, inserts only edge-local
  SWAPs, and updates its `final_mapping` after each SWAP. The output may use
  more physical than logical qubits when `p > n`.
- `validate_two_qubit_legality` checks adjacency on undirected edges. It does
  **not** check the direction of CX, basis-gate support, unitary equivalence,
  classical conditions, or execution fidelity.
- The `validation: true` field in existing smoke JSON denotes this edge
  legality check. It must not be described as a complete correctness proof.
- New smoke runs use a UTC timestamp plus a random run identifier and create
  files exclusively; a filename collision fails rather than replacing raw
  records. Existing September 8 records remain untouched.

## Small-instance semantic test

`tests/test_semantic_equivalence.py` tests equal-width, four- and six-qubit
unitary circuits. It forms the wire-permutation operators for the initial and final
layouts and checks `U_routed P_initial = P_final U_logical` numerically. It also
checks that the oracle rejects a wrong final layout. This is a test of the
deterministic router; it is not yet a production validator for arbitrary
circuits or a test of the SABRE output.

## Acceptance gate for experimental records

Before widening the benchmark suite or using numbers as a paper result:

1. Normalize every candidate to a common input gate set and hardware graph.
   Reject or explicitly preprocess three-plus-qubit gates, dynamic circuits,
   measurements, resets, conditioned operations, and directed-coupling cases
   until corresponding semantics are tested.
2. Check mapping completeness/injectivity, edge legality, and the returned
   final layout for the deterministic methods. Add a semantic validator for
   the agreed gate families and a Qiskit-layout-aware check for SABRE.
3. Record separate booleans and error details for mapping, legality, and
   semantics. A failure stays in the dataset and counts toward validity rate;
   it never contributes a depth/SWAP win.
4. Keep tests for isolated logical qubits, idle physical nodes, disconnected
   hardware, tied shortest paths, nonadjacent CX, mixed unitary gates, and
   adversarial layouts. Compare small cases to the mapping oracle only when
   it reports `OPTIMAL`, and compare the same stated objective.

This scope is deliberately narrower than arbitrary quantum circuits. The
paper must state the tested family and avoid using edge legality as a proxy
for semantic equivalence.
