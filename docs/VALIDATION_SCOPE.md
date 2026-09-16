# Validation contract (draft, 2026-09-16)

This contract distinguishes the checks the code performs now from the checks
required before a result can support a paper claim.

## Current deterministic interface

- `mapping[logical_qubit] = physical_qubit`. It must contain exactly the logical
  indices `0..n-1`, map injectively, and use hardware nodes.
- The deterministic Qiskit adapter accepts only unconditioned one- and
  two-qubit `Gate` operations, no classical bits, and no barriers, resets,
  measurements, delays, or three-plus-qubit gates. It preserves global phase.
- Hardware must be an undirected simple NetworkX graph with non-negative
  integer labels and no self-loops. The deterministic router can represent idle
  and sparse physical labels; the SABRE adapter requires contiguous labels
  `0..p-1` and passes both directions of each undirected edge to Qiskit.
- The router processes operations in source order, inserts only edge-local
  SWAPs, and updates its `final_mapping` after each inserted SWAP. The output may use
  more physical than logical qubits when `p > n`.
- `validate_routing_result` replays the one-event-per-output-operation trace,
  source gate order, inserted SWAP layouts, final mapping, output width, and
  global phase. `validate_two_qubit_legality` checks adjacency on undirected
  edges and rejects operations outside the accepted gate family. These are
  structural checks. They do **not** check the direction of CX, basis-gate support, arbitrary unitary
  equivalence, classical conditions, or execution fidelity.
- The `validation: true` field in existing smoke JSON denotes this edge
  legality check. It must not be described as a complete correctness proof.
- New smoke runs use a UTC timestamp plus a random run identifier and create
  files exclusively; a filename collision fails rather than replacing raw
  records. Existing September 8 records remain untouched.

## Small-instance semantic test

`validate_small_unitary_equivalence` tests equal-width cases of at most six
qubits, forming the initial/final wire permutations and checking
`U_routed P_initial = P_final U_logical` up to global phase. It returns `None`
outside that scope and raises for a mismatch. Tests include mixed gates,
source SWAPs, and a wrong final layout. The SABRE adapter applies the same
equal-width oracle using Qiskit's initial/final transpiler layouts. Neither
path is a validator for arbitrary circuits, idle-site cases, or wider hardware.

The list scheduler checks input/output edge legality, retains each physical
wire's gate order, and reports timed layers and the depth change on the *same*
routed circuit. Its timing metadata does not certify semantic equivalence; the
small unitary oracle is run for the deterministic and equal-width SABRE smoke
cases. Wider or idle-site SABRE semantics remain unassessed. CP-SAT remains an oracle
only for the scaled static initial-mapping objective on tiny connected graphs.

`compile_deterministic` integrates Basic or weighted mapping with the shared
router and optional scheduler. Its `semantic_validation: None` means the
small-unitary oracle was outside scope, not that correctness passed.

## Acceptance gate for experimental records

Before widening the benchmark suite or using numbers as a paper result:

1. Normalize every candidate to a common input gate set and hardware graph.
   Reject or explicitly preprocess three-plus-qubit gates, dynamic circuits,
   measurements, resets, conditioned operations, and directed-coupling cases
   until corresponding semantics are tested.
2. Check mapping completeness/injectivity, edge legality, and the returned
   final layout for the deterministic methods. Add a semantic validator for
   the agreed gate families and extend the Qiskit-layout-aware SABRE check
   beyond its current small equal-width scope.
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
