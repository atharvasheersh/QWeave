# QWeave v0.2.0 release notes

QWeave v0.2.0 packages the correctness-first deterministic compiler path,
fixed-route scheduling, guarded learned routing, and the reviewed benchmark-v2
evidence used by the manuscript.

## Included work

- Deterministic Basic and weighted mapping/routing paths with strict Qiskit
  input validation, route replay, layout consistency checks, and explicit
  unsupported-case errors.
- Qiskit SABRE and a small CP-SAT initial-mapping oracle as distinct reference
  methods. The oracle does not establish optimal routing.
- Exact small-unitary checks plus deterministic layout-aware statevector probes
  for wider circuits and idle physical qubits.
- Fixed-route list scheduling with mapping, routing, and scheduling metrics kept
  separate.
- Directed-coupler lowering for reverse CNOTs and SWAPs before declared
  duration and synthetic edge-error evaluation.
- A guarded Gymnasium environment, masked GNN actor-critic, PPO training,
  deterministic fallback, and message-passing/untrained ablations.
- Frozen benchmark v2: 180 cases, 36 held-out test cases, 792 final records,
  five policy seeds, ten checkpoints, paired bootstrap analysis, learning
  curves, and counterexamples.
- Editable manuscript source, DOCX, and visually checked PDF in separate
  paper folders.

## Verification

- GitHub Actions run #5 passed on commit
  `09f538006b635afe796db88f98b2436d6c2f47e7`.
- A clean remote clone installed from `requirements-ci.txt` and passed all 69
  tests under Python 3.11.15 and Qiskit 2.5.2.
- All 18 files listed in `results/benchmark_v2/SHA256SUMS.txt` matched their
  published hashes in the clean clone.
- The exact benchmark-v2 rerun and comparison are recorded in
  `docs/REPRODUCTION_AUDIT_BENCHMARK_V2_2026-09-21.md`.

## Evidence boundary

The reviewed study supports PPO training relative to the untrained policy. It
does not show a message-passing benefit or learned-routing advantage over the
weighted deterministic reference. The durations and edge-error values are
declared synthetic proxies, not device calibration. This release does not
claim optimal routing, production readiness, hardware fidelity, general
superiority, or state-of-the-art performance.

## Release contents

The source archive includes the implementation, tests, protocols, frozen
manifest, reviewed raw records, summaries, figures, checkpoints, manuscript
sources, DOCX, and PDF. `RELEASE_SHA256SUMS.txt` in the external release bundle
records the archive and document hashes.
