# Academic and evidence audit — 2026-09-21

## Scope

This audit checks the manuscript against the tracked implementation, frozen
benchmark-v2 record, primary cited papers, current Qiskit documentation, and
the clean-clone reproduction. It evaluates whether each conclusion is no
stronger than the available evidence; it is not peer review.

## Claim audit

| Manuscript claim | Evidence | Assessment |
|---|---|---|
| Basic, weighted mapping/routing, SABRE, and a small CP-SAT mapping oracle are separate references | Compiler, baseline, and oracle modules plus tests | Supported. The text correctly states that CP-SAT does not prove optimal routing. |
| Scheduling is measured after a route is fixed | Scheduler API, experiment records, and zero unit-duration deltas | Supported. The paper does not attribute mapping or routing changes to scheduling. |
| Wider and idle-site unitary circuits receive numerical semantic probes | Adapter/semantic tests and benchmark-v2 records | Supported within the declared 12-physical-qubit probe limit. Reset and measurement semantics remain structural only. |
| All 792 benchmark-v2 outputs passed applicable numerical checks | `results/benchmark_v2/raw.json` and reviewed summary | Supported for the frozen synthetic suite and implemented probes. This is not a general proof. |
| All direction-lowered cost circuits pass directed legality | Corrected schema-2.1 raw record and directed-lowering tests | Supported. The retained pre-correction raw file and correction disclosure make the audit trail explicit. |
| PPO training helps relative to an untrained GNN | Paired depth and SWAP intervals in the reviewed report | Supported on the held-out test split. |
| The tested GNN message passing and learned path do not outperform deterministic references | Paired medians, intervals, family results, fallbacks, and counterexamples | Supported as a bounded negative result. No general superiority or SOTA claim is made. |
| Hardware-cost results are calibrated | The paper explicitly says the durations and error rates are synthetic | No calibrated-hardware claim is made. |

## Citation audit

Checked against the linked primary records on 2026-09-21:

- Li, Ding, and Xie, arXiv:1809.02573 / ASPLOS 2019: title, authors,
  bidirectional SABRE description, reverse traversal, and mapping/routing role
  agree with the manuscript.
- Pozzi et al., arXiv:2007.15957: title, authors, deep reinforcement-learning
  routing scope, SWAP insertion, and depth objective agree with the manuscript.
- Sinha, Azad, and Singh, arXiv:2104.01992 / AAAI 2022: title, authors, GNN-aided
  Monte Carlo tree search, architecture-agnostic routing, and depth objective
  agree with the manuscript.
- Nannicini et al., arXiv:2106.06446 / ACM TQC, DOI 10.1145/3544563: title,
  authors, joint allocation/routing integer program, and small-instance role
  agree with the manuscript.
- IBM Quantum's Qiskit 2.5.2 transpiler-stage documentation distinguishes
  layout, routing, translation, optimization, and scheduling and explains that
  routing inserts SWAPs.
- The Qiskit `SabreLayout` reference confirms iterative bidirectional routing,
  initial/final layouts, idle-qubit allocation, seed trials, and its default
  combined layout/routing behavior.

No citation contradicts the claims for which it is used. The six bibliography
entries are all cited in the manuscript and there are no unresolved citation
keys.

## Reproducibility audit

- Pushed source commit: `09f538006b635afe796db88f98b2436d6c2f47e7`.
- GitHub Actions run #5: success; duration 2m19s; deterministic-core job
  success; smoke artifact present.
- Clean remote clone: `QWeave_clean_v2_20260921`.
- Locked environment: Python 3.11.15, Qiskit 2.5.2, and packages from
  `requirements-ci.txt`.
- Clean-clone suite: 69 passed in 9.35s.
- Clean-clone smoke script: completed and wrote an immutable result.
- Published benchmark-v2 package: all 18 listed SHA-256 values matched.
- Full clean-clone benchmark-v2 rerun: completed with the frozen protocol,
  including 1,024-step candidate selection, ten 4,096-step final policies,
  five fixed seeds, three timed repetitions after warmup, and 5,000 bootstrap
  resamples.
- Fresh raw SHA-256:
  `a75cc47dac8585bffd2a45f096ca5ac0e501870d18e14c3423e2dd5fbf59578b`.
- Comparison result: manifest, aggressive candidate selection, validation
  scores, all 792 non-runtime records, and all ten training summaries matched
  exactly. Every candidate record had status `ok`; all 792 passed semantic and
  directed-legality checks. The machine-readable comparison is
  `docs/benchmark_v2_rerun_comparison_2026-09-21.json`.
- The fresh raw file used schema label 2.0 because the full-study runner had
  not been relabeled when the corrected directed-cost computation became its
  default. Release v0.2.0 updates new full runs to schema 2.1 and adds a
  regression assertion; this changes metadata only, not recorded metrics.

## Remaining submission decisions

Scientific claims, code, tests, and tracked evidence are internally consistent.
External submission still requires human agreement on author order,
affiliations, corresponding author, contribution statement, conflicts,
funding, acknowledgements, target venue, and that venue's template and policy
requirements. These are listed in `paper/SUBMISSION_CHECKLIST.md` and are not
inferred from repository ownership.
