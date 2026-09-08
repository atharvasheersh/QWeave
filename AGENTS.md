# QWeave Agent Guidance

QWeave is research software. Correctness and reproducibility take priority over feature volume.

- Preserve shared interfaces; never change them silently.
- Understand another owner's interface contract before modifying their work.
- Do not assume SABRE or QWeave is superior, and never fabricate experiment results.
- Never overwrite raw result files. Every algorithm change requires tests; metric changes require regression tests.
- The mapping convention is `mapping[logical_qubit] = physical_qubit`.
- Routing must preserve mapping consistency, and compiled circuits must be validated before experiments.
- All randomness uses explicit seeds. Keep learned routing separate from deterministic routing.
- Do not introduce PPO/GNN code into deterministic modules.
- CP-SAT is a tiny-instance oracle, not a scalable production router.
- Isolate Qiskit compatibility logic when APIs differ by version.
- Avoid massive refactors during experiment periods; prefer one logical change per commit/PR.
- Update documentation when algorithm behaviour changes.

## Code owners

- Atharva Sheersh Pandey: integration, Qiskit interfaces, experiment harness
- Shrivardhini N: formal model, scheduler, correctness requirements
- Diptesh Das: mapper, deterministic router, baselines, oracle
- Haridasu Sreedhar: RL environment, GNN, PPO

