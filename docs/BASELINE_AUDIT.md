# Local baseline audit — 2026-09-16 (Asia/Kolkata)

Source checkout: public `atharvasheersh/QWeave`, `main` at
`30ab11c17b4c002fb48ea23069881d10906d54d2`. A clean Python 3.11.15
virtual environment was installed from the project/test extras. The pinned
CI constraints were generated in `requirements-ci.txt`; they constrain
runtime/test dependencies but do not hash-lock build-system downloads.

Installed versions: Qiskit 2.5.2, NetworkX 3.6.1, NumPy 2.4.6, pandas
3.0.5, OR-Tools 9.15.6755, pytest 9.1.1. Local OS: Windows. Original suite:
12 passed in 17.30 seconds. The final expanded suite (including three
unitary-layout tests and a raw-record preservation test): 16 passed in
5.83 seconds using `python -m pytest -q -p no:cacheprovider` with an
isolated fixture directory. The sandboxed initial pytest invocation emitted
a cache-directory permission warning; a later `tmp_path` fixture required
an unrestricted local test run because the sandbox denied access to its
temporary directory. These are local filesystem constraints, not test
assertion failures.

`run_smoke('run_artifacts/initial_smoke')` generated a fresh JSON/CSV pair at
UTC timestamp `20260915T202624Z` without touching the repository's tracked
September 8 smoke files. SHA-256:

```text
smoke_20260915T202624Z.json CDD3DDC5021D932505DFE44AE364BB489D8B23EB811BFD2BB0EA1F126398AF8B
smoke_20260915T202624Z.csv  BE285BA61CE52CF3647F74ABCF9A05E5734D97438EE1E8CF20A40911AA5A60CD
```

After the writer changed to UUID-suffixed exclusive creation, a second
fresh smoke run produced
`run_artifacts/postfix_smoke/smoke_20260915T203515Z_888c0ac123be45a2a86c75f4e8227ed9.json`
and its CSV peer. Their SHA-256 hashes are respectively
`B6147831FA71C49317833A829F8BD645FEB8840B12855DEB2EE5EB4EF16E5E46`
and `152537B927966796BB3D5D3810D03D4886200F95AD77C446101B3FDB4515B7AF`.
The records remain local and ignored by Git.

For CX-only `cnot_chain`, `star`, and `random_like` on `line4`, the rerun
reported compiled `(depth, SWAPs)` as follows:

| Circuit | Basic | Weighted + fallback | Qiskit SABRE |
| --- | --- | --- | --- |
| cnot_chain | (3, 0) | (3, 0) | (3, 0) |
| star | (5, 2) | (4, 1) | (5, 2) |
| random_like | (11, 8) | (5, 2) | (7, 4) |

All three records set `validation: true`, which is edge-legality validation
only. They do not carry a semantic check for SABRE, a representative circuit
sample, or a frozen hardware/gate-basis comparison. Runtime includes
different paths for the methods and is particularly unstable at this tiny
scale. These numbers are a local sanity audit, not a publishable superiority
result. The new GitHub Actions workflow is staged locally and has **not**
run on GitHub yet.
