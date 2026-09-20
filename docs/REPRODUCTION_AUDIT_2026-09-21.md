# QWeave pushed baseline reproduction audit

## GitHub CI

Commit `ba6c0fbdbe5bc472b6c11701f040fe32a58f639d` was present on
`origin/main`. GitHub Actions run
[`35533543360`](https://github.com/atharvasheersh/QWeave/actions/runs/35533543360)
completed successfully. Its dependency installation, pytest, isolated smoke,
and smoke-artifact upload steps all reported `success`.

## Independent clean clone

The public repository was cloned into a separate directory with no copied
virtual environment or run artifacts. Python 3.11.15 installed the committed
`requirements-ci.txt` constraints into a new environment. The clean clone
passed all 52 tests and produced a new smoke record.

Benchmark v1 was then rerun with its committed defaults: 256 PPO steps, policy
seeds 7/17/29, SABRE seeds 3/7/11/17/29, one timing warmup, and three measured
repetitions. The rerun produced 64 records. Record keys were identical to the
original run, every semantic result was true, and all scheduling depth deltas
were zero.

The comparison excluded timestamps and runtime values. Across depth, operation
count, two-qubit count, SWAP count, depth overhead, relative overhead,
scheduled depth, schedule delta, makespan, scheduled SWAP count, semantic
validity, fallback count, and status, there were **zero mismatches**.

The original raw JSON SHA-256 is
`19dca6b75451aea31afccce468169a6f2e455cbccb83b93d05c4a5f8aea8f473`.
The independent raw JSON SHA-256 is
`b33564f2d4db23ba3e49f155b840530af0b4f12a997a555cd870f4544c755283`.
The files differ because timestamps, wall-clock samples, and run metadata are
expected to change; the declared non-runtime outcomes reproduce exactly.
