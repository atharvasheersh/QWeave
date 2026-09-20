# QWeave benchmark v1 report

The frozen run completed from Git commit `fe6e0725c1f3cfad4ea751e151e28b5e195d878f`
with a clean working tree. It used 256 PPO environment steps per policy seed,
policy seeds 7, 17, and 29, SABRE seeds 3, 7, 11, 17, and 29, one runtime
warmup, and three measured repetitions. The untouched test split contains four
family-topology cases. The raw JSON is 86,685 bytes with SHA-256
`19dca6b75451aea31afccce468169a6f2e455cbccb83b93d05c4a5f8aea8f473`.

| Method | Test cases | Seed runs | Median depth | Median SWAPs | Median runtime s | Semantic validity |
|---|---:|---:|---:|---:|---:|---:|
| Basic | 4 | 4 | 9.5 | 3.0 | 0.00850 | 100% |
| Weighted | 4 | 4 | 7.5 | 1.5 | 0.00831 | 100% |
| SABRE | 4 | 20 | 10.0 | 2.5 | 0.00523 | 100% |
| GNN-PPO | 4 | 12 | 7.5 | 1.0 | 0.05883 | 100% |
| No-GNN PPO | 4 | 12 | 7.5 | 1.0 | 0.02601 | 100% |
| Untrained GNN | 4 | 12 | 7.5 | 1.0 | 0.05941 | 100% |

Seeds are collapsed within each case before aggregation across the four cases.
All 64 outputs passed the small-unitary semantic check. All 64 unit-duration
scheduling deltas were zero. GNN-PPO and no-GNN PPO had zero test fallbacks.
The untrained control invoked deterministic fallback 16 times across its 12
records and contained severe seed-specific outliers.

The learned medians are numerically lower than Basic and SABRE in this small
suite, but the identical no-GNN and untrained aggregate medians mean the run
does not isolate a benefit from message passing or PPO. The suite is synthetic,
contains only four final test cases, and used a small training budget. These
results do not support a SOTA, general superiority, or production claim.
