# QWeave benchmark v2 protocol

This protocol and `benchmarks/benchmark_v2_manifest.json` are frozen before
running benchmark v2. Expanding or changing cases after evaluation requires a
new benchmark version.

## Dataset and split

Version 2.0 contains 180 deterministic cases at 4, 6, and 8 logical qubits,
with two instances per circuit family. Training families are GHZ, layered CX,
seeded interactions, Bernstein-Vazirani, and QAOA MaxCut. QFT and a
hardware-efficient variational ansatz are held out from training.

- Training: 60 cases on path and ring hardware.
- Validation: 84 cases covering topology transfer plus held-out families on
  path/ring. Hyperparameters may use only this split.
- Test: 36 cases combining the two held-out families with two-row grid,
  chorded-ring, and larger two-row grids containing idle physical sites.

Every case stores source QASM2 and SHA-256, explicit undirected and directed
edge lists, logical and physical width, family, instance, generation seed, and
declared edge-error values. The frozen manifest SHA-256 is
`8d768d96a9a3f0609d6f2de2436110b91e7b716d7a7a01d97756545ee13716af`.

## Semantic acceptance

Every QWeave route must pass operation-by-operation symbolic replay, mapping
evolution, classical-target replay, and physical edge legality. Deterministic
statevector probes provide an additional numerical check for supported unitary
circuits up to 12 physical qubits, including wider logical circuits and idle
physical sites. Terminal measurements and resets are structurally replayed;
terminal measurement destinations are preserved. Dynamic classical control
remains an explicit rejection because branch-dependent layouts are not yet
routed. Directed coupling legality is measured separately: ordered gates need
the ordered arc, while a native SWAP requires reciprocal arcs.

## Training and model selection

Three GNN-PPO candidates are trained for 1,024 steps with selection seed 101:

| Candidate | Learning rate | Entropy coefficient |
|---|---:|---:|
| standard | 0.0003 | 0.010 |
| conservative | 0.0001 | 0.020 |
| aggressive | 0.0005 | 0.005 |

All use two message-passing rounds, hidden size 64, rollout size 128, four
update epochs, and minibatches of 32. The validation score is the median over
validation cases of `depth + 0.5 * SWAPs + 5 * fallbacks`; lower is better and
candidate name breaks an exact tie. Test cases cannot participate in selection.

The selected configuration is trained for 4,096 steps at seeds 7, 17, 29, 41,
and 53. A no-message-passing PPO control receives the same configuration and
budget. An untrained GNN control is initialized at the same seeds. Every final
trained model is checkpointed, and each rollout contributes a learning-curve
record.

## Comparators and costs

Test evaluation includes GNN-PPO, no-GNN PPO, untrained GNN, Basic,
weighted deterministic routing, and SABRE with seeds 7, 17, 29, 41, and 53.
Runtime uses one warmup and three measured repetitions. Timing is descriptive
and is not compared between the original and independent machines.

Fixed-route scheduling is reported both with unit durations and with the
declared `declared_superconducting_proxy_v1` profile. The profile uses explicit
nanosecond durations and independent gate-error proxies. Its edge values are
synthetic assumptions stored in the manifest, not device calibration. Directed
legality is reported separately and is never silently converted into an
undirected success.

## Statistical analysis

Seeds are collapsed within an instance before comparing methods. The report
includes per-family median and interquartile range, paired differences against
weighted routing and SABRE, deterministic 95% bootstrap confidence intervals
with 5,000 resamples and seed 20260921, validity and directed-legality rates,
fallback totals, runtime distributions, and the largest positive and negative
per-instance differences. The analysis is descriptive for this synthetic suite;
no superiority or state-of-the-art claim follows from a confidence interval
alone.
