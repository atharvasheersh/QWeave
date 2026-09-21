# Benchmark v2 reviewed report

Benchmark v2.0 was frozen before evaluation. The aggressive validation-only candidate was selected (score 19.5; conservative and standard each scored 24.0), then GNN-PPO and the no-message-passing PPO control were trained for 4,096 environment steps at seeds 7, 17, 29, 41, and 53. The untouched test split contains 36 cases from QFT and hardware-efficient families at widths 4, 6, and 8 on grid, chorded-ring, and idle-site hardware. The final dataset has 792 method/seed/case records.

## Main result

All 792 outputs passed the applicable numerical semantic check. All QWeave records also passed symbolic route replay; SABRE symbolic replay is unassessed because Qiskit does not expose QWeave's event trace. After explicit reverse-CNOT and SWAP lowering, all 792 hardware-cost circuits passed directed-coupler legality.

The study does not support a superiority claim for GNN-PPO. Its paired median depth difference versus weighted routing was 0 gates with a 95% bootstrap interval [0, 0], and versus SABRE was 0 [0, 5]. The GNN and no-message-passing PPO control were also indistinguishable in paired median depth and SWAP count: both differences were 0 with intervals [0, 0]. PPO training did matter relative to the untrained graph control, but message passing did not show a benefit at this scale. Weighted routing matched or improved the aggregate learned results without fallbacks and was substantially faster.

## Method-level test summary

Seeds are collapsed within each case for depth and SWAP quartiles. Runtime quartiles use all measured method/seed records.

| Method | Cases | Depth median [IQR] | SWAP median [IQR] | Runtime median [IQR] s | Semantic | Directed | Fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|
| Basic | 36 | 48.5 [31.5, 68.2] | 11.5 [1.0, 19.2] | 0.0022 [0.0015, 0.0036] | 1.00 | 1.00 | 0 |
| Weighted | 36 | 36.0 [24.0, 68.2] | 0.0 [0.0, 16.2] | 0.0036 [0.0024, 0.0054] | 1.00 | 1.00 | 0 |
| SABRE | 36 | 46.0 [27.5, 65.2] | 7.0 [1.0, 15.0] | 0.0028 [0.0025, 0.0030] | 1.00 | 1.00 | 0 |
| GNN-PPO | 36 | 37.5 [24.0, 75.2] | 0.0 [0.0, 39.0] | 0.3473 [0.1720, 1.1694] | 1.00 | 1.00 | 4543 |
| No-message PPO | 36 | 36.0 [24.0, 81.0] | 0.0 [0.0, 26.2] | 0.1040 [0.0568, 0.1958] | 1.00 | 1.00 | 742 |
| Untrained GNN | 36 | 198.0 [109.8, 323.2] | 193.0 [110.0, 334.0] | 0.6681 [0.3472, 1.3409] | 1.00 | 1.00 | 8059 |

## Paired comparisons

A negative depth/SWAP/makespan difference favors the first method. A positive estimated-log-success difference favors the first method.

| Comparison | Metric | Paired median | 95% bootstrap interval | Cases |
|---|---|---:|---:|---:|
| GNN-PPO - Weighted | depth | 0.00 | [0.00, 0.00] | 36 |
| GNN-PPO - SABRE | depth | 0.00 | [0.00, 5.00] | 36 |
| GNN-PPO - Weighted | swap_count | 0.00 | [0.00, 5.00] | 36 |
| GNN-PPO - Weighted | hardware_makespan_ns | 0.00 | [0.00, 1113.00] | 36 |
| GNN-PPO - No-message PPO | depth | 0.00 | [0.00, 0.00] | 36 |
| GNN-PPO - No-message PPO | swap_count | 0.00 | [0.00, 0.00] | 36 |
| GNN-PPO - Untrained GNN | depth | -162.50 | [-184.00, -106.00] | 36 |
| GNN-PPO - Untrained GNN | swap_count | -175.00 | [-223.00, -148.00] | 36 |

## Family behavior and counterexamples

On hardware-efficient circuits, weighted, GNN-PPO, and no-message PPO all had median depth 27 and median SWAP count 0. On QFT, weighted routing had median depth 71.5 and 16.5 SWAPs, GNN-PPO had 83.5 and 41, and no-message PPO had 90 and 31.5. The strongest GNN-PPO depth improvement over weighted routing was 8 gates on `qft_i1_6q_chorded_ring`; the largest regression was 89 gates and 130 additional SWAPs on `qft_i1_8q_grid2`.

The trained GNN incurred 4,543 test fallbacks, compared with 742 for the no-message PPO control and zero for deterministic methods. Seed 29 was a clear instability case: its GNN training curve ended with 212 fallbacks and a recent mean return near -103. Learning curves and checkpoints are included with this report.

## Scheduling and directed costs

Every unit-duration fixed-route scheduling depth delta was zero. This is a within-route result and does not imply routing improvement. Before direction synthesis, most routed circuits were illegal on the one-way couplers. The reviewed cost pass therefore lowers reverse CNOTs by Hadamard conjugation and decomposes SWAPs into three direction-lowered CNOTs. All lowered circuits pass ordered-arc validation. Median added depth was 32 for weighted routing, 34 for SABRE, 37.5 for GNN-PPO, 28.5 for no-message PPO, 57 for Basic, and 441 for the untrained GNN.

Gate durations and edge-error values are declared synthetic proxies. They are not calibration data and cannot establish hardware fidelity. The corrected paired GNN-PPO minus weighted makespan difference was 0 ns with interval [0, 1113], so the proxy-cost analysis also gives no learned advantage.

## Reproducibility

- Frozen manifest SHA-256: `8d768d96a9a3f0609d6f2de2436110b91e7b716d7a7a01d97756545ee13716af`
- Corrected raw SHA-256: `7f38e5d788aaa451980a13145504e5eebaf1ca22f5b8ff3a75030e34c6bbafda`
- Pre-correction raw SHA-256: `f063c62864bea577cdc1057019c31103aa598362265deee9efbd399753c6f400`
- Training/evaluation commit: `b77744aa4a6bbb27b0e83c763d25480e5f7126ee` (clean)
- Directed-cost correction commit: `2b7331195ed1689abe077db1ee6f9b25a55de584` (clean)
- Python 3.11.15; Qiskit 2.5.2; 5,000 bootstrap resamples; bootstrap seed 20260921.

The source raw record is retained because the initial audit found that its duration calculation scheduled pre-synthesis circuits on an undirected graph. No routing, policy, split, or non-runtime route metric changed during correction.

## Claim boundary

This synthetic, finite study supports three bounded conclusions: the tested outputs passed the declared semantic checks; PPO training strongly improved on the untrained policy; and the tested GNN message passing did not improve paired median outcomes over the no-message control. It does not support a state-of-the-art, hardware-fidelity, optimal-routing, or general superiority claim.
