# Learned routing scope

QWeave's learned path is isolated under `qweave.learning`; it does not change
the deterministic compiler. `QubitRoutingEnv` exposes one `EXECUTE` action and
one SWAP action per sorted hardware edge. Its observation contains node and
edge features, the next source operation, the current logical-to-physical
mapping, and a binary legal-action mask.

An illegal action never changes the circuit or layout. Repeated non-progress,
or the episode limit, invokes the validated deterministic shortest-path router.
Every completed episode is replay-validated before a result is returned. This
guarantees a legal result on the declared connected, undirected, contiguous
hardware scope; it does not prove that a policy found a good route.

`GraphActorCritic` performs configurable message passing over the hardware
graph and scores each hardware edge plus `EXECUTE`. The PPO loop uses clipped
policy updates, GAE, value and entropy terms, gradient clipping, explicit seeds,
and the action mask. It processes one variable-size graph transition at a time.
This implementation is intentionally small and auditable; it is not presented
as a production trainer or a novelty claim.

The benchmark includes two controls:

- `mlp_ppo` sets message-passing steps to zero while retaining PPO training.
- `gnn_untrained` uses the graph policy without PPO updates.

These controls estimate the contributions of message passing and training only
within the frozen v1 setup. They do not establish causality beyond that setup.
All learned variants start from the weighted deterministic initial mapping, so
their comparison primarily tests route selection. Basic, weighted deterministic
routing, and Qiskit SABRE remain separate reference methods.
