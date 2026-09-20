"""Focused checks for the masked graph policy and PPO implementation."""

from functools import partial

import networkx as nx
import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit

from qweave.learning import (
    GraphActorCritic,
    PPOConfig,
    QubitRoutingEnv,
    edge_index_from_env,
    evaluate_policy,
    load_checkpoint,
    save_checkpoint,
    train_ppo,
)


def _environment(width: int = 3) -> QubitRoutingEnv:
    circuit = QuantumCircuit(width)
    circuit.h(0)
    circuit.cx(0, width - 1)
    circuit.cx(width - 1, 1)
    graph = nx.path_graph(width)
    return QubitRoutingEnv(circuit, graph, {index: index for index in range(width)},
                           max_nonprogress=3)


def test_graph_policy_respects_action_mask_and_shapes() -> None:
    env = _environment()
    observation, _ = env.reset(seed=5)
    env.step(0)  # execute H so the non-local CX is at the frontier
    observation = env._observation()
    model = GraphActorCritic(hidden=16)
    logits, value = model(observation, edge_index_from_env(env))
    distribution, _ = model.distribution(observation, edge_index_from_env(env))
    assert logits.shape == (env.action_space.n,)
    assert value.ndim == 0
    assert observation["action_mask"][0] == 0
    assert distribution.probs[0].item() < 1e-12


def test_short_ppo_run_and_greedy_evaluation_are_valid() -> None:
    factories = [partial(_environment, 3), partial(_environment, 4)]
    config = PPOConfig(total_steps=24, rollout_steps=12, update_epochs=1,
                       minibatch_size=6, hidden=16, seed=11)
    model, metrics = train_ppo(factories, config)
    records = evaluate_policy(model, factories, seeds=[11, 13])
    assert metrics["steps"] == 24
    assert metrics["losses"] and np.isfinite(metrics["losses"]).all()
    assert metrics["training_curve"][-1]["steps"] == 24
    assert len(records) == 4
    assert all(record["terminated"] or record["truncated"] for record in records)
    assert all(record["inserted_swaps"] >= 0 for record in records)


def test_checkpoint_round_trip_preserves_policy(tmp_path) -> None:
    env = _environment()
    observation, _ = env.reset(seed=7)
    config = PPOConfig(total_steps=8, rollout_steps=8, update_epochs=1,
                       minibatch_size=4, hidden=12, message_passing_steps=0)
    model = GraphActorCritic(hidden=config.hidden, message_passing_steps=0)
    path = tmp_path / "policy.pt"
    save_checkpoint(str(path), model, config, {"case": "round-trip"})
    loaded, loaded_config, metadata = load_checkpoint(str(path))
    before, _ = model(observation, edge_index_from_env(env))
    after, _ = loaded(observation, edge_index_from_env(env))
    assert torch.equal(before, after)
    assert loaded_config == config
    assert metadata == {"case": "round-trip"}


@pytest.mark.parametrize("kwargs", [
    {"total_steps": 0},
    {"learning_rate": 0},
    {"gamma": 1.1},
    {"message_passing_steps": -1},
])
def test_invalid_ppo_configuration_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        PPOConfig(**kwargs)
