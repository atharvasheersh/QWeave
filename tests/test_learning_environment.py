"""The learned-routing environment must never bypass deterministic guards."""

import networkx as nx
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from qiskit import QuantumCircuit

from qweave.core.validation import validate_routing_result, validate_small_unitary_equivalence
from qweave.learning import QubitRoutingEnv


def _circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 3)
    circuit.x(2)
    circuit.cx(2, 1)
    return circuit


def test_gymnasium_contract_and_observation_shapes() -> None:
    source = _circuit()
    mapping = {logical: logical for logical in range(4)}
    env = QubitRoutingEnv(source, nx.path_graph(4), mapping)
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=17)
    assert env.observation_space.contains(observation)
    assert np.array_equal(observation["action_mask"], info["action_mask"])
    assert observation["node_features"].shape == (4, 5)
    assert observation["edge_features"].shape == (3, 4)
    assert observation["action_mask"][0] == 1  # one-qubit H is executable


def test_illegal_execute_is_guarded_without_mutation() -> None:
    source = QuantumCircuit(3)
    source.cx(0, 2)
    env = QubitRoutingEnv(source, nx.path_graph(3), {0: 0, 1: 1, 2: 2}, max_nonprogress=3)
    before, _ = env.reset(seed=3)
    after, reward, terminated, truncated, info = env.step(0)
    assert reward < 0
    assert not terminated and not truncated
    assert env.pointer == 0 and len(env.output.data) == 0
    assert np.array_equal(before["mapping"], after["mapping"])
    assert not info["fallback_used"]


def test_repeated_nonprogress_invokes_fallback_and_terminates_validly() -> None:
    source = QuantumCircuit(3)
    source.cx(0, 2)
    mapping = {0: 0, 1: 1, 2: 2}
    graph = nx.path_graph(3)
    env = QubitRoutingEnv(source, graph, mapping, max_nonprogress=2)
    env.reset(seed=5)
    env.step(0)  # illegal EXECUTE, no mutation
    _, _, terminated, _, info = env.step(0)
    assert terminated and info["fallback_used"]
    result = env.routing_result()
    assert result.swap_count == 1
    assert validate_routing_result(source, result, mapping, graph)
    assert validate_small_unitary_equivalence(source, result, mapping) is True


@pytest.mark.parametrize("seed", [2, 7, 19])
def test_random_masked_policy_always_returns_a_valid_route(seed: int) -> None:
    source = _circuit()
    graph = nx.cycle_graph(4)
    mapping = {logical: logical for logical in range(4)}
    env = QubitRoutingEnv(source, graph, mapping, max_nonprogress=4)
    observation, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    terminated = truncated = False
    while not (terminated or truncated):
        legal = np.flatnonzero(observation["action_mask"])
        action = int(rng.choice(legal))
        observation, _, terminated, truncated, _ = env.step(action)
    result = env.routing_result()
    assert validate_routing_result(source, result, mapping, graph)
    assert validate_small_unitary_equivalence(source, result, mapping) is True


def test_environment_rejects_hardware_without_fallback_guarantee() -> None:
    source = QuantumCircuit(2)
    source.cx(0, 1)
    graph = nx.Graph([(0, 1)])
    graph.add_node(2)
    with pytest.raises(ValueError, match="connected hardware"):
        QubitRoutingEnv(source, graph, {0: 0, 1: 1})
