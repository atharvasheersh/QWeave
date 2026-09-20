"""Guarded Gymnasium environment for sequential qubit routing."""

from dataclasses import dataclass
import math

import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import SwapGate

from qweave.core.qiskit_adapter import physical_output_circuit, source_operations, validate_hardware_graph
from qweave.core.types import Mapping, RoutingResult
from qweave.core.validation import validate_mapping, validate_mapping_inverse, validate_routing_result
from qweave.routing.shortest_path import deterministic_shortest_path


@dataclass(frozen=True)
class RoutingReward:
    """Reward terms kept explicit for reproducible ablations."""

    execute: float = 1.0
    depth: float = 1.0
    swap: float = 0.25
    illegal: float = 5.0
    fallback: float = 1.0
    potential: float = 0.2
    gamma: float = 0.99


class QubitRoutingEnv(gym.Env):
    """Route a fixed unitary circuit with masked SWAP/EXECUTE actions.

    Action 0 executes the next source gate when it is physically legal.
    Actions 1..E insert a SWAP on the corresponding sorted hardware edge.
    Illegal actions never mutate the circuit or layout. Repeated non-progress
    invokes the deterministic shortest-path fallback, which guarantees
    progress on the declared connected undirected hardware scope.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, circuit: QuantumCircuit, coupling_graph: nx.Graph,
                 mapping: Mapping, *, lookahead: int = 4,
                 max_nonprogress: int = 8, max_steps: int | None = None,
                 reward: RoutingReward | None = None):
        super().__init__()
        self.source = circuit
        self.operations = source_operations(circuit)
        self.width = validate_hardware_graph(coupling_graph, circuit.num_qubits)
        validate_mapping(mapping, circuit, coupling_graph)
        if set(coupling_graph.nodes) != set(range(self.width)):
            raise ValueError("learning environment requires contiguous physical labels 0..p-1")
        if not nx.is_connected(coupling_graph):
            raise ValueError("learning environment requires connected hardware for fallback termination")
        if type(lookahead) is not int or lookahead < 1:
            raise ValueError("lookahead must be a positive integer")
        if type(max_nonprogress) is not int or max_nonprogress < 1:
            raise ValueError("max_nonprogress must be a positive integer")
        self.graph = coupling_graph.copy()
        self.initial_mapping = dict(mapping)
        self.edges = sorted(tuple(sorted(edge)) for edge in self.graph.edges)
        self.lookahead = lookahead
        self.max_nonprogress = max_nonprogress
        self.max_steps = max_steps or max(1, len(self.operations) * (self.width + max_nonprogress))
        self.reward_terms = reward or RoutingReward()
        if any(not math.isfinite(value) or value < 0 for value in vars(self.reward_terms).values()):
            raise ValueError("reward coefficients must be finite and non-negative")
        self.action_space = spaces.Discrete(len(self.edges) + 1)
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(-1.0, 1.0, (self.width, 5), dtype=np.float32),
            "edge_features": spaces.Box(-1.0, 1.0, (len(self.edges), 4), dtype=np.float32),
            "frontier": spaces.Box(-1.0, 1.0, (5,), dtype=np.float32),
            "mapping": spaces.Box(-1, self.width - 1, (circuit.num_qubits,), dtype=np.int64),
            "action_mask": spaces.MultiBinary(len(self.edges) + 1),
        })
        self._reset_state()

    def _reset_state(self) -> None:
        self.pointer = 0
        self.current = dict(self.initial_mapping)
        self.inverse = validate_mapping_inverse(self.current)
        self.output = physical_output_circuit(self.source, self.width)
        self.trace: list[dict] = []
        self.inserted_swaps = 0
        self.steps = 0
        self.nonprogress = 0
        self.fallback_count = 0
        self._state_visits: dict[tuple, int] = {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._observation(), self._info()

    @property
    def done(self) -> bool:
        return self.pointer >= len(self.operations)

    def _front(self):
        return None if self.done else self.operations[self.pointer]

    def action_masks(self) -> np.ndarray:
        """Return the binary mask expected by masked policy implementations."""

        mask = np.zeros(len(self.edges) + 1, dtype=np.int8)
        front = self._front()
        if front is None:
            return mask
        if len(front.qubits) == 1:
            mask[0] = 1
        else:
            left, right = (self.current[logical] for logical in front.qubits)
            mask[0] = int(self.graph.has_edge(left, right))
        for index, (left, right) in enumerate(self.edges, start=1):
            mask[index] = int(left in self.inverse or right in self.inverse)
        return mask

    def _potential(self) -> float:
        total = 0.0
        for item in self.operations[self.pointer:self.pointer + self.lookahead]:
            if len(item.qubits) == 2:
                left, right = (self.current[logical] for logical in item.qubits)
                total += max(0, nx.shortest_path_length(self.graph, left, right) - 1)
        return -total / max(1, self.lookahead * (self.width - 1))

    def _observation(self) -> dict[str, np.ndarray]:
        node = np.zeros((self.width, 5), dtype=np.float32)
        edge = np.zeros((len(self.edges), 4), dtype=np.float32)
        front = self._front()
        occupied_scale = max(1, self.source.num_qubits - 1)
        for physical in range(self.width):
            logical = self.inverse.get(physical)
            node[physical, 0] = float(logical is not None)
            node[physical, 1] = -1.0 if logical is None else logical / occupied_scale
        frontier = np.full(5, -1.0, dtype=np.float32)
        if front is not None:
            frontier[0] = len(front.qubits) / 2.0
            frontier[1] = front.qubits[0] / occupied_scale
            if len(front.qubits) == 2:
                frontier[2] = front.qubits[1] / occupied_scale
                first, second = (self.current[logical] for logical in front.qubits)
                node[first, 2], node[second, 3] = 1.0, 1.0
                distance = nx.shortest_path_length(self.graph, first, second)
                frontier[4] = distance / max(1, self.width - 1)
                for physical in range(self.width):
                    node[physical, 4] = nx.shortest_path_length(self.graph, physical, second) / max(1, self.width - 1)
            frontier[3] = (len(self.operations) - self.pointer) / max(1, len(self.operations))
        for index, (left, right) in enumerate(self.edges):
            edge[index, 0] = (int(left in self.inverse) + int(right in self.inverse)) / 2.0
            edge[index, 1] = max(node[left, 2], node[right, 2])
            edge[index, 2] = max(node[left, 3], node[right, 3])
            if front is not None and len(front.qubits) == 2:
                before = int(round(frontier[4] * max(1, self.width - 1)))
                trial = dict(self.current)
                left_logical, right_logical = self.inverse.get(left), self.inverse.get(right)
                if left_logical is not None:
                    trial[left_logical] = right
                if right_logical is not None:
                    trial[right_logical] = left
                after = nx.shortest_path_length(self.graph, trial[front.qubits[0]], trial[front.qubits[1]])
                edge[index, 3] = np.clip((before - after) / max(1, self.width - 1), -1.0, 1.0)
        return {"node_features": node, "edge_features": edge, "frontier": frontier,
                "mapping": np.asarray([self.current[index] for index in range(self.source.num_qubits)], dtype=np.int64),
                "action_mask": self.action_masks()}

    def _info(self, **extra) -> dict:
        return {"action_mask": self.action_masks(), "source_index": self.pointer,
                "inserted_swaps": self.inserted_swaps, "fallback_count": self.fallback_count,
                **extra}

    def _execute(self) -> None:
        item = self._front()
        if item is None:
            raise RuntimeError("cannot execute after termination")
        physical = [self.current[logical] for logical in item.qubits]
        self.output.append(item.operation, physical,
                           [self.output.clbits[index] for index in item.clbits])
        self.trace.append({"kind": "source_gate", "operation": item.operation.name,
                           "source_index": item.source_index,
                           "logical_qubits": list(item.qubits),
                           "physical_qubits": physical,
                           "classical_bits": list(item.clbits)})
        self.pointer += 1

    def _swap(self, left: int, right: int) -> None:
        self.output.append(SwapGate(), [left, right], [])
        left_logical, right_logical = self.inverse.get(left), self.inverse.get(right)
        if left_logical is not None:
            self.current[left_logical] = right
        if right_logical is not None:
            self.current[right_logical] = left
        self.inverse[left], self.inverse[right] = right_logical, left_logical
        self.inserted_swaps += 1
        self.trace.append({"kind": "inserted_swap", "operation": "swap",
                           "edge": [left, right], "mapping": dict(self.current)})

    def _fallback_current(self) -> tuple[int, int]:
        """Route and execute one source operation; return swaps and depth delta."""

        before_depth = self.output.depth()
        item = self._front()
        swaps = 0
        if item is not None and len(item.qubits) == 2:
            left, right = (self.current[logical] for logical in item.qubits)
            if not self.graph.has_edge(left, right):
                path = deterministic_shortest_path(self.graph, left, right)
                for edge in zip(path, path[1:-1]):
                    self._swap(*edge)
                    swaps += 1
        self._execute()
        return swaps, self.output.depth() - before_depth

    def _fallback_all(self) -> tuple[int, int]:
        swaps = depth = 0
        while not self.done:
            added_swaps, added_depth = self._fallback_current()
            swaps += added_swaps
            depth += added_depth
        return swaps, depth

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step called after episode termination")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be in 0..{self.action_space.n - 1}")
        self.steps += 1
        mask = self.action_masks()
        old_potential = self._potential()
        before_depth = self.output.depth()
        fallback_used = False
        if not mask[action]:
            reward = -self.reward_terms.illegal
            self.nonprogress += 1
        elif action == 0:
            self._execute()
            depth_delta = self.output.depth() - before_depth
            reward = self.reward_terms.execute - self.reward_terms.depth * depth_delta
            self.nonprogress = 0
        else:
            old_front = self._front()
            old_distance = None
            if old_front is not None and len(old_front.qubits) == 2:
                old_distance = nx.shortest_path_length(
                    self.graph, *(self.current[logical] for logical in old_front.qubits))
            self._swap(*self.edges[action - 1])
            new_distance = None
            if old_front is not None and len(old_front.qubits) == 2:
                new_distance = nx.shortest_path_length(
                    self.graph, *(self.current[logical] for logical in old_front.qubits))
            reward = -self.reward_terms.swap
            self.nonprogress = self.nonprogress + 1 if old_distance is None or new_distance >= old_distance else 0
        state_key = (self.pointer, tuple(sorted(self.current.items())))
        self._state_visits[state_key] = self._state_visits.get(state_key, 0) + 1
        if self.nonprogress >= self.max_nonprogress or self._state_visits[state_key] > 2:
            swaps, depth_delta = self._fallback_current()
            reward -= self.reward_terms.fallback + self.reward_terms.swap * swaps + self.reward_terms.depth * depth_delta
            self.fallback_count += 1
            self.nonprogress = 0
            fallback_used = True
        truncated = self.steps >= self.max_steps and not self.done
        if truncated:
            swaps, depth_delta = self._fallback_all()
            reward -= self.reward_terms.fallback + self.reward_terms.swap * swaps + self.reward_terms.depth * depth_delta
            self.fallback_count += 1
            fallback_used = True
        reward += self.reward_terms.potential * (
            self.reward_terms.gamma * self._potential() - old_potential)
        terminated = self.done
        if terminated:
            validate_routing_result(self.source, self.routing_result(), self.initial_mapping, self.graph)
        return self._observation(), float(reward), terminated, truncated, self._info(fallback_used=fallback_used)

    def routing_result(self) -> RoutingResult:
        if not self.done:
            raise RuntimeError("routing result is available only after episode termination")
        return RoutingResult(self.output, dict(self.current), list(self.trace), self.inserted_swaps)

    def render(self) -> str:
        return (f"source={self.pointer}/{len(self.operations)} mapping={self.current} "
                f"swaps={self.inserted_swaps} fallback={self.fallback_count}")
