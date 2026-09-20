"""Small message-passing actor-critic and masked PPO training loop."""

from dataclasses import asdict, dataclass
import random
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .environment import QubitRoutingEnv


def edge_index_from_env(env: QubitRoutingEnv) -> np.ndarray:
    return np.asarray(env.edges, dtype=np.int64).T


def _tensor(value: np.ndarray, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


class GraphActorCritic(nn.Module):
    """Two-step message passing with edge actions and one EXECUTE action."""

    def __init__(self, node_features: int = 5, edge_features: int = 4,
                 frontier_features: int = 5, hidden: int = 64,
                 message_passing_steps: int = 2):
        super().__init__()
        if type(message_passing_steps) is not int or message_passing_steps < 0:
            raise ValueError("message_passing_steps must be a non-negative integer")
        self.message_passing_steps = message_passing_steps
        self.node_encoder = nn.Linear(node_features, hidden)
        self.message = nn.Linear(hidden * 2 + edge_features, hidden)
        self.update = nn.Linear(hidden * 2, hidden)
        self.edge_actor = nn.Sequential(
            nn.Linear(hidden * 2 + edge_features + frontier_features, hidden),
            nn.Tanh(), nn.Linear(hidden, 1),
        )
        self.execute_actor = nn.Sequential(
            nn.Linear(hidden + frontier_features, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden + frontier_features, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )

    def forward(self, observation: dict[str, np.ndarray], edge_index: np.ndarray):
        device = next(self.parameters()).device
        node = _tensor(observation["node_features"], torch.float32, device)
        edge = _tensor(observation["edge_features"], torch.float32, device)
        frontier = _tensor(observation["frontier"], torch.float32, device)
        indices = _tensor(edge_index, torch.long, device)
        hidden = torch.tanh(self.node_encoder(node))
        for _ in range(self.message_passing_steps):
            left, right = hidden[indices[0]], hidden[indices[1]]
            messages = torch.tanh(self.message(torch.cat([left, right, edge], dim=-1)))
            aggregate = torch.zeros_like(hidden)
            aggregate.index_add_(0, indices[0], messages)
            aggregate.index_add_(0, indices[1], messages)
            degrees = torch.zeros(hidden.shape[0], device=device)
            degrees.index_add_(0, indices[0], torch.ones(indices.shape[1], device=device))
            degrees.index_add_(0, indices[1], torch.ones(indices.shape[1], device=device))
            aggregate = aggregate / degrees.clamp_min(1).unsqueeze(-1)
            hidden = torch.tanh(self.update(torch.cat([hidden, aggregate], dim=-1)))
        left, right = hidden[indices[0]], hidden[indices[1]]
        frontier_edges = frontier.expand(edge.shape[0], -1)
        swap_logits = self.edge_actor(
            torch.cat([left, right, edge, frontier_edges], dim=-1)).squeeze(-1)
        pooled = hidden.mean(dim=0)
        context = torch.cat([pooled, frontier], dim=-1)
        execute_logit = self.execute_actor(context).reshape(1)
        logits = torch.cat([execute_logit, swap_logits])
        value = self.critic(context).squeeze(-1)
        return logits, value

    def distribution(self, observation: dict[str, np.ndarray], edge_index: np.ndarray) -> tuple[Categorical, torch.Tensor]:
        logits, value = self(observation, edge_index)
        device = logits.device
        mask = _tensor(observation["action_mask"], torch.bool, device)
        if not bool(mask.any()):
            raise ValueError("policy received a state without a legal action")
        return Categorical(logits=logits.masked_fill(~mask, -1e9)), value

    @torch.no_grad()
    def act(self, observation: dict[str, np.ndarray], edge_index: np.ndarray,
            deterministic: bool = False) -> tuple[int, float, float]:
        distribution, value = self.distribution(observation, edge_index)
        action = torch.argmax(distribution.logits) if deterministic else distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 512
    rollout_steps: int = 128
    update_epochs: int = 4
    minibatch_size: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    seed: int = 7
    hidden: int = 64
    message_passing_steps: int = 2

    def __post_init__(self):
        integer_positive = (self.total_steps, self.rollout_steps, self.update_epochs,
                            self.minibatch_size, self.hidden)
        if any(type(value) is not int or value < 1 for value in integer_positive):
            raise ValueError("PPO step, epoch, batch, and hidden sizes must be positive integers")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and GAE lambda must be in their probability ranges")
        if self.learning_rate <= 0 or self.clip_ratio <= 0 or self.max_grad_norm <= 0:
            raise ValueError("PPO learning rate, clip ratio, and gradient norm must be positive")
        if type(self.message_passing_steps) is not int or self.message_passing_steps < 0:
            raise ValueError("message_passing_steps must be a non-negative integer")


@dataclass
class Transition:
    observation: dict[str, np.ndarray]
    edge_index: np.ndarray
    action: int
    old_log_probability: float
    reward: float
    value: float
    next_value: float
    terminal: bool


def _copy_observation(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in observation.items()}


def _advantages(transitions: Sequence[Transition], config: PPOConfig) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(transitions), dtype=np.float32)
    accumulator = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        item = transitions[index]
        nonterminal = 0.0 if item.terminal else 1.0
        delta = item.reward + config.gamma * item.next_value * nonterminal - item.value
        accumulator = delta + config.gamma * config.gae_lambda * nonterminal * accumulator
        advantages[index] = accumulator
    returns = advantages + np.asarray([item.value for item in transitions], dtype=np.float32)
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


def train_ppo(env_factories: Sequence[Callable[[], QubitRoutingEnv]],
              config: PPOConfig = PPOConfig(), device: str = "cpu") -> tuple[GraphActorCritic, dict]:
    """Train on explicitly supplied environments; variable graphs are allowed."""

    if not env_factories:
        raise ValueError("at least one training environment is required")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch_device = torch.device(device)
    model = GraphActorCritic(hidden=config.hidden,
                             message_passing_steps=config.message_passing_steps).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    env_index = 0
    env = env_factories[env_index]()
    observation, _ = env.reset(seed=config.seed)
    episode_return = 0.0
    episode_returns: list[float] = []
    fallback_counts: list[int] = []
    losses: list[float] = []
    training_curve: list[dict] = []
    consumed = 0
    while consumed < config.total_steps:
        transitions: list[Transition] = []
        for _ in range(min(config.rollout_steps, config.total_steps - consumed)):
            edges = edge_index_from_env(env)
            action, log_probability, value = model.act(observation, edges)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            terminal = terminated or truncated
            if terminal:
                next_value = 0.0
            else:
                with torch.no_grad():
                    _, value_tensor = model(next_observation, edges)
                    next_value = float(value_tensor.item())
            transitions.append(Transition(_copy_observation(observation), edges.copy(), action,
                                          log_probability, reward, value, next_value, terminal))
            consumed += 1
            episode_return += reward
            observation = next_observation
            if terminal:
                episode_returns.append(episode_return)
                fallback_counts.append(env.fallback_count)
                episode_return = 0.0
                env_index = (env_index + 1) % len(env_factories)
                env = env_factories[env_index]()
                observation, _ = env.reset(seed=config.seed + consumed + env_index)
        advantages, returns = _advantages(transitions, config)
        indices = np.arange(len(transitions))
        loss_start = len(losses)
        for _ in range(config.update_epochs):
            rng.shuffle(indices)
            for start in range(0, len(indices), config.minibatch_size):
                batch = indices[start:start + config.minibatch_size]
                policy_terms, value_terms, entropies = [], [], []
                for index in batch:
                    item = transitions[int(index)]
                    distribution, value = model.distribution(item.observation, item.edge_index)
                    action = torch.tensor(item.action, device=torch_device)
                    ratio = torch.exp(distribution.log_prob(action) - item.old_log_probability)
                    advantage = torch.tensor(advantages[index], device=torch_device)
                    unclipped = ratio * advantage
                    clipped = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * advantage
                    policy_terms.append(-torch.minimum(unclipped, clipped))
                    target = torch.tensor(returns[index], device=torch_device)
                    value_terms.append((value - target).pow(2))
                    entropies.append(distribution.entropy())
                loss = (torch.stack(policy_terms).mean()
                        + config.value_coefficient * torch.stack(value_terms).mean()
                        - config.entropy_coefficient * torch.stack(entropies).mean())
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                losses.append(float(loss.detach().item()))
        recent_losses = losses[loss_start:]
        training_curve.append({
            "steps": consumed,
            "episodes": len(episode_returns),
            "mean_recent_loss": (float(np.mean(recent_losses))
                                 if recent_losses else None),
            "mean_recent_return": (float(np.mean(episode_returns[-10:]))
                                   if episode_returns else None),
            "fallback_total": int(sum(fallback_counts)),
        })
    metrics = {"config": asdict(config), "steps": consumed,
               "episodes": len(episode_returns), "episode_returns": episode_returns,
               "fallback_counts": fallback_counts, "losses": losses,
               "training_curve": training_curve}
    return model, metrics


@torch.no_grad()
def evaluate_policy(model: GraphActorCritic,
                    env_factories: Sequence[Callable[[], QubitRoutingEnv]],
                    seeds: Sequence[int]) -> list[dict]:
    """Evaluate greedily and retain validation-relevant episode metadata."""

    records = []
    for case_index, factory in enumerate(env_factories):
        for seed in seeds:
            env = factory()
            observation, _ = env.reset(seed=seed)
            total_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _, _ = model.act(observation, edge_index_from_env(env), deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
            result = env.routing_result()
            records.append({"case_index": case_index, "seed": seed,
                            "reward": total_reward, "compiled_depth": result.circuit.depth(),
                            "inserted_swaps": result.swap_count,
                            "fallback_count": env.fallback_count,
                            "terminated": terminated, "truncated": truncated})
    return records


def save_checkpoint(path: str, model: GraphActorCritic, config: PPOConfig,
                    metadata: dict | None = None) -> None:
    torch.save({"model_state": model.state_dict(), "config": asdict(config),
                "metadata": dict(metadata or {})}, path)


def load_checkpoint(path: str, device: str = "cpu") -> tuple[GraphActorCritic, PPOConfig, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    config = PPOConfig(**payload["config"])
    model = GraphActorCritic(hidden=config.hidden,
                             message_passing_steps=config.message_passing_steps).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config, payload.get("metadata", {})
