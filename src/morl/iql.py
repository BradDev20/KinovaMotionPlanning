from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.distributions import Normal
except ImportError:  # pragma: no cover - runtime guarded
    torch = None
    nn = None
    Normal = None

from .scalarization import scalarize_torch

if torch is None:
    def _no_grad_decorator():  # pragma: no cover - import guard only
        def decorator(func):
            return func
        return decorator
else:
    _no_grad_decorator = torch.no_grad


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required. Install it in the active environment before training or evaluation.")


if nn is None:

    class MLP:  # pragma: no cover - import guard only
        def __init__(self, *args, **kwargs):
            require_torch()


    class VectorQNetwork:  # pragma: no cover - import guard only
        def __init__(self, *args, **kwargs):
            require_torch()


    class VectorValueNetwork:  # pragma: no cover - import guard only
        def __init__(self, *args, **kwargs):
            require_torch()


    class GaussianPolicy:  # pragma: no cover - import guard only
        def __init__(self, *args, **kwargs):
            require_torch()

else:

    class MLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int = 3):
            super().__init__()
            layers = []
            current_dim = input_dim
            for _ in range(depth):
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.ReLU())
                current_dim = hidden_dim
            layers.append(nn.Linear(current_dim, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, inputs):
            return self.net(inputs)


    class VectorQNetwork(nn.Module):
        def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
            super().__init__()
            self.model = MLP(obs_dim + action_dim, hidden_dim, 2)

        def forward(self, observation, action):
            return self.model(torch.cat([observation, action], dim=-1))


    class VectorValueNetwork(nn.Module):
        def __init__(self, obs_dim: int, hidden_dim: int):
            super().__init__()
            self.model = MLP(obs_dim, hidden_dim, 2)

        def forward(self, observation):
            return self.model(observation)


    class GaussianPolicy(nn.Module):
        def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
            super().__init__()
            self.backbone = MLP(obs_dim + 2, hidden_dim, hidden_dim)
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.full((action_dim,), -1.5))

        def distribution(self, observation, weights):
            features = self.backbone(torch.cat([observation, weights], dim=-1))
            mean = self.mean_head(features)
            std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
            return Normal(mean, std)

        def forward(self, observation, weights):
            return self.distribution(observation, weights)


@dataclass(frozen=True)
class IQLConfig:
    hidden_dim: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    expectile: float = 0.7
    beta: float = 3.0
    exp_adv_clip: float = 100.0
    rho: float = 0.01
    scalarizer_mode: str = "sum"
    action_limit: float = 0.35


class PreferenceConditionedIQL:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: IQLConfig,
        device: str = "cpu",
        obs_mean: np.ndarray | None = None,
        obs_std: np.ndarray | None = None,
    ):
        require_torch()
        self.config = config
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.q1 = VectorQNetwork(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.q2 = VectorQNetwork(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.value = VectorValueNetwork(obs_dim, config.hidden_dim).to(self.device)
        self.policy = GaussianPolicy(obs_dim, action_dim, config.hidden_dim).to(self.device)

        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=config.lr)
        self.v_optimizer = torch.optim.Adam(self.value.parameters(), lr=config.lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)

        mean = np.zeros(obs_dim, dtype=np.float32) if obs_mean is None else np.asarray(obs_mean, dtype=np.float32)
        std = np.ones(obs_dim, dtype=np.float32) if obs_std is None else np.asarray(obs_std, dtype=np.float32)
        self.obs_mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device)
        self.obs_std = torch.as_tensor(np.maximum(std, 1e-6), dtype=torch.float32, device=self.device)

    def _normalize_obs(self, observation):
        return (observation - self.obs_mean) / self.obs_std

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        observation = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        next_observation = torch.as_tensor(batch["next_observations"], dtype=torch.float32, device=self.device)
        reward_vector = torch.as_tensor(batch["reward_vectors"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device)

        observation = self._normalize_obs(observation)
        next_observation = self._normalize_obs(next_observation)

        with torch.no_grad():
            target_value = self.value(next_observation)
            target_q = reward_vector + self.config.gamma * (1.0 - done) * target_value

        q1_prediction = self.q1(observation, action)
        q2_prediction = self.q2(observation, action)
        q_loss = torch.nn.functional.mse_loss(q1_prediction, target_q) + torch.nn.functional.mse_loss(q2_prediction, target_q)

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        with torch.no_grad():
            q_min = torch.minimum(self.q1(observation, action), self.q2(observation, action))

        value_prediction = self.value(observation)
        advantage_vector = q_min - value_prediction
        scalar_advantage = scalarize_torch(advantage_vector, weights, self.config.scalarizer_mode, rho=self.config.rho)
        expectile_weights = torch.where(
            scalar_advantage >= 0.0,
            torch.full_like(scalar_advantage, self.config.expectile),
            torch.full_like(scalar_advantage, 1.0 - self.config.expectile),
        ).unsqueeze(-1)
        value_loss = (expectile_weights * (q_min.detach() - value_prediction).pow(2)).mean()

        self.v_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self.v_optimizer.step()

        with torch.no_grad():
            value_for_actor = self.value(observation)
            actor_advantage = scalarize_torch(
                torch.minimum(self.q1(observation, action), self.q2(observation, action)) - value_for_actor,
                weights,
                self.config.scalarizer_mode,
                rho=self.config.rho,
            )
            actor_weights = torch.exp(self.config.beta * actor_advantage).clamp(max=self.config.exp_adv_clip)

        distribution = self.policy.distribution(observation, weights)
        log_prob = distribution.log_prob(action).sum(dim=-1)
        policy_loss = -(actor_weights * log_prob).mean()

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()

        return {
            "q_loss": float(q_loss.item()),
            "value_loss": float(value_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "advantage_mean": float(actor_advantage.mean().item()),
        }

    @_no_grad_decorator()
    def act(self, observation: np.ndarray, weights: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        weights_tensor = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs_tensor = self._normalize_obs(obs_tensor)
        distribution = self.policy.distribution(obs_tensor, weights_tensor)
        action = distribution.mean if deterministic else distribution.sample()
        action = action.squeeze(0).cpu().numpy()
        return np.clip(action, -self.config.action_limit, self.config.action_limit).astype(np.float32)

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "config": self.config.__dict__,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "obs_mean": self.obs_mean.detach().cpu().numpy(),
            "obs_std": self.obs_std.detach().cpu().numpy(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "value": self.value.state_dict(),
            "policy": self.policy.state_dict(),
        }

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any], device: str = "cpu") -> "PreferenceConditionedIQL":
        config = IQLConfig(**payload["config"])
        agent = cls(
            obs_dim=int(payload["obs_dim"]),
            action_dim=int(payload["action_dim"]),
            config=config,
            device=device,
            obs_mean=np.asarray(payload["obs_mean"], dtype=np.float32),
            obs_std=np.asarray(payload["obs_std"], dtype=np.float32),
        )
        agent.q1.load_state_dict(payload["q1"])
        agent.q2.load_state_dict(payload["q2"])
        agent.value.load_state_dict(payload["value"])
        agent.policy.load_state_dict(payload["policy"])
        return agent
