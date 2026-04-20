from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    action_scale: float = 0.35
    max_joint_velocity: float = 1.3
    success_threshold: float = 0.03
    terminal_failure_penalty: float = 1.0
    terminate_on_contact: bool = True
    max_steps: int | None = None
