from __future__ import annotations

import numpy as np

from core.base import Environment

BanditState = int
BanditAction = int


class KArmedBandit(Environment[BanditState, BanditAction]):
    """K-armed bandit environment with optional non-stationary random walk."""

    def __init__(
        self,
        k: int = 10,
        stationary: bool = True,
        walk_std: float = 0.01,
        seed: int | None = None,
    ) -> None:
        self.k = k
        self.stationary = stationary
        self.walk_std = walk_std
        self.rng = np.random.default_rng(seed)
        self.state: BanditState = 0
        self.q_true = np.zeros(self.k, dtype=np.float64)
        self.optimal_action = 0
        self.reset()

    def reset(self) -> BanditState:
        self.q_true = self.rng.standard_normal(self.k)
        self.optimal_action = int(np.argmax(self.q_true))
        self.state = 0
        return self.state

    def available_actions(self, state: BanditState) -> list[BanditAction]:
        return list(range(self.k))

    def step(self, action: BanditAction) -> tuple[BanditState, float, bool]:
        reward = float(self.rng.standard_normal() + self.q_true[action])

        if not self.stationary:
            self.q_true += self.rng.normal(0.0, self.walk_std, self.k)
            self.optimal_action = int(np.argmax(self.q_true))

        return self.state, reward, True
