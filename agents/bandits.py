from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from envs.k_bandits import BanditAction


class BanditAgent(ABC):
    def __init__(self, k: int = 10, seed: int | None = None) -> None:
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.Q = np.zeros(self.k, dtype=np.float64)
        self.N = np.zeros(self.k, dtype=np.float64)
        self.t = 0

    @abstractmethod
    def select_action(self) -> BanditAction:
        raise NotImplementedError

    @abstractmethod
    def update(self, action: BanditAction, reward: float) -> None:
        raise NotImplementedError


class EpsilonGreedy(BanditAgent):
    def __init__(
        self,
        k: int = 10,
        epsilon: float = 0.1,
        alpha: float | None = None,
        optimistic: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.epsilon = epsilon
        self.alpha = alpha
        self.optimistic = optimistic
        super().__init__(k=k, seed=seed)

    def reset(self) -> None:
        super().reset()
        self.Q[:] = self.optimistic

    def select_action(self) -> BanditAction:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.k))
        return int(np.argmax(self.Q))

    def update(self, action: BanditAction, reward: float) -> None:
        self.t += 1
        self.N[action] += 1
        if self.alpha is None:
            self.Q[action] += (reward - self.Q[action]) / self.N[action]
        else:
            self.Q[action] += self.alpha * (reward - self.Q[action])


class UCB(BanditAgent):
    def __init__(self, k: int = 10, c: float = 2.0, seed: int | None = None) -> None:
        self.c = c
        super().__init__(k=k, seed=seed)

    def select_action(self) -> BanditAction:
        self.t += 1
        for action in range(self.k):
            if self.N[action] == 0:
                return action
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        return int(np.argmax(ucb_values))

    def update(self, action: BanditAction, reward: float) -> None:
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


class GradientBandit(BanditAgent):
    def __init__(
        self,
        k: int = 10,
        alpha: float = 0.1,
        baseline: bool = True,
        seed: int | None = None,
    ) -> None:
        self.alpha = alpha
        self.baseline = baseline
        super().__init__(k=k, seed=seed)

    def reset(self) -> None:
        super().reset()
        self.H = np.zeros(self.k, dtype=np.float64)
        self.avg_reward = 0.0

    def _policy(self) -> np.ndarray:
        exp = np.exp(self.H - np.max(self.H))
        return exp / np.sum(exp)

    def select_action(self) -> BanditAction:
        probs = self._policy()
        return int(self.rng.choice(self.k, p=probs))

    def update(self, action: BanditAction, reward: float) -> None:
        self.t += 1
        probs = self._policy()

        if self.baseline:
            self.avg_reward += (reward - self.avg_reward) / self.t
            baseline = self.avg_reward
        else:
            baseline = 0.0

        for a in range(self.k):
            if a == action:
                self.H[a] += self.alpha * (reward - baseline) * (1.0 - probs[a])
            else:
                self.H[a] -= self.alpha * (reward - baseline) * probs[a]
