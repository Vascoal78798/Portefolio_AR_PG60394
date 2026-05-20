from __future__ import annotations

import numpy as np

from agents.bandits import BanditAgent
from envs.k_bandits import KArmedBandit


def run_experiment(
    agent: BanditAgent,
    env: KArmedBandit,
    steps: int = 1_000,
    runs: int = 2_000,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.zeros((runs, steps), dtype=np.float64)
    optimal = np.zeros((runs, steps), dtype=np.float64)

    for run in range(runs):
        env.reset()
        agent.reset()

        for step in range(steps):
            action = agent.select_action()
            _, reward, _ = env.step(action)
            agent.update(action, reward)

            rewards[run, step] = reward
            optimal[run, step] = float(action == env.optimal_action)

    return rewards.mean(axis=0), optimal.mean(axis=0)
