from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from envs.gridworld import Gridworld
from policies.gridworld import GridworldGreedyPolicy

ARROW = {"U": "↑", "D": "↓", "L": "←", "R": "→", ".": "."}


def plot_grid_values_and_policy(
    env: Gridworld,
    V: np.ndarray,
    policy: Optional[GridworldGreedyPolicy] = None,
    title: str = "",
    value_fmt: str = "{:.2f}",
    ax=None,
):
    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    ax.set_title(title)
    ax.set_xlim(0, env.n_cols)
    ax.set_ylim(0, env.n_rows)
    ax.set_xticks(np.arange(env.n_cols + 1))
    ax.set_yticks(np.arange(env.n_rows + 1))
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for state in env.terminal_states:
        row, col = state
        rect = plt.Rectangle((col, row), 1, 1, fill=True, alpha=0.15)
        ax.add_patch(rect)

    for row in range(env.n_rows):
        for col in range(env.n_cols):
            state = (row, col)
            ax.text(col + 0.5, row + 0.45, value_fmt.format(V[state]), ha="center", va="center", fontsize=12)
            if policy is not None:
                action = policy.get(state, ".")
                ax.text(col + 0.5, row + 0.78, ARROW.get(action, "."), ha="center", va="center", fontsize=18)

    return fig, ax
