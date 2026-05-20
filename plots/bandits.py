from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_reward_curves(
    curves: dict[str, np.ndarray],
    title: str,
    ylabel: str = "Average reward",
):
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for label, values in curves.items():
        ax.plot(values, label=label)
    ax.set_xlabel("Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig, ax


def plot_optimal_action_curves(
    curves: dict[str, np.ndarray],
    title: str,
):
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for label, values in curves.items():
        ax.plot(100.0 * values, label=label)
    ax.set_xlabel("Steps")
    ax.set_ylabel("% optimal action")
    ax.set_title(title)
    ax.legend()
    return fig, ax
