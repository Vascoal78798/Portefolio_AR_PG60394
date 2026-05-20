from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mdps.car_rental import CarRentalMDP, CarRentalParams, CarRentalPolicy, poisson_pmf_truncated


def plot_poisson_distributions(params: CarRentalParams):
    dists = [
        {"name": "Requests Loc 1", "lambda": params.lambdas[0], "max_k": params.max_requests_1, "color": "red", "ls": "-"},
        {"name": "Requests Loc 2", "lambda": params.lambdas[1], "max_k": params.max_requests_2, "color": "blue", "ls": "-"},
        {"name": "Returns Loc 1", "lambda": params.lambdas[2], "max_k": params.max_returns_1, "color": "green", "ls": "--"},
        {"name": "Returns Loc 2", "lambda": params.lambdas[3], "max_k": params.max_returns_2, "color": "purple", "ls": "--"},
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    for dist in dists:
        pmf = poisson_pmf_truncated(dist["lambda"], dist["max_k"])
        k_values = np.arange(len(pmf))
        ax.plot(
            k_values,
            pmf,
            marker="o",
            linestyle=dist["ls"],
            color=dist["color"],
            label=f'{dist["name"]} (lambda = {dist["lambda"]})',
        )

    ax.set_title("Poisson PMF for All Four Distributions")
    ax.set_xlabel("Number of Events (k)")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_xticks(np.arange(0, max(dist["max_k"] for dist in dists) + 1))
    fig.tight_layout()
    return fig, ax


def policy_to_array(mdp: CarRentalMDP, policy: CarRentalPolicy) -> np.ndarray:
    arr = np.zeros((mdp.params.max_cars_1 + 1, mdp.params.max_cars_2 + 1), dtype=int)
    for (n1, n2), action in policy.items():
        arr[n1, n2] = action
    return arr


def plot_policy(mdp: CarRentalMDP, policy: CarRentalPolicy, title: str = ""):
    arr = policy_to_array(mdp, policy)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("# cars at location 2")
    ax.set_ylabel("# cars at location 1")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.85)
    return fig, ax


def plot_values(mdp: CarRentalMDP, V: np.ndarray, title: str = ""):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(V, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("# cars at location 2")
    ax.set_ylabel("# cars at location 1")
    fig.colorbar(im, ax=ax, shrink=0.85)
    return fig, ax
