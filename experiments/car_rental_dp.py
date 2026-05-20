from __future__ import annotations

import numpy as np

from mdps.car_rental import CarRentalMDP, CarRentalPolicy, CarRentalState


def zeros_V(mdp: CarRentalMDP) -> np.ndarray:
    return np.zeros((mdp.params.max_cars_1 + 1, mdp.params.max_cars_2 + 1), dtype=float)


def q_from_v(
    mdp: CarRentalMDP,
    V: np.ndarray,
    state: CarRentalState,
    action: int,
    gamma: float,
) -> float:
    p_next_1, p_next_2, exp_revenue = mdp.expected_transition(state, action)

    expected_next = 0.0
    for n1, p1 in enumerate(p_next_1):
        if p1 == 0.0:
            continue
        for n2, p2 in enumerate(p_next_2):
            if p2 == 0.0:
                continue
            expected_next += p1 * p2 * V[n1, n2]

    move_cost = mdp.params.cost_per_moved * abs(action)
    reward = exp_revenue - move_cost
    return float(reward + gamma * expected_next)


def bellman_expectation_backup_v(
    mdp: CarRentalMDP,
    V: np.ndarray,
    state: CarRentalState,
    policy: CarRentalPolicy,
    gamma: float,
) -> float:
    return q_from_v(mdp, V, state, policy[state], gamma)


def bellman_optimality_backup_v(
    mdp: CarRentalMDP,
    V: np.ndarray,
    state: CarRentalState,
    gamma: float,
) -> float:
    best = -float("inf")
    for action in mdp.possible_actions(state):
        best = max(best, q_from_v(mdp, V, state, action, gamma))
    return float(best)


def policy_evaluation(
    mdp: CarRentalMDP,
    policy: CarRentalPolicy,
    gamma: float,
    theta: float = 1e-6,
    max_iters: int = 10_000,
) -> tuple[np.ndarray, int]:
    V = zeros_V(mdp)

    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()

        for state in mdp.states():
            action = policy[state]
            v_new = q_from_v(mdp, V_old, state, action, gamma)
            delta = max(delta, abs(v_new - V[state]))
            V[state] = v_new

        if delta < theta:
            return V, it + 1

    return V, max_iters


def policy_improvement(
    mdp: CarRentalMDP,
    V: np.ndarray,
    old_policy: CarRentalPolicy | None,
    gamma: float,
) -> tuple[CarRentalPolicy, bool]:
    new_policy: CarRentalPolicy = {}
    stable = True

    for state in mdp.states():
        best_action = None
        best_q = -np.inf
        for action in mdp.possible_actions(state):
            q = q_from_v(mdp, V, state, action, gamma)
            if q > best_q:
                best_q = q
                best_action = action

        new_policy[state] = int(best_action) if best_action is not None else 0
        if old_policy is not None and old_policy[state] != new_policy[state]:
            stable = False

    return new_policy, stable


def policy_iteration(
    mdp: CarRentalMDP,
    gamma: float = 0.9,
    theta: float = 1e-6,
    max_outer: int = 50,
):
    policy: CarRentalPolicy = {state: 0 for state in mdp.states()}
    history: list[tuple[np.ndarray, CarRentalPolicy]] = []

    for _ in range(max_outer):
        V, _ = policy_evaluation(mdp, policy, gamma, theta)
        new_policy, stable = policy_improvement(mdp, V, policy, gamma)
        history.append((V.copy(), dict(policy)))
        policy = new_policy
        if stable:
            break

    return V, policy, history


def value_iteration(
    mdp: CarRentalMDP,
    gamma: float = 0.9,
    theta: float = 1e-6,
    max_iters: int = 10_000,
):
    V = zeros_V(mdp)

    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()

        for state in mdp.states():
            v_new = max(q_from_v(mdp, V_old, state, action, gamma) for action in mdp.possible_actions(state))
            delta = max(delta, abs(v_new - V[state]))
            V[state] = v_new

        if delta < theta:
            break

    policy: CarRentalPolicy = {}
    for state in mdp.states():
        best_action = None
        best_q = -np.inf
        for action in mdp.possible_actions(state):
            q = q_from_v(mdp, V, state, action, gamma)
            if q > best_q:
                best_q = q
                best_action = action
        policy[state] = int(best_action) if best_action is not None else 0

    return V, policy, it + 1
