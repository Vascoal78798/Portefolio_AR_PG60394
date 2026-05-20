from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mdps.base import TabularMDP

CarRentalState = tuple[int, int]
CarRentalAction = int
CarRentalPolicy = dict[CarRentalState, CarRentalAction]


@dataclass(frozen=True)
class CarRentalParams:
    max_cars_1: int = 20
    max_cars_2: int = 20
    max_moveable: int = 5
    revenue_per_rental: float = 10.0
    cost_per_moved: float = 2.0
    lambdas: tuple[float, float, float, float] = (3.0, 4.0, 3.0, 2.0)
    max_requests_1: int = 8
    max_requests_2: int = 10
    max_returns_1: int = 8
    max_returns_2: int = 8


def poisson_pmf_truncated(lam: float, max_k: int) -> np.ndarray:
    probs = np.zeros(max_k + 1, dtype=float)
    probs[0] = np.exp(-lam)
    for k in range(1, max_k):
        probs[k] = probs[k - 1] * lam / k
    probs[max_k] = max(0.0, 1.0 - probs[:max_k].sum())
    probs /= probs.sum()
    return probs


class CarRentalMDP(TabularMDP[CarRentalState, CarRentalAction]):
    def __init__(self, params: CarRentalParams):
        self.params = params
        self.req1 = poisson_pmf_truncated(params.lambdas[0], params.max_requests_1)
        self.req2 = poisson_pmf_truncated(params.lambdas[1], params.max_requests_2)
        self.ret1 = poisson_pmf_truncated(params.lambdas[2], params.max_returns_1)
        self.ret2 = poisson_pmf_truncated(params.lambdas[3], params.max_returns_2)
        self._loc_cache: dict[tuple[int, int], tuple[np.ndarray, float]] = {}

    def states(self) -> list[CarRentalState]:
        return [
            (n1, n2)
            for n1 in range(self.params.max_cars_1 + 1)
            for n2 in range(self.params.max_cars_2 + 1)
        ]

    def is_terminal(self, state: CarRentalState) -> bool:
        return False

    def possible_actions(self, state: CarRentalState) -> list[CarRentalAction]:
        n1, n2 = state
        a_min = -min(self.params.max_moveable, n2, self.params.max_cars_1 - n1)
        a_max = min(self.params.max_moveable, n1, self.params.max_cars_2 - n2)
        return list(range(a_min, a_max + 1))

    def after_move(self, state: CarRentalState, action: CarRentalAction) -> CarRentalState:
        n1, n2 = state
        return n1 - action, n2 + action

    def _loc_outcomes(self, loc_id: int, cars_after_move: int) -> tuple[np.ndarray, float]:
        key = (loc_id, cars_after_move)
        if key in self._loc_cache:
            return self._loc_cache[key]

        if loc_id == 1:
            req = self.req1
            ret = self.ret1
            cap = self.params.max_cars_1
        else:
            req = self.req2
            ret = self.ret2
            cap = self.params.max_cars_2

        p_next = np.zeros(cap + 1, dtype=float)
        exp_rented = 0.0

        for k_req, p_req in enumerate(req):
            rented = min(cars_after_move, k_req)
            exp_rented += p_req * rented
            cars_left = cars_after_move - rented

            for k_ret, p_ret in enumerate(ret):
                next_cars = min(cap, cars_left + k_ret)
                p_next[next_cars] += p_req * p_ret

        p_next /= p_next.sum()
        self._loc_cache[key] = (p_next, exp_rented)
        return p_next, exp_rented

    def expected_transition(
        self,
        state: CarRentalState,
        action: CarRentalAction,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if action not in self.possible_actions(state):
            raise ValueError("Illegal action")

        n1_moved, n2_moved = self.after_move(state, action)
        p_next_1, exp_rent1 = self._loc_outcomes(1, n1_moved)
        p_next_2, exp_rent2 = self._loc_outcomes(2, n2_moved)
        exp_revenue = (exp_rent1 + exp_rent2) * self.params.revenue_per_rental
        return p_next_1, p_next_2, exp_revenue

    def transitions(
        self,
        state: CarRentalState,
        action: CarRentalAction,
    ):
        p_next_1, p_next_2, exp_revenue = self.expected_transition(state, action)
        move_cost = self.params.cost_per_moved * abs(action)
        reward = exp_revenue - move_cost

        for n1, p1 in enumerate(p_next_1):
            if p1 == 0.0:
                continue
            for n2, p2 in enumerate(p_next_2):
                if p2 == 0.0:
                    continue
                yield p1 * p2, (n1, n2), reward, False
