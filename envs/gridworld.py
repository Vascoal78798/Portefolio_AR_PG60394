from __future__ import annotations

from dataclasses import dataclass

from core.base import Environment

GridworldState = tuple[int, int]
GridworldAction = str

ACTIONS: tuple[GridworldAction, ...] = ("U", "D", "L", "R")
ACTION_TO_DELTA: dict[GridworldAction, tuple[int, int]] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


@dataclass(frozen=True)
class Gridworld(Environment[GridworldState, GridworldAction]):
    n_rows: int = 4
    n_cols: int = 4
    terminal_states: tuple[GridworldState, ...] = ((0, 0), (3, 3))
    step_reward: float = -1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "_state", (0, 0))

    def states(self) -> list[GridworldState]:
        return [(row, col) for row in range(self.n_rows) for col in range(self.n_cols)]

    def is_terminal(self, state: GridworldState) -> bool:
        return state in self.terminal_states

    def reset(self) -> GridworldState:
        object.__setattr__(self, "_state", (0, 0))
        return self._state

    def available_actions(self, state: GridworldState) -> list[GridworldAction]:
        if self.is_terminal(state):
            return []
        return list(ACTIONS)

    def transition(
        self,
        state: GridworldState,
        action: GridworldAction,
    ) -> tuple[GridworldState, float, bool]:
        if self.is_terminal(state):
            return state, 0.0, True

        if action not in ACTION_TO_DELTA:
            raise ValueError(f"Unknown action: {action}. Valid actions: {ACTIONS}")

        d_row, d_col = ACTION_TO_DELTA[action]
        row, col = state
        next_row = min(max(row + d_row, 0), self.n_rows - 1)
        next_col = min(max(col + d_col, 0), self.n_cols - 1)
        next_state = (next_row, next_col)
        reward = self.step_reward
        done = self.is_terminal(next_state)
        return next_state, reward, done

    def step(self, action: GridworldAction) -> tuple[GridworldState, float, bool]:
        next_state, reward, done = self.transition(self._state, action)
        object.__setattr__(self, "_state", next_state)
        return next_state, reward, done
