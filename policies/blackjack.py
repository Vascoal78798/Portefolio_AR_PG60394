from __future__ import annotations

from core.base import Policy
from envs.blackjack import BlackjackAction, BlackjackState


class ThresholdPolicy(Policy[BlackjackState, BlackjackAction]):
    def __init__(self, threshold: int = 20):
        self.threshold = threshold

    def select_action(self, state: BlackjackState) -> BlackjackAction:
        player_sum, _, _ = state
        return "hit" if player_sum < self.threshold else "stick"
