from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ActionSpec:
    min_bid: int
    max_bid: int


class AuctionEnv:
    """Minimal 4-player sealed-bid environment.

    Each round samples a hidden true value and public estimate.
    Highest valid bid wins, then receives settlement (true_value - bid).
    """

    def __init__(
        self,
        num_players: int = 4,
        initial_bankroll: int = 100,
        total_rounds: int = 12,
        value_low: int = 20,
        value_high: int = 100,
        estimate_noise: int = 15,
    ) -> None:
        self.num_players = num_players
        self.initial_bankroll = initial_bankroll
        self.total_rounds = total_rounds
        self.value_low = value_low
        self.value_high = value_high
        self.estimate_noise = estimate_noise
        self.player_ids = [f"seat_{i}" for i in range(num_players)]

        self._rng = random.Random()
        self.round_idx = 0
        self.bankrolls: Dict[str, float] = {}
        self.history: List[Dict[str, float]] = []
        self.current_value = 0
        self.current_estimate = 0

    def reset(self, seed: int) -> Dict[str, Dict]:
        self._rng.seed(seed)
        self.round_idx = 0
        self.bankrolls = {pid: float(self.initial_bankroll) for pid in self.player_ids}
        self.history = []
        self._sample_round()
        return {pid: self.get_observation(pid) for pid in self.player_ids}

    def _sample_round(self) -> None:
        self.current_value = self._rng.randint(self.value_low, self.value_high)
        noise = self._rng.randint(-self.estimate_noise, self.estimate_noise)
        self.current_estimate = max(1, self.current_value + noise)

    def legal_actions(self, player_id: str, _obs: Dict | None = None) -> List[ActionSpec]:
        max_bid = int(max(self.bankrolls[player_id], 0))
        return [ActionSpec(min_bid=0, max_bid=max_bid)]

    def get_observation(self, player_id: str) -> Dict:
        recent = self.history[-3:]
        recent_winning_bid = sum(item["winning_bid"] for item in recent) / len(recent) if recent else 0.0
        return {
            "player_id": player_id,
            "round_idx": self.round_idx,
            "total_rounds": self.total_rounds,
            "bankroll": round(self.bankrolls[player_id], 2),
            "public_estimate": self.current_estimate,
            "recent_avg_winning_bid": round(recent_winning_bid, 2),
            "recent_avg_profit": round(sum(item["winner_profit"] for item in recent) / len(recent), 2)
            if recent
            else 0.0,
        }

    def state_hash(self) -> str:
        state_str = f"{self.round_idx}|{self.current_estimate}|{[(pid, round(self.bankrolls[pid],2)) for pid in self.player_ids]}"
        return hashlib.sha256(state_str.encode("utf-8")).hexdigest()[:16]

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, Dict], Dict[str, float], bool, Dict]:
        legal_actions = {pid: self.legal_actions(pid)[0] for pid in self.player_ids}
        clipped = {}
        illegal_flags = {}
        for pid in self.player_ids:
            bid = int(actions.get(pid, 0))
            spec = legal_actions[pid]
            legal = spec.min_bid <= bid <= spec.max_bid
            illegal_flags[pid] = not legal
            clipped[pid] = min(max(bid, spec.min_bid), spec.max_bid)

        highest = max(clipped.values())
        candidates = [pid for pid, bid in clipped.items() if bid == highest]
        winner = self._rng.choice(candidates)
        winning_bid = clipped[winner]

        rewards = {pid: 0.0 for pid in self.player_ids}
        winner_profit = float(self.current_value - winning_bid)
        rewards[winner] = winner_profit

        for pid in self.player_ids:
            self.bankrolls[pid] += rewards[pid]

        self.history.append(
            {
                "round": self.round_idx,
                "true_value": self.current_value,
                "public_estimate": self.current_estimate,
                "winner": winner,
                "winning_bid": winning_bid,
                "winner_profit": winner_profit,
            }
        )

        self.round_idx += 1
        done = self.round_idx >= self.total_rounds
        info = {
            "winner": winner,
            "winning_bid": winning_bid,
            "true_value": self.current_value,
            "illegal_flags": illegal_flags,
            "final_bankrolls": dict(self.bankrolls) if done else None,
        }

        if not done:
            self._sample_round()
            next_obs = {pid: self.get_observation(pid) for pid in self.player_ids}
        else:
            next_obs = {pid: self.get_observation(pid) for pid in self.player_ids}

        return next_obs, rewards, done, info
