from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class EpisodeLog:
    episode_id: str
    t: int
    seat_id: str
    state_hash: str
    prompt: str
    raw_output: str
    parsed_action: Dict[str, Any]
    legal: bool
    reward_delta: float
    bankroll_after: float
    done: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreferencePair:
    pair_id: str
    state_hash: str
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    margin: float
    source_episode_ids: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
