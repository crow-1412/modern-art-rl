from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


AUCTION_TYPES = ("hidden", "open", "once", "fixed", "double")


@dataclass
class Card:
    artist: str
    auction_type: str

    def to_dict(self) -> Dict[str, str]:
        return {"artist": self.artist, "auction_type": self.auction_type}


@dataclass
class ActionSpec:
    min_bid: int
    max_bid: int


class AuctionEnv:
    """Rule-aligned Modern Art style environment (4 players, 4 seasons).

    Implemented rule points:
    - 4 seasons; highest cash after season 4 wins.
    - Season ends immediately when an artist's 5th card is played.
      The triggering card is counted for season ranking but not auctioned/owned.
    - End-of-season valuation uses rank 30/20/10, tie-broken by left-to-right artist order.
    - Artist value is cumulative value blocks, but only artists in current season top-3 score this season.
    - Owned paintings are sold to bank at season end and then discarded (ownership resets each season).
    - Fixed price cannot exceed auctioneer cash; all bids are cash-bounded.
    - Double auction supports second-card insertion by auctioneer then clockwise players;
      if someone inserts, they become new auctioneer and receive auction revenue.
    """

    def __init__(
        self,
        num_players: int = 4,
        initial_cash: int = 100,
        max_seasons: int = 4,
        artist_order: List[str] | None = None,
        cards_per_season_per_player: int = 8,
        seed: int = 0,
    ) -> None:
        self.num_players = num_players
        self.initial_cash = initial_cash
        self.max_seasons = max_seasons
        self.cards_per_season_per_player = cards_per_season_per_player
        self.player_ids = [f"seat_{i}" for i in range(num_players)]
        self.artist_order = artist_order or ["A", "B", "C", "D", "E"]
        self._rng = random.Random(seed)

        self.cash: Dict[str, float] = {}
        self.hands: Dict[str, List[Card]] = {}
        self.owned_this_season: Dict[str, Dict[str, int]] = {}
        self.played_count: Dict[str, int] = {}
        self.value_blocks: Dict[str, List[int]] = {}

        self.deck: List[Card] = []
        self.season_idx = 0
        self.turn_idx = 0
        self.current_player_idx = 0
        self.history: List[Dict] = []

    def reset(self, seed: int) -> Dict[str, Dict]:
        self._rng.seed(seed)
        self.cash = {pid: float(self.initial_cash) for pid in self.player_ids}
        self.hands = {pid: [] for pid in self.player_ids}
        self.owned_this_season = {pid: {a: 0 for a in self.artist_order} for pid in self.player_ids}
        self.played_count = {a: 0 for a in self.artist_order}
        self.value_blocks = {a: [] for a in self.artist_order}
        self.deck = self._build_deck()

        self.season_idx = 0
        self.turn_idx = 0
        self.current_player_idx = 0
        self.history = []

        self._deal_for_new_season()
        return {pid: self.get_observation(pid) for pid in self.player_ids}

    def _build_deck(self) -> List[Card]:
        # Balanced pool: enough cards for 4 seasons and 5 artists.
        cards: List[Card] = []
        weights = [
            ("hidden", 4),
            ("open", 4),
            ("once", 4),
            ("fixed", 3),
            ("double", 2),
        ]
        bag: List[str] = []
        for t, w in weights:
            bag.extend([t] * w)

        for artist in self.artist_order:
            for _ in range(16):
                cards.append(Card(artist=artist, auction_type=self._rng.choice(bag)))

        self._rng.shuffle(cards)
        return cards

    def _deal_for_new_season(self) -> None:
        if self.season_idx >= self.max_seasons:
            return
        for pid in self.player_ids:
            for _ in range(self.cards_per_season_per_player):
                if self.deck:
                    self.hands[pid].append(self.deck.pop())
        self.owned_this_season = {pid: {a: 0 for a in self.artist_order} for pid in self.player_ids}
        self.played_count = {a: 0 for a in self.artist_order}

    def legal_actions(self, player_id: str, _obs: Dict | None = None) -> List[ActionSpec]:
        max_bid = int(max(self.cash[player_id], 0))
        return [ActionSpec(min_bid=0, max_bid=max_bid)]

    def get_observation(self, player_id: str) -> Dict:
        cards = [c.to_dict() for c in self.hands[player_id]]
        return {
            "player_id": player_id,
            "season_idx": self.season_idx,
            "max_seasons": self.max_seasons,
            "turn_idx": self.turn_idx,
            "auctioneer": self.player_ids[self.current_player_idx] if self.season_idx < self.max_seasons else None,
            "cash": round(self.cash[player_id], 2),
            "hand_size": len(cards),
            "hand": cards,
            "played_count": dict(self.played_count),
            "value_block_sum": {a: int(sum(self.value_blocks[a])) for a in self.artist_order},
            "artist_order": list(self.artist_order),
            # No-talk mode: no private comms channel and no direct opponent cash visibility.
            "mode": "no_talk_hidden_cash",
        }

    def state_hash(self) -> str:
        h = {
            "season": self.season_idx,
            "turn": self.turn_idx,
            "auctioneer": self.current_player_idx,
            "played": self.played_count,
            "cash": {k: round(v, 1) for k, v in self.cash.items()},
            "hand_sizes": {k: len(v) for k, v in self.hands.items()},
        }
        return hashlib.sha256(str(h).encode("utf-8")).hexdigest()[:16]

    def _left_order_from(self, player_id: str) -> List[str]:
        idx = self.player_ids.index(player_id)
        return [self.player_ids[(idx + k) % self.num_players] for k in range(1, self.num_players)]

    def _clamp_bid(self, pid: str, bid: int) -> int:
        return max(0, min(int(bid), int(self.cash[pid])))

    def _artist_rank_top3(self) -> List[str]:
        ranked = sorted(
            self.artist_order,
            key=lambda a: (-self.played_count[a], self.artist_order.index(a)),
        )
        return [a for a in ranked if self.played_count[a] > 0][:3]

    def _settle_season(self) -> Dict:
        top3 = self._artist_rank_top3()
        for i, v in enumerate([30, 20, 10]):
            if i < len(top3):
                self.value_blocks[top3[i]].append(v)

        active_value = {a: 0 for a in self.artist_order}
        for a in top3:
            active_value[a] = int(sum(self.value_blocks[a]))

        payout_by_player = {pid: 0.0 for pid in self.player_ids}
        for pid in self.player_ids:
            payout = 0.0
            for a in self.artist_order:
                payout += self.owned_this_season[pid][a] * active_value[a]
            self.cash[pid] += payout
            payout_by_player[pid] = payout

        # Paintings sold to bank and discarded after settlement.
        self.owned_this_season = {pid: {a: 0 for a in self.artist_order} for pid in self.player_ids}

        summary = {
            "season_idx": self.season_idx,
            "top3": top3,
            "played_count": dict(self.played_count),
            "active_value": active_value,
            "payout_by_player": payout_by_player,
            "cash_after_settlement": {pid: round(v, 2) for pid, v in self.cash.items()},
        }
        self.history.append({"event": "season_settlement", **summary})
        return summary

    def _resolve_single_card_auction(
        self,
        card: Card,
        seller: str,
        actions: Dict[str, Dict],
    ) -> Dict:
        bidders = [pid for pid in self.player_ids if pid != seller]
        auction_type = card.auction_type

        if auction_type == "fixed":
            asked = int(actions.get(seller, {}).get("price", 0))
            fixed_price = self._clamp_bid(seller, asked)  # cannot exceed seller cash
            buyer = None
            for pid in self._left_order_from(seller):
                offered = self._clamp_bid(pid, int(actions.get(pid, {}).get("bid", 0)))
                if offered >= fixed_price:
                    buyer = pid
                    break
            if buyer is None:
                buyer = seller
            self.cash[buyer] -= fixed_price
            self.cash[seller] += fixed_price
            self.owned_this_season[buyer][card.artist] += 1
            return {
                "auction_type": "fixed",
                "seller": seller,
                "winner": buyer,
                "payment": fixed_price,
            }

        # hidden/open/once all resolved as one-shot highest legal bid in this environment.
        bids = {}
        for pid in bidders:
            bid = self._clamp_bid(pid, int(actions.get(pid, {}).get("bid", 0)))
            bids[pid] = bid
        high = max(bids.values()) if bids else 0
        winners = [pid for pid, v in bids.items() if v == high]
        winner = self._rng.choice(winners) if winners else seller
        payment = high
        self.cash[winner] -= payment
        self.cash[seller] += payment
        self.owned_this_season[winner][card.artist] += 1
        return {
            "auction_type": auction_type,
            "seller": seller,
            "winner": winner,
            "payment": payment,
            "bids": bids,
        }

    def _resolve_lot(
        self,
        auctioneer: str,
        actions: Dict[str, Dict],
    ) -> Dict:
        # 1) auctioneer plays one card
        hand = self.hands[auctioneer]
        if not hand:
            return {"event": "skip_no_card", "auctioneer": auctioneer}

        card_idx = int(actions.get(auctioneer, {}).get("play_card_index", 0))
        card_idx = min(max(card_idx, 0), len(hand) - 1)
        first_card = hand.pop(card_idx)

        self.played_count[first_card.artist] += 1
        if self.played_count[first_card.artist] >= 5:
            return {
                "event": "season_end_trigger",
                "trigger_artist": first_card.artist,
                "trigger_card": first_card.to_dict(),
                "auctioned": False,
                "reason": "fifth_card_played",
            }

        if first_card.auction_type != "double":
            auction_result = self._resolve_single_card_auction(first_card, seller=auctioneer, actions=actions)
            return {
                "event": "auction_resolved",
                "first_card": first_card.to_dict(),
                "auction": auction_result,
            }

        # 2) double auction insertion subphase
        insert_order = [auctioneer] + self._left_order_from(auctioneer)
        new_seller = None
        second_card = None
        for pid in insert_order:
            idx = int(actions.get(pid, {}).get("insert_card_index", -1))
            if idx < 0:
                continue
            phand = self.hands[pid]
            if idx >= len(phand):
                continue
            cand = phand[idx]
            if cand.artist != first_card.artist:
                continue
            if cand.auction_type == "double":
                continue
            second_card = phand.pop(idx)
            new_seller = pid
            break

        if second_card is None:
            return {
                "event": "double_no_insert",
                "first_card": first_card.to_dict(),
                "auctioned": False,
                "reason": "no_valid_second_card",
            }

        self.played_count[second_card.artist] += 1
        if self.played_count[second_card.artist] >= 5:
            return {
                "event": "season_end_trigger",
                "trigger_artist": second_card.artist,
                "trigger_card": second_card.to_dict(),
                "auctioned": False,
                "reason": "fifth_card_played_in_double_insert",
                "first_card": first_card.to_dict(),
            }

        auction_result = self._resolve_single_card_auction(second_card, seller=new_seller, actions=actions)
        winner = auction_result["winner"]
        self.owned_this_season[winner][first_card.artist] += 1
        return {
            "event": "double_resolved",
            "first_card": first_card.to_dict(),
            "second_card": second_card.to_dict(),
            "inserter_new_seller": new_seller,
            "auction": auction_result,
            "bundle_owner": winner,
        }

    def _advance_auctioneer(self) -> None:
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        self.turn_idx += 1

    def step(self, actions: Dict[str, Dict]) -> Tuple[Dict[str, Dict], Dict[str, float], bool, Dict]:
        if self.season_idx >= self.max_seasons:
            return (
                {pid: self.get_observation(pid) for pid in self.player_ids},
                {pid: 0.0 for pid in self.player_ids},
                True,
                {"final_cash": dict(self.cash)},
            )

        before = dict(self.cash)
        auctioneer = self.player_ids[self.current_player_idx]
        lot_result = self._resolve_lot(auctioneer=auctioneer, actions=actions)

        season_ended = lot_result.get("event") == "season_end_trigger"
        season_summary = None
        if season_ended:
            season_summary = self._settle_season()
            self.season_idx += 1
            if self.season_idx < self.max_seasons:
                self._deal_for_new_season()
        self._advance_auctioneer()

        rewards = {pid: self.cash[pid] - before[pid] for pid in self.player_ids}
        done = self.season_idx >= self.max_seasons
        info = {
            "lot_result": lot_result,
            "season_ended": season_ended,
            "season_summary": season_summary,
            "legal_flags": {pid: True for pid in self.player_ids},
            "final_cash": dict(self.cash) if done else None,
        }
        obs = {pid: self.get_observation(pid) for pid in self.player_ids}
        return obs, rewards, done, info
