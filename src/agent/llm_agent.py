from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class AgentDecision:
    action: Dict[str, int]
    raw_output: str
    legal: bool


class BaseAgent:
    def act(self, obs: Dict, min_bid: int, max_bid: int, seat_id: str) -> AgentDecision:
        raise NotImplementedError


class HeuristicAgent(BaseAgent):
    def __init__(self, aggression: float = 0.8, seed: int = 0):
        self.aggression = aggression
        self.rng = random.Random(seed)

    def _pick_insert_index(self, obs: Dict, target_artist: str) -> int:
        for i, card in enumerate(obs.get("hand", [])):
            if card["artist"] == target_artist and card["auction_type"] != "double":
                return i
        return -1

    def act(self, obs: Dict, min_bid: int, max_bid: int, seat_id: str) -> AgentDecision:
        bankroll = float(obs["cash"])
        hand = obs.get("hand", [])
        play_idx = 0 if hand else -1
        price = int(min(max_bid, bankroll * 0.6))
        bid = int(min(max_bid, bankroll * self.aggression))

        # Opportunistically prepare a legal insert candidate for double auctions.
        target_artist = hand[play_idx]["artist"] if hand and play_idx >= 0 else ""
        insert_idx = self._pick_insert_index(obs, target_artist)

        action = {
            "play_card_index": play_idx,
            "price": max(0, price),
            "bid": max(0, bid),
            "insert_card_index": insert_idx,
        }
        raw = json.dumps({"action": action, "reason": "heuristic"}, ensure_ascii=True)
        return AgentDecision(action=action, raw_output=raw, legal=True)


class RandomAgent(BaseAgent):
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, obs: Dict, min_bid: int, max_bid: int, seat_id: str) -> AgentDecision:
        hand = obs.get("hand", [])
        play_idx = self.rng.randint(0, max(0, len(hand) - 1)) if hand else -1
        action = {
            "play_card_index": play_idx,
            "price": self.rng.randint(min_bid, max_bid),
            "bid": self.rng.randint(min_bid, max_bid),
            "insert_card_index": -1,
        }
        raw = json.dumps({"action": action, "reason": "random"}, ensure_ascii=True)
        return AgentDecision(action=action, raw_output=raw, legal=True)


class LLMSeatAgent(BaseAgent):
    """Single base model reused by all seats with seat-specific prompts."""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self._loaded = False
        self._tokenizer = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        dtype_obj = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16
        quant_cfg = None
        if self.load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype_obj,
                bnb_4bit_use_double_quant=True,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=dtype_obj,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quant_cfg,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._loaded = True

    @staticmethod
    def _prompt(obs: Dict, min_bid: int, max_bid: int, seat_id: str) -> str:
        policy = (
            "You are a Modern-Art auction player. Output strict JSON only with keys "
            "play_card_index, price, bid, insert_card_index, reasoning."
        )
        payload = {
            "seat_id": seat_id,
            "observation": obs,
            "constraints": {
                "bid_min": min_bid,
                "bid_max": max_bid,
                "fixed_price_must_not_exceed_cash": True,
                "insert_card_must_match_artist_and_non_double": True,
            },
        }
        return f"{policy}\nInput: {json.dumps(payload, ensure_ascii=True)}\nOutput:"  # ASCII payload

    @staticmethod
    def _extract_int(obj: Dict[str, Any], key: str, default: int) -> int:
        try:
            return int(obj.get(key, default))
        except Exception:
            return default

    def _extract_action(self, text: str, min_bid: int, max_bid: int, hand_size: int) -> Dict[str, int]:
        parsed: Dict[str, Any] = {}
        try:
            maybe = json.loads(text)
            if isinstance(maybe, dict):
                parsed = maybe
                if "action" in maybe and isinstance(maybe["action"], dict):
                    parsed = maybe["action"]
        except Exception:
            # fallback: first number as bid
            match = re.search(r"-?\d+", text)
            bid = int(match.group(0)) if match else 0
            return {
                "play_card_index": 0 if hand_size > 0 else -1,
                "price": max(0, min(max_bid, bid)),
                "bid": max(0, min(max_bid, bid)),
                "insert_card_index": -1,
            }

        play_idx = self._extract_int(parsed, "play_card_index", 0 if hand_size > 0 else -1)
        if hand_size == 0:
            play_idx = -1
        else:
            play_idx = min(max(play_idx, 0), hand_size - 1)

        price = max(0, min(max_bid, self._extract_int(parsed, "price", 0)))
        bid = max(0, min(max_bid, self._extract_int(parsed, "bid", 0)))
        insert_idx = self._extract_int(parsed, "insert_card_index", -1)
        if insert_idx >= hand_size:
            insert_idx = -1

        return {
            "play_card_index": play_idx,
            "price": price,
            "bid": bid,
            "insert_card_index": insert_idx,
        }

    def act(self, obs: Dict, min_bid: int, max_bid: int, seat_id: str) -> AgentDecision:
        self._lazy_load()
        prompt = self._prompt(obs, min_bid, max_bid, seat_id)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
        )
        full_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_output = full_text[len(prompt) :].strip() if full_text.startswith(prompt) else full_text
        action = self._extract_action(raw_output, min_bid, max_bid, hand_size=int(obs.get("hand_size", 0)))
        legal = True
        return AgentDecision(action=action, raw_output=raw_output, legal=legal)

    def build_prompt(self, obs: Dict, min_bid: int, max_bid: int, seat_id: str) -> str:
        return self._prompt(obs, min_bid, max_bid, seat_id)
