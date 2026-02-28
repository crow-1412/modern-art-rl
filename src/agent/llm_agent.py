from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AgentDecision:
    bid: int
    raw_output: str
    legal: bool


class BaseAgent:
    def act(self, obs: Dict, min_bid: int, max_bid: int, seat_id: str) -> AgentDecision:
        raise NotImplementedError


class HeuristicAgent(BaseAgent):
    def __init__(self, aggression: float = 0.8):
        self.aggression = aggression

    def act(self, obs: Dict, min_bid: int, max_bid: int, seat_id: str) -> AgentDecision:
        estimate = float(obs["public_estimate"])
        bankroll = float(obs["bankroll"])
        target = int(min(estimate * self.aggression, bankroll))
        bid = min(max(target, min_bid), max_bid)
        raw = json.dumps({"bid": bid, "reason": "heuristic"}, ensure_ascii=True)
        return AgentDecision(bid=bid, raw_output=raw, legal=True)


class RandomAgent(BaseAgent):
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, obs: Dict, min_bid: int, max_bid: int, seat_id: str) -> AgentDecision:
        bid = self.rng.randint(min_bid, max_bid)
        raw = json.dumps({"bid": bid, "reason": "random"}, ensure_ascii=True)
        return AgentDecision(bid=bid, raw_output=raw, legal=True)


class LLMSeatAgent(BaseAgent):
    """Single base model reused by all seats with seat-specific prompts."""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 96,
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
            "You are an auction agent. Return valid JSON only with keys: bid, reasoning. "
            "Bid must be an integer in legal range."
        )
        payload = {
            "seat_id": seat_id,
            "observation": obs,
            "legal_bid_range": [min_bid, max_bid],
            "task": "Choose one sealed bid for this round.",
        }
        return f"{policy}\nInput: {json.dumps(payload, ensure_ascii=True)}\nOutput:"  # ASCII-only payload

    @staticmethod
    def _extract_bid(text: str, min_bid: int, max_bid: int) -> int:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "bid" in obj:
                return int(obj["bid"])
        except Exception:
            pass
        # Fallback: first integer in output
        match = re.search(r"-?\d+", text)
        if not match:
            return min_bid
        bid = int(match.group(0))
        return min(max(bid, min_bid), max_bid)

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
        bid = self._extract_bid(raw_output, min_bid=min_bid, max_bid=max_bid)
        legal = min_bid <= bid <= max_bid
        return AgentDecision(bid=bid, raw_output=raw_output, legal=legal)

    def build_prompt(self, obs: Dict, min_bid: int, max_bid: int, seat_id: str) -> str:
        return self._prompt(obs, min_bid, max_bid, seat_id)
