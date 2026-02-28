from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent.llm_agent import HeuristicAgent, LLMSeatAgent, RandomAgent
from src.env.auction_env import AuctionEnv


def build_policy(name: str, model_path: str | None):
    if name == "heuristic":
        return HeuristicAgent(aggression=0.8)
    if name == "random":
        return RandomAgent(seed=0)
    if name == "llm":
        if not model_path:
            raise ValueError("--model-path-a/--model-path-b required when policy is llm")
        return LLMSeatAgent(model_path=model_path, load_in_4bit=False)
    raise ValueError(f"Unknown policy: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tournament between two policies + two heuristic bots.")
    parser.add_argument("--policy-a", choices=["llm", "heuristic", "random"], required=True)
    parser.add_argument("--policy-b", choices=["llm", "heuristic", "random"], required=True)
    parser.add_argument("--model-path-a", default="")
    parser.add_argument("--model-path-b", default="")
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--report-path", required=True)
    args = parser.parse_args()

    env = AuctionEnv(total_rounds=args.rounds)
    agent_a = build_policy(args.policy_a, args.model_path_a or None)
    agent_b = build_policy(args.policy_b, args.model_path_b or None)
    bot_c = HeuristicAgent(aggression=0.6)
    bot_d = HeuristicAgent(aggression=0.95)

    wins = {"seat_0": 0, "seat_1": 0, "seat_2": 0, "seat_3": 0}
    bankroll_sum = {k: 0.0 for k in wins}

    for game_idx in range(args.num_games):
        obs = env.reset(seed=args.seed + game_idx)
        done = False
        while not done:
            actions = {}
            for seat_id, agent in {
                "seat_0": agent_a,
                "seat_1": agent_b,
                "seat_2": bot_c,
                "seat_3": bot_d,
            }.items():
                spec = env.legal_actions(seat_id, obs[seat_id])[0]
                dec = agent.act(obs[seat_id], spec.min_bid, spec.max_bid, seat_id)
                actions[seat_id] = dec.bid
            obs, _rewards, done, info = env.step(actions)

        final_bankrolls = info["final_bankrolls"]
        winner = max(final_bankrolls, key=lambda k: final_bankrolls[k])
        wins[winner] += 1
        for k, v in final_bankrolls.items():
            bankroll_sum[k] += v

    report = {
        "num_games": args.num_games,
        "avg_bankroll": {k: bankroll_sum[k] / args.num_games for k in bankroll_sum},
        "wins": wins,
        "win_rate": {k: wins[k] / args.num_games for k in wins},
        "mapping": {
            "seat_0": args.policy_a,
            "seat_1": args.policy_b,
            "seat_2": "heuristic_0.6",
            "seat_3": "heuristic_0.95",
        },
    }

    out = Path(args.report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
