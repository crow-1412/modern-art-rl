from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent.llm_agent import HeuristicAgent, LLMSeatAgent, RandomAgent
from src.env.auction_env import AuctionEnv


ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT / "web" / "static"
ASSETS_DIR = ROOT / "web" / "assets"


class NewGameRequest(BaseModel):
    seed: int = 42
    policy_map: Dict[str, str] | None = None
    model_path: str = "/root/Qwen3-8B"


@dataclass
class GameSession:
    env: AuctionEnv
    agents: Dict[str, object]
    last_obs: Dict[str, Dict]
    done: bool
    events: List[Dict]


APP_STATE: Dict[str, GameSession] = {}
SESSION_KEY = "default"


def build_agent(kind: str, seed: int, model_path: str):
    if kind == "heuristic":
        return HeuristicAgent(aggression=0.8, seed=seed)
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "llm":
        return LLMSeatAgent(model_path=model_path, load_in_4bit=False)
    raise ValueError(f"unknown policy: {kind}")


def default_policy_map() -> Dict[str, str]:
    return {
        "seat_0": "heuristic",
        "seat_1": "heuristic",
        "seat_2": "random",
        "seat_3": "heuristic",
    }


def run_single_step(session: GameSession) -> Dict:
    env = session.env
    actions = {}
    for pid in env.player_ids:
        obs = session.last_obs[pid]
        spec = env.legal_actions(pid, obs)[0]
        dec = session.agents[pid].act(obs, spec.min_bid, spec.max_bid, pid)
        actions[pid] = dec.action

    obs, rewards, done, info = env.step(actions)
    session.last_obs = obs
    session.done = done

    event = {
        "turn_idx": env.turn_idx,
        "season_idx": env.season_idx,
        "actions": actions,
        "rewards": rewards,
        "info": info,
    }
    session.events.append(event)
    return event


def serialize_state(session: GameSession) -> Dict:
    env = session.env
    public = {
        "season_idx": env.season_idx,
        "max_seasons": env.max_seasons,
        "turn_idx": env.turn_idx,
        "auctioneer": env.player_ids[env.current_player_idx] if env.season_idx < env.max_seasons else None,
        "played_count": dict(env.played_count),
        "value_block_sum": {a: int(sum(env.value_blocks[a])) for a in env.artist_order},
        "artist_order": list(env.artist_order),
        "done": session.done,
    }
    seats = {}
    for pid in env.player_ids:
        seats[pid] = {
            "cash": round(env.cash[pid], 2),
            "hand_size": len(env.hands[pid]),
            "owned_this_season": dict(env.owned_this_season[pid]),
        }
    return {
        "public": public,
        "seats": seats,
        "events": session.events[-30:],
    }


app = FastAPI(title="Modern Art RL Viewer")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/new_game")
def new_game(req: NewGameRequest):
    policy_map = req.policy_map or default_policy_map()
    env = AuctionEnv(seed=req.seed)
    obs = env.reset(seed=req.seed)
    agents = {
        pid: build_agent(policy_map.get(pid, "heuristic"), seed=req.seed + i, model_path=req.model_path)
        for i, pid in enumerate(env.player_ids)
    }
    APP_STATE[SESSION_KEY] = GameSession(env=env, agents=agents, last_obs=obs, done=False, events=[])
    return serialize_state(APP_STATE[SESSION_KEY])


@app.post("/api/step")
def step_once():
    session = APP_STATE.get(SESSION_KEY)
    if not session:
        raise HTTPException(status_code=400, detail="No active game. Call /api/new_game first.")
    if session.done:
        return serialize_state(session)
    run_single_step(session)
    return serialize_state(session)


@app.post("/api/auto_play")
def auto_play(max_steps: int = 100):
    session = APP_STATE.get(SESSION_KEY)
    if not session:
        raise HTTPException(status_code=400, detail="No active game. Call /api/new_game first.")
    steps = 0
    while (not session.done) and steps < max_steps:
        run_single_step(session)
        steps += 1
    return serialize_state(session)


@app.get("/api/state")
def get_state():
    session = APP_STATE.get(SESSION_KEY)
    if not session:
        raise HTTPException(status_code=400, detail="No active game. Call /api/new_game first.")
    return serialize_state(session)


@app.get("/api/assets_manifest")
def assets_manifest():
    artists = {}
    players = {}
    for artist in ["A", "B", "C", "D", "E"]:
        p = ASSETS_DIR / "generated" / "artists" / f"{artist}.png"
        artists[artist] = f"/assets/generated/artists/{artist}.png" if p.exists() else None
    for i in range(4):
        p = ASSETS_DIR / "generated" / "players" / f"seat_{i}.png"
        players[f"seat_{i}"] = f"/assets/generated/players/seat_{i}.png" if p.exists() else None
    return {"artists": artists, "players": players}
