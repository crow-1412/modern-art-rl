from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT_ARTISTS = ROOT / "web" / "assets" / "generated" / "artists"
OUT_PLAYERS = ROOT / "web" / "assets" / "generated" / "players"


def ensure_dirs() -> None:
    OUT_ARTISTS.mkdir(parents=True, exist_ok=True)
    OUT_PLAYERS.mkdir(parents=True, exist_ok=True)


def draw_placeholder(path: Path, label: str) -> None:
    img = Image.new("RGB", (512, 512), color=(44, 66, 96))
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 492, 492), outline=(120, 180, 255), width=4)
    draw.text((40, 240), label, fill=(235, 240, 255))
    img.save(path)


def call_image_api(base_url: str, api_key: str, model: str, prompt: str) -> bytes | None:
    url = base_url.rstrip("/") + "/v1/images/generations"
    payload = {
        "model": model,
        "prompt": prompt,
        "size": "1024x1024",
        "response_format": "b64_json",
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", [])
        if not data:
            return None
        item = data[0]
        if "b64_json" in item:
            return base64.b64decode(item["b64_json"])
        if "url" in item:
            r = requests.get(item["url"], timeout=120)
            if r.status_code == 200:
                return r.content
    except Exception:
        return None
    return None


def write_image(path: Path, content: bytes | None, fallback_label: str) -> None:
    if content:
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
            image.save(path)
            return
        except Exception:
            pass
    draw_placeholder(path, fallback_label)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate artist/player images for frontend.")
    parser.add_argument("--model", default="", help="Override model name (default from .env GEMINI_MODEL)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    base_url = os.getenv("GOOGLE_GEMINI_BASE_URL", "").strip()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = args.model.strip() or os.getenv("GEMINI_MODEL", "nano-banana")

    ensure_dirs()

    artists = {
        "A": "A modern art painting style icon for artist A, abstract geometric, clean composition",
        "B": "A modern art painting style icon for artist B, bold brushstroke, vibrant colors",
        "C": "A modern art painting style icon for artist C, minimalist shapes, museum style",
        "D": "A modern art painting style icon for artist D, surreal modern painting look",
        "E": "A modern art painting style icon for artist E, pop art influence, collectible look",
    }

    players = {
        "seat_0": "Portrait avatar for board game player seat_0, stylish, neutral background",
        "seat_1": "Portrait avatar for board game player seat_1, stylish, neutral background",
        "seat_2": "Portrait avatar for board game player seat_2, stylish, neutral background",
        "seat_3": "Portrait avatar for board game player seat_3, stylish, neutral background",
    }

    for artist, prompt in artists.items():
        img_bytes = None
        if base_url and api_key:
            img_bytes = call_image_api(base_url, api_key, model, prompt)
        write_image(OUT_ARTISTS / f"{artist}.png", img_bytes, f"Artist {artist}")

    for player, prompt in players.items():
        img_bytes = None
        if base_url and api_key:
            img_bytes = call_image_api(base_url, api_key, model, prompt)
        write_image(OUT_PLAYERS / f"{player}.png", img_bytes, player)

    print(f"assets ready under {ROOT / 'web' / 'assets' / 'generated'}")


if __name__ == "__main__":
    main()
