# No-Talk Frontend Viewer

## What it does
- Runs a full 4-player game in no-talk mode.
- Shows public board progress (season, turn, played count, value blocks).
- Shows each seat status (cash, hand size, seasonal holdings).
- Shows latest lot resolution events.

## Start
```bash
pip install -r requirements.txt
make gen-assets
make ui
```
Then open `http://localhost:8000`.

## Asset generation
`make gen-assets` reads `.env`:
- `GOOGLE_GEMINI_BASE_URL`
- `GEMINI_API_KEY`
- `GEMINI_MODEL`

It tries `POST {BASE_URL}/v1/images/generations`.
If the endpoint is unavailable, placeholder images are generated automatically.

## No-talk guarantee (env side)
Per-seat observation does not include opponent cash and does not include any message field.
