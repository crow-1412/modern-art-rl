# modern-art-rl

Single-GPU (A10 24GB) workflow for 4-player self-play data generation and DPO fine-tuning.

## Quick start

1. Read `docs/SOP_SELFPLAY_DPO_A10.md`.
2. Install dependencies from `requirements.txt`.
3. Generate self-play logs.
4. Build preference pairs.
5. Train with DPO.
6. Evaluate with tournament script.

## Project layout

- `docs/`: environment audit, SOP, schemas
- `src/env/`: sealed-bid auction environment
- `src/agent/`: LLM and baseline agents
- `src/data/`: self-play rollouts and pair building
- `src/train/`: DPO training entrypoint
- `src/eval/`: tournament evaluation
