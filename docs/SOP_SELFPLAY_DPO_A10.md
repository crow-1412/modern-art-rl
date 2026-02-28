# SOP: A10 Single-GPU Self-Play + DPO (8B train, 14B judge)

## 0. Scope
This SOP runs a minimal but executable pipeline:
1) generate self-play logs, 2) build DPO pairs, 3) train QLoRA-DPO, 4) evaluate tournament.

## 1. Pre-flight checks

### 1.1 Free GPU memory (required)
```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
docker stop next-edit
nvidia-smi
```
Acceptance: free memory should be enough for 8B inference/training (target >20GB free before training).

### 1.2 Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.3 Smoke import
```bash
python3 -c "import torch, transformers, trl, peft, datasets, bitsandbytes; print('ok')"
```

## 2. Generate self-play data

### 2.1 Fast smoke run (heuristic)
```bash
PYTHONPATH=. python3 src/data/generate_selfplay.py \
  --episodes 20 \
  --rounds 8 \
  --agent heuristic \
  --out data/raw/episode_logs_smoke.jsonl
```

### 2.2 Main run (LLM self-play)
```bash
PYTHONPATH=. python3 src/data/generate_selfplay.py \
  --episodes 2000 \
  --rounds 12 \
  --agent llm \
  --model-path /root/Qwen3-8B \
  --out data/raw/episode_logs_main.jsonl
```
Acceptance:
- file exists and non-empty
- parse success rate >= 98%
- illegal action rate <= 1%

## 3. Build DPO preference pairs

```bash
PYTHONPATH=. python3 src/data/build_pairs.py \
  --in data/raw/episode_logs_main.jsonl \
  --out data/processed/preference_pairs_main.jsonl \
  --margin 0.1
```
Acceptance:
- output exists
- each row has `chosen_score > rejected_score`
- pair count is sufficient for first run (recommended >= 20k)

## 4. Split dataset (train/eval)
```bash
PYTHONPATH=. python3 src/data/split_pairs.py \
  --in data/processed/preference_pairs_main.jsonl \
  --train-out data/processed/preference_pairs_train.jsonl \
  --eval-out data/processed/preference_pairs_eval.jsonl \
  --eval-ratio 0.1 \
  --seed 42
```

## 5. Train DPO (QLoRA)

```bash
PYTHONPATH=. python3 src/train/train_dpo.py \
  --model-path /root/Qwen3-8B \
  --train-pairs data/processed/preference_pairs_train.jsonl \
  --eval-pairs data/processed/preference_pairs_eval.jsonl \
  --output-dir outputs/dpo-qwen3-8b-r16 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lr 5e-6 \
  --epochs 1 \
  --batch 1 \
  --grad-acc 16 \
  --max-len 1024 \
  --max-prompt-len 768 \
  --beta 0.1 \
  --load-in-4bit
```
Acceptance:
- no OOM crash
- adapter checkpoint saved under `output-dir`

## 6. Evaluate tournament

```bash
PYTHONPATH=. python3 src/eval/eval_tournament.py \
  --policy-a llm \
  --model-path-a /root/Qwen3-8B \
  --policy-b heuristic \
  --num-games 500 \
  --report-path reports/tournament_baseline.json
```

For post-training checkpoint, point `--model-path-a` to your merged/adapted model.

Acceptance:
- compare win-rate and avg bankroll vs baseline run
- reject checkpoint if illegal-action rate increases materially

## 7. Use 14B as offline judge
14B is not in training loop. Use it only to re-score ambiguous samples (`margin` near threshold) or to perform periodic quality checks.

## 8. End-of-run operations
```bash
# If needed, restore online inference container
docker start next-edit
```
