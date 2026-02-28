PYTHONPATH := .

.PHONY: audit smoke-selfplay build-pairs split-pairs smoke-eval gen-assets ui

audit:
	python3 scripts/audit_env.py

smoke-selfplay:
	PYTHONPATH=$(PYTHONPATH) python3 src/data/generate_selfplay.py --episodes 10 --agent heuristic --out data/raw/episode_logs_smoke.jsonl

build-pairs:
	PYTHONPATH=$(PYTHONPATH) python3 src/data/build_pairs.py --in data/raw/episode_logs_smoke.jsonl --out data/processed/preference_pairs_smoke.jsonl --margin 0.1

split-pairs:
	PYTHONPATH=$(PYTHONPATH) python3 src/data/split_pairs.py --in data/processed/preference_pairs_smoke.jsonl --train-out data/processed/preference_pairs_train.jsonl --eval-out data/processed/preference_pairs_eval.jsonl

smoke-eval:
	PYTHONPATH=$(PYTHONPATH) python3 src/eval/eval_tournament.py --policy-a heuristic --policy-b random --num-games 20 --report-path reports/tournament_smoke.json

gen-assets:
	python3 scripts/generate_visual_assets.py

ui:
	PYTHONPATH=$(PYTHONPATH) uvicorn src.frontend.app:app --host 0.0.0.0 --port 8000 --reload
