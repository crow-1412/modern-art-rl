# Environment Audit (2026-02-28)

## Host
- OS: Ubuntu 22.04.4 LTS
- Kernel: 5.15.0-91-generic
- Python: 3.10.12
- RAM: 54GiB (swap disabled)
- Disk: ~117GiB free on `/`

## GPU
- GPU: NVIDIA A10 (24GB)
- Driver: 580.95.05
- NVIDIA-SMI reports CUDA version: 13.0
- PyTorch: 2.7.1+cu126, CUDA available = True

## Installed ML stack
- transformers 4.57.6
- trl 0.24.0
- peft 0.18.0
- accelerate 1.11.0
- datasets 3.1.0

## Missing packages for this SOP
- bitsandbytes (required for QLoRA 4-bit)

## Operational blocker observed
- Docker container `next-edit` (image `vllm/vllm-openai:latest`) occupies ~22.9GB GPU memory.
- Training/data generation should run only after stopping that container during the training window.

## Local model assets
- `/root/Qwen3-8B` (~16GB)
- `/root/Qwen3-14B` (~28GB)

## Decision lock
- Primary training model: 8B
- 14B role: offline judge/evaluator only
- Data generation: pure self-play
- Delivery style: executable SOP
