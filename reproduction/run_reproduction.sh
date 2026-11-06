#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Run after `conda activate LLM-hw1`; this deliberately uses that environment's
# global Python and uv instead of creating a project virtual environment.
PY="${PYTHON:-python}"
mkdir -p reproduction/logs reproduction/results reproduction/figures

# 1. Data and deterministic baseline.
$PY reproduction/scripts/prepare_data.py
$PY reproduction/scripts/prepare_c4.py
$PY reproduction/scripts/evaluate.py --output reproduction/results/baseline_2048.json --sample
$PY reproduction/scripts/evaluate.py --seqlen 4096 --output reproduction/results/baseline_4096.json
$PY reproduction/scripts/evaluate.py --dataset c4 --output reproduction/results/baseline_c4.json

# 2. Paper-scale calibration. For a faster smoke test lower --samples.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY reproduction/scripts/calibrate.py --kind whiten --samples 256 --seqlen 2048
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY reproduction/scripts/calibrate.py --kind fisher --samples 2048 --seqlen 1024
$PY reproduction/scripts/merge_fisher.py

# 3. Core 50% G-LRD; use --uniform for the rank-allocation ablation.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY reproduction/scripts/compress_model.py --group-size 4 --ratio .5 --output reproduction/checkpoints/g4_r50
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY reproduction/scripts/evaluate_suite.py --model reproduction/checkpoints/g4_r50 --name g4_r50 --quantization

# 4. Core checks and released kernel.
$PY reproduction/scripts/check_core.py
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY reproduction/scripts/benchmark_kernel.py
$PY reproduction/scripts/make_report_assets.py
