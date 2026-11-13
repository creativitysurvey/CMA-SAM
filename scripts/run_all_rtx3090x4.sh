#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# CMA-SAM Full Paper Experimental Suite Launcher (Option A + SIGHUP/OOM Hardened)
# - Uses scripts/safe_torchrun.py so SSH idle disconnects (SIGHUP / signal 1)
#   never terminate distributed training.
# - Tuned for shared 2x NVIDIA A800 (~8 GiB peak VRAM per GPU, global batch = 32).
# - Automatically resumes both completed and mid-training checkpoints.
# ==============================================================================

trap '' HUP

NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29507}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
KVASIR_DIR="${KVASIR_DIR:-./data/Kvasir-SEG}"
ISIC_DIR="${ISIC_DIR:-./data/ISIC2018}"
HAM_DIR="${HAM_DIR:-./data/HAM10000}"
OUT_DIR="${OUT_DIR:-./results}"

export PYTHONNOUSERSITE=1
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "[0/9] Pre-caching 1024x1024 resized images across 16 CPU cores..."
python3 run_experiments.py --mode cache_resized \
  --kvasir_dir "${KVASIR_DIR}" --isic_dir "${ISIC_DIR}" --ham10000_dir "${HAM_DIR}"

echo "[1/9] Verifying deterministic dataset splits (Table II)..."
python3 run_experiments.py --mode splits \
  --kvasir_dir "${KVASIR_DIR}" --isic_dir "${ISIC_DIR}" --output_dir "${OUT_DIR}"

echo "[2/9] Running cross-dataset Zero-Shot experiments (Table III)..."
python3 scripts/safe_torchrun.py --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" run_experiments.py --mode zeroshot \
  --kvasir_dir "${KVASIR_DIR}" --isic_dir "${ISIC_DIR}" --output_dir "${OUT_DIR}" \
  --epochs 50 --warmup_epochs 5 --batch_size "${BATCH_SIZE}" --grad_accum_steps "${GRAD_ACCUM}" --lr 1e-4

echo "[3/9] Running 3-seed cross-domain Few-Shot N={1,5,10} adaptation (Figure 8)..."
python3 run_experiments.py --mode fewshot \
  --kvasir_dir "${KVASIR_DIR}" --output_dir "${OUT_DIR}" \
  --fewshot_epochs 120 --batch_size "${BATCH_SIZE}" --lr 1e-4

echo "[4/9] Running AAG Point K & Bounding Box B ablation (Table IV & Figure 10)..."
python3 scripts/safe_torchrun.py --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" run_experiments.py --mode ablation_aag \
  --kvasir_dir "${KVASIR_DIR}" --output_dir "${OUT_DIR}" \
  --epochs 50 --warmup_epochs 5 --batch_size "${BATCH_SIZE}" --grad_accum_steps "${GRAD_ACCUM}" --lr 1e-4

echo "[5/9] Running Spatial Softmax Temperature tau ablation (Table V)..."
python3 scripts/safe_torchrun.py --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" run_experiments.py --mode ablation_tau \
  --kvasir_dir "${KVASIR_DIR}" --output_dir "${OUT_DIR}" \
  --epochs 50 --warmup_epochs 5 --batch_size "${BATCH_SIZE}" --grad_accum_steps "${GRAD_ACCUM}" --lr 1e-4

echo "[6/9] Running Cross-Modal Fusion strategy ablation & 1D profiles (Table VII & Figure 11)..."
python3 scripts/safe_torchrun.py --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" run_experiments.py --mode ablation_fusion \
  --kvasir_dir "${KVASIR_DIR}" --isic_dir "${ISIC_DIR}" --output_dir "${OUT_DIR}" \
  --epochs 50 --warmup_epochs 5 --batch_size "${BATCH_SIZE}" --grad_accum_steps "${GRAD_ACCUM}" --lr 1e-4

echo "[7/9] Evaluating Simple vs. Complex Clinical Text stability (Table VI)..."
python3 run_experiments.py --mode prompt_complexity \
  --kvasir_dir "${KVASIR_DIR}" --output_dir "${OUT_DIR}" --batch_size "${BATCH_SIZE}"

echo "[8/9] Measuring Computational Efficiency & Multi-Dataset ALE (Table VIII, Fig 7, Fig 12)..."
python3 run_experiments.py --mode efficiency_and_ale \
  --kvasir_dir "${KVASIR_DIR}" --isic_dir "${ISIC_DIR}" --ham10000_dir "${HAM_DIR}" \
  --output_dir "${OUT_DIR}" --batch_size "${BATCH_SIZE}"

echo "[9/9] Exporting Qualitative Figures (Fig 9, Fig 13) & LaTeX Tables..."
python3 run_experiments.py --mode qualitative \
  --kvasir_dir "${KVASIR_DIR}" --isic_dir "${ISIC_DIR}" --output_dir "${OUT_DIR}"
python3 run_experiments.py --mode export_latex --output_dir "${OUT_DIR}"

echo "=== ALL CMA-SAM EXPERIMENTS COMPLETED! ==="
