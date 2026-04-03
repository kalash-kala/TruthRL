#!/bin/bash
# ============================================================
# SAM-Guided Targeted Perturbation Pipeline Runner
# ============================================================
# Infrastructure: 2×A100 (160 GB VRAM)
# Input: 700 images with visual cues (JSONL)
# Output: Paired answerable/unanswerable JSONL + images
# ============================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="/home/sriramg/kalashabhayk"

INPUT_JSONL="${BASE_DIR}/visual-question-answering/clean_vqa_with_visual_cues_tagged.jsonl"
OUTPUT_JSONL="${BASE_DIR}/visual-question-answering/unanswerable_sam_targeted.jsonl"
IMAGE_DIR="${BASE_DIR}/visual-question-answering/sam_perturbed_images"
IMAGE_ROOT="${BASE_DIR}/visual-question-answering/processed_for_verl/images"
ENV_PATH="${BASE_DIR}/Perception-R1/.env"

# ── Config ───────────────────────────────────────────────────
SAM_TYPE="sam2"           # Options: sam2, vit_h, vit_l, vit_b
DEVICE="cuda:0"           # Use first A100
SLEEP_INTERVAL=2.0        # Seconds between Gemini API calls
FEATHER_SIGMA=5.0         # Mask edge feathering
SEED=42

# ── Optional: row range for partial processing ───────────────
# START_ROW=0
# END_ROW=10
# TARGET_COUNT=0           # 0 = process all

echo "============================================================"
echo "SAM-Guided Targeted Perturbation Pipeline"
echo "============================================================"
echo "Input:       ${INPUT_JSONL}"
echo "Output:      ${OUTPUT_JSONL}"
echo "Image Dir:   ${IMAGE_DIR}"
echo "Image Root:  ${IMAGE_ROOT}"
echo "SAM Type:    ${SAM_TYPE}"
echo "Device:      ${DEVICE}"
echo "============================================================"

python "${SCRIPT_DIR}/sam_targeted_perturbation.py" \
    --input_jsonl   "${INPUT_JSONL}" \
    --output_jsonl  "${OUTPUT_JSONL}" \
    --image_dir     "${IMAGE_DIR}" \
    --image_root    "${IMAGE_ROOT}" \
    --env_path      "${ENV_PATH}" \
    --sam_type      "${SAM_TYPE}" \
    --device        "${DEVICE}" \
    --sleep_interval "${SLEEP_INTERVAL}" \
    --feather_sigma "${FEATHER_SIGMA}" \
    --seed          "${SEED}" \
    ${START_ROW:+--start_row "${START_ROW}"} \
    ${END_ROW:+--end_row "${END_ROW}"} \
    ${TARGET_COUNT:+--target_count "${TARGET_COUNT}"}

echo ""
echo "Done! Output: ${OUTPUT_JSONL}"
echo "Images: ${IMAGE_DIR}"
