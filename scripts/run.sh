#!/bin/bash
conda init
conda activate /hdd/yuke/fanjiang/conda_env/lumina-dimoo
cd /scr/dataset/yuke/fanjiang/repo/unified-model/Lumina-DiMOO
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/scr/dataset/yuke/fanjiang/repo/unified-model/Lumina-DiMOO:$PYTHONPATH

LOG_DIR="/scr/dataset/yuke/fanjiang/repo/unified-model/Lumina-DiMOO/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/${TIMESTAMP}_run.log"
exec >> "$LOG_FILE" 2>&1

set -x

python scripts/inference_t2i.py\
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --prompt "A striking photograph of a glass of orange juice on a wooden kitchen table, capturing a playful moment. The orange juice splashes out of the glass and forms the word \"Smile\" in a whimsical, swirling script just above the glass. The background is softly blurred, revealing a cozy, homely kitchen with warm lighting and a sense of comfort." \
    --height 768 \
    --width 1536 \
    --timesteps 64 \
    --cfg_scale 4.0 \
    --seed 65513 \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir output/results_text_to_image
