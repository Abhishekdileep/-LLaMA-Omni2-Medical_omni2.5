#!/usr/bin/env bash
# MedQA fine-tune across all three stages, in dependency order.
#
# Each stage's output is the next one's frozen backbone, so the order is forced:
#
#   1a  <question_wav, gold answer>  -> adapter + last LLM layer   (moves accuracy)
#   1b  <text, units>                -> MTTS on medical vocabulary
#   re-cache                         -> Stage I(a) changed the LLM, so the Stage II
#                                       hidden-state cache is stale and MUST be rebuilt
#   2   <question_wav, text, units>  -> gate fusion + MTTS, continuing from 1b
#
# The re-cache is the step that is easy to skip by hand and silently wrong if you do:
# Stage II would train the fusion against hidden states from the pre-fine-tune LLM.
#
#   bash scripts/run_medqa_finetune.sh
#   SKIP_CORPUS=1 bash scripts/run_medqa_finetune.sh      # reuse an existing corpus
#   TRAIN_N=800 TEST_N=200 bash scripts/run_medqa_finetune.sh
set -euo pipefail

cd "$(dirname "$0")/.."                       # LLaMA-Omni2-main
REPO="$(pwd)"
PROJECT="$(dirname "$REPO")"

PY_OMNI="${PY_OMNI:-/DATA/hnc/conda_envs/llama-omni2/bin/python}"
PY_COSY="${PY_COSY:-/DATA/hnc/conda_envs/cosyvoice/bin/python}"

BASE_MODEL="${BASE_MODEL:-models/LLaMA-Omni2-7B-Bilingual}"
VOCODER_DIR="${VOCODER_DIR:-models/cosy2_decoder}"
RUN_DIR="${RUN_DIR:-$PROJECT/data/medqa_runs/$(date +%Y%m%d)}"
CORPUS_DIR="${CORPUS_DIR:-$PROJECT/data/medqa_corpus}"

TRAIN_N="${TRAIN_N:-400}"
TEST_N="${TEST_N:-150}"
SEED="${SEED:-0}"
EPOCHS_1A="${EPOCHS_1A:-3}"
EPOCHS_1B="${EPOCHS_1B:-3}"
EPOCHS_2="${EPOCHS_2:-3}"

mkdir -p "$RUN_DIR"
echo "run dir: $RUN_DIR"

# --------------------------------------------------------------- 0. corpus
# Question audio is chunked at 28 s: MedQA's median question synthesises to ~57 s,
# and only 3.5% of the dataset fits in whisper's 30 s window in one piece.
if [ -z "${SKIP_CORPUS:-}" ]; then
  echo "=== 0. building MedQA corpus (CosyVoice env) ==="
  "$PY_COSY" scripts/build_medqa_corpus.py --split train --num "$TRAIN_N" --seed "$SEED" \
      --out-dir "$CORPUS_DIR/train"
  "$PY_COSY" scripts/build_medqa_corpus.py --split test  --num "$TEST_N"  --seed "$SEED" \
      --out-dir "$CORPUS_DIR/test"
fi
"$PY_OMNI" scripts/project_manifest.py --manifest "$CORPUS_DIR/train/medqa_corpus.json"
"$PY_OMNI" scripts/project_manifest.py --manifest "$CORPUS_DIR/test/medqa_corpus.json"

# --------------------------------------------------------------- baseline
echo "=== baseline: $BASE_MODEL on the held-out test split ==="
"$PY_OMNI" scripts/eval_medqa.py --model-path "$BASE_MODEL" \
    --manifest "$CORPUS_DIR/test/medqa_corpus.json" \
    --out "$RUN_DIR/predictions_baseline.json"
"$PY_OMNI" bleuscore.py --predictions "$RUN_DIR/predictions_baseline.json" \
    --manifest "$CORPUS_DIR/test/medqa_corpus.json" --quiet \
    | tee "$RUN_DIR/accuracy_baseline.txt"

# --------------------------------------------------------------- 1a
# --gradient_checkpointing is not optional here: the speech adaptor sits below the
# LLM, so its gradient runs through all 28 layers, and a chunked MedQA question is
# ~900 speech frames. Without it this OOMs on a 24 GB card.
echo "=== 1a. speech -> text ==="
"$PY_OMNI" llama_omni2/train_stage1a.py \
    --model_name_or_path "$BASE_MODEL" \
    --data_path "$CORPUS_DIR/train/stage1a_data.json" \
    --instruction mcq \
    --output_dir "$RUN_DIR/stage1a" \
    --bf16 True --gradient_checkpointing True \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
    --num_train_epochs "$EPOCHS_1A" --learning_rate 5e-5 \
    --save_strategy no --logging_steps 10 --report_to none

# --------------------------------------------------------------- 1b
echo "=== 1b. text -> units ==="
"$PY_OMNI" llama_omni2/train_stage1b.py \
    --omni2_path "$BASE_MODEL" \
    --data_path "$CORPUS_DIR/train/stage1b_data.json" \
    --output_dir "$RUN_DIR/stage1b" \
    --bf16 True --per_device_train_batch_size 4 \
    --num_train_epochs "$EPOCHS_1B" --learning_rate 2e-5 \
    --save_strategy no --logging_steps 10 --report_to none

# --------------------------------------------------------------- re-cache
# Against the Stage I(a) output, NOT the base model.
echo "=== re-caching LLM hidden states against stage1a ==="
rm -rf "$RUN_DIR/cache"
"$PY_OMNI" llama_omni2/precompute_stage2_hidden.py \
    --model_name_or_path "$RUN_DIR/stage1a" \
    --data_path "$CORPUS_DIR/train/stage2_data.json" \
    --cache_dir "$RUN_DIR/cache" --instruction mcq

# --------------------------------------------------------------- 2
echo "=== 2. gate fusion + MTTS ==="
"$PY_OMNI" llama_omni2/train_stage2.py \
    --model_name_or_path "$RUN_DIR/stage1a" \
    --mtts_init "$RUN_DIR/stage1b" \
    --data_path "$CORPUS_DIR/train/stage2_data.json" \
    --hidden_cache_dir "$RUN_DIR/cache" \
    --output_dir "$RUN_DIR/stage2" \
    --bf16 True --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
    --num_train_epochs "$EPOCHS_2" --learning_rate 2e-5 --fusion_lr 1e-4 \
    --save_strategy no --logging_steps 10 --report_to none

# --------------------------------------------------------------- eval
echo "=== fine-tuned: $RUN_DIR/stage2 on the held-out test split ==="
"$PY_OMNI" scripts/eval_medqa.py --model-path "$RUN_DIR/stage2" \
    --manifest "$CORPUS_DIR/test/medqa_corpus.json" \
    --out "$RUN_DIR/predictions_finetuned.json" \
    --vocoder-dir "$VOCODER_DIR"
"$PY_OMNI" bleuscore.py --predictions "$RUN_DIR/predictions_finetuned.json" \
    --manifest "$CORPUS_DIR/test/medqa_corpus.json" --quiet \
    | tee "$RUN_DIR/accuracy_finetuned.txt"

echo
echo "=============================================="
echo "baseline:"   ; grep "Accuracy %" "$RUN_DIR/accuracy_baseline.txt"
echo "fine-tuned:" ; grep "Accuracy %" "$RUN_DIR/accuracy_finetuned.txt"
echo "=============================================="
