#!/bin/bash

set -e

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_PATH="Leslie04/parathinker-math-6K/train" # Set your own training dataset path here
EVAL_DATASET_PATH="Leslie04/parathinker-math-6K/test"
OUTPUT_DIR="train/ParaThinker-1.5B"
DEEPSPEED_CONFIG="train/config/ds_z3_config.json"
WANDB_DISABLED=true

TEMPLATE="qwen"
BATCH_SIZE=1
GRAD_ACCUM=16
LR="1e-5"
EPOCHS=3
CUTOFF_LEN=28672
GPUS="0,1,2,3"

LR_SCHEDULER="constant"
WARMUP_RATIO=0.1
LOGGING_STEPS=20
SAVE_STEPS=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --dataset)
      DATASET_PATH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --deepspeed)
      DEEPSPEED_CONFIG="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --grad_accum)
      GRAD_ACCUM="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --cutoff_len)
      CUTOFF_LEN="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

# Print training configuration
echo "========================================="
echo "  TRAINING CONFIGURATION"
echo "========================================="
echo "Model:                  $MODEL"
echo "Dataset path:           $DATASET_PATH"
echo "Output directory:       $OUTPUT_DIR"
echo "DeepSpeed config:       $DEEPSPEED_CONFIG"
echo "Batch size:             $BATCH_SIZE"
echo "Gradient accumulation:  $GRAD_ACCUM"
echo "Learning rate:          $LR"
echo "Training epochs:        $EPOCHS"
echo "Context length:         $CUTOFF_LEN"
echo "GPUs:                   $GPUS"
echo "========================================="
echo "Starting training in 3 seconds..."
sleep 3

# Set environment variables
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FORCE_TORCHRUN=1
export TOKENIZERS_PARALLELISM=false

# Get number of visible GPUs for torchrun
NUM_GPUS=$(echo "$GPUS" | tr -cd ',' | wc -c)
NUM_GPUS=$((NUM_GPUS + 1))

echo "Using $NUM_GPUS GPUs for training"

# Run the training script
if [ "$NUM_GPUS" -gt 1 ]; then
  # Multi-GPU training with torchrun
  echo "Starting distributed training with torchrun..."
  python -m torch.distributed.run --nproc_per_node="$NUM_GPUS" train/train_script.py \
    --model_name_or_path "$MODEL" \
    --finetuning_type full \
    --dataset "$DATASET_PATH" \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --template "$TEMPLATE" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --weight_decay 0.05 \
    --lr_scheduler_type "$LR_SCHEDULER" \
    --warmup_ratio "$WARMUP_RATIO" \
    --do_train \
    --bf16 \
    --gradient_checkpointing \
    --overwrite_cache \
    --max_grad_norm 0.5 \
    --cutoff_len "$CUTOFF_LEN" \
    --num_train_epochs "$EPOCHS" \
    --report_to none \
    --log_level info \
    --logging_steps "$LOGGING_STEPS" \
    --resize_vocab true

else
  # Single GPU training
  echo "Please use distributed training ..."
fi
