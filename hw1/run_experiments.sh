#!/usr/bin/env bash
# Experiment plan (see RUNBOOK.md). The final results in README come from:
#   ./run_experiments.sh prepare   # tokenize ruwiki into output_dir/*.parquet (once)
#   ./run_experiments.sh quick     # speed check, template baseline, LR 6e-4 / 1e-3 / 2e-3, plots (~80 min)
#   ./run_experiments.sh extra     # LR 3e-4, schedulers, grad accumulation x2/x4, fp32 master, Adafactor (~2 h)
# Stages of the first A100 series (padding-free era; padded batches are the default now):
#   ./run_experiments.sh check     # padding-free vs padded forward-pass check
#   ./run_experiments.sh speed     # ~2-4 min per config: throughput of system-level options
#   ./run_experiments.sh base      # template hyperparameters (baseline), 15 min
#   ./run_experiments.sh lr        # LR sweep, 15 min each
#   BEST_LR=1e-3 ./run_experiments.sh sched | bs | system
#   BEST_LR=1e-3 ./run_experiments.sh plots
set -uo pipefail
cd "$(dirname "$0")"

BEST_LR=${BEST_LR:-1e-3}
BEST_BS=${BEST_BS:-16}
LOG_DIR=output_dir/logs
mkdir -p "$LOG_DIR"
# All stage runs were made with gradient_accumulation_steps=1 (the code default before the best configuration was
# found), so it is pinned here; a later --set in a stage (e.g. bs_16x2, bs_16x4) overrides it.
PIN="--set gradient_accumulation_steps=1"

train() {  # train <run name> [--set key=value ...]
  local name=$1; shift
  if [ -f "output_dir/runs/$name/run_summary.json" ]; then
    echo "== $name already done, skipping"; return
  fi
  echo "== $name  $*"
  python your_solution.py train --name "$name" $PIN "$@" 2>&1 | tee "$LOG_DIR/$name.log"
}

speed() {  # speed <name> [--set key=value ...]
  local name=$1; shift
  echo "== speed $name  $*"
  python benchmark.py speed --name "$name" $PIN "$@" 2>&1 | tee "$LOG_DIR/speed_$name.log" | tail -15
}

case "${1:-}" in
  prepare)
    python your_solution.py prepare 2>&1 | tee "$LOG_DIR/prepare.log"
    ;;
  check)
    python benchmark.py check 2>&1 | tee "$LOG_DIR/check.log"
    ;;
  speed)
    speed default                                       # padding-free, bs16, compile, fused AdamW, bf16 weights
    speed padded           --set padding_free=false
    speed nocompile        --set torch_compile=false
    speed padded_nocompile --set padding_free=false --set torch_compile=false
    speed bs8              --set per_device_train_batch_size=8
    speed bs32             --set per_device_train_batch_size=32
    speed bs64             --set per_device_train_batch_size=64
    speed adamw_foreach    --set optim=adamw_torch
    speed adafactor        --set optim=adafactor
    speed fp32master       --set fp32_master_weights=true
    speed fp32master_notf32 --set fp32_master_weights=true --set tf32=false
    ;;
  base)
    # Hyperparameters of the template: bs 4, lr 5e-5, linear decay over the full epoch (~constant in 15 min),
    # padded batches, eval/save every 100 steps with load_best_model_at_end.
    train base_template \
      --set per_device_train_batch_size=4 --set learning_rate=5e-5 --set warmup_steps=200 \
      --set weight_decay=0.01 --set adam_beta2=0.999 --set lr_scheduler_type=linear --set time_lr_schedule=null \
      --set padding_free=false \
      --set eval_strategy=steps --set eval_steps=100 --set save_strategy=steps --set save_steps=100 \
      --set save_total_limit=2 --set load_best_model_at_end=true --set metric_for_best_model=eval_subset_loss \
      --set save_only_model=false --set logging_steps=1
    ;;
  lr)
    for lr in 3e-4 6e-4 1e-3 2e-3 4e-3; do
      train "lr_${lr}" --set learning_rate=$lr --set per_device_train_batch_size=$BEST_BS
    done
    ;;
  sched)
    for s in cosine linear constant; do
      train "sched_${s}" --set time_lr_schedule=$s --set learning_rate=$BEST_LR --set per_device_train_batch_size=$BEST_BS
    done
    train sched_wsd10 --set decay_fraction=0.1 --set learning_rate=$BEST_LR --set per_device_train_batch_size=$BEST_BS
    train sched_wsd50 --set decay_fraction=0.5 --set learning_rate=$BEST_LR --set per_device_train_batch_size=$BEST_BS
    ;;
  bs)
    train bs_8  --set per_device_train_batch_size=8  --set learning_rate=$BEST_LR
    train bs_32 --set per_device_train_batch_size=32 --set learning_rate=$BEST_LR
    train bs_64 --set per_device_train_batch_size=64 --set learning_rate=$BEST_LR
    ;;
  system)
    common="--set learning_rate=$BEST_LR --set per_device_train_batch_size=$BEST_BS"
    train sys_fp32master --set fp32_master_weights=true $common
    train sys_padded     --set padding_free=false $common
    train sys_nocompile  --set torch_compile=false $common
    ;;
  quick)
    speed padded_compile                                 # current default: padded, bs16, compile
    speed padded_compile_bs32 --set per_device_train_batch_size=32
    train base_template \
      --set per_device_train_batch_size=4 --set learning_rate=5e-5 --set warmup_steps=200 \
      --set weight_decay=0.01 --set adam_beta2=0.999 --set lr_scheduler_type=linear --set time_lr_schedule=null \
      --set padding_free=false \
      --set eval_strategy=steps --set eval_steps=100 --set save_strategy=steps --set save_steps=100 \
      --set save_total_limit=2 --set load_best_model_at_end=true --set metric_for_best_model=eval_subset_loss \
      --set save_only_model=false --set logging_steps=1
    for lr in 6e-4 1e-3 2e-3; do
      train "lr_${lr}" --set learning_rate=$lr --set per_device_train_batch_size=$BEST_BS
    done
    python plot_losses.py --group "overview=base_template,lr_6e-4,lr_1e-3,lr_2e-3"
    ;;
  extra)
    # One change at a time relative to the best quick run (lr 6e-4, bs 16, WSD by time, AdamW, bf16 weights).
    base="--set learning_rate=6e-4 --set per_device_train_batch_size=16"
    train lr_3e-4          --set learning_rate=3e-4 --set per_device_train_batch_size=16
    train sched_cosine     $base --set time_lr_schedule=cosine
    # Standard HF cosine over the whole epoch (~121k steps): within 15 minutes the LR barely moves.
    train sched_hf_cosine  $base --set time_lr_schedule=null --set lr_scheduler_type=cosine
    train bs_16x2          $base --set gradient_accumulation_steps=2
    train bs_16x4          $base --set gradient_accumulation_steps=4
    # fp32 weights + grads + Adam states take 16 GB: a smaller eval batch keeps the eval peak within 32 GB.
    train sys_fp32master   $base --set fp32_master_weights=true --set per_device_eval_batch_size=25
    train optim_adafactor  $base --set optim=adafactor
    python plot_losses.py \
      --group "overview=base_template,lr_6e-4,lr_1e-3,lr_2e-3" \
      --group "sched=lr_6e-4,sched_cosine,sched_hf_cosine" \
      --group "batch=lr_6e-4,bs_16x2,bs_16x4" \
      --group "optim_dtype=lr_6e-4,sys_fp32master,optim_adafactor"
    ;;
  plots)
    best="lr_${BEST_LR}"
    python plot_losses.py \
      --group "sched=${best},sched_cosine,sched_linear,sched_constant,sched_wsd10,sched_wsd50" \
      --group "bs=bs_8,${best},bs_32,bs_64" \
      --group "sys=${best},sys_fp32master,sys_padded,sys_nocompile" \
      --group "overview=base_template,${best}"
    ;;
  *)
    sed -n '2,12p' "$0"; exit 1
    ;;
esac
