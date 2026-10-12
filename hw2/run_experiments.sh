#!/usr/bin/env bash
# Запуск экспериментов ДЗ 2 по стадиям (подробности в RUN_ON_VM.md)
#   ./run_experiments.sh prepare    подготовка данных (один раз)
#   ./run_experiments.sh smoke      быстрая проверка двух конфигураций по 30 с
#   ./run_experiments.sh short      все конфигурации по 120 с
#   ./run_experiments.sh only <имя>...   выбранные конфигурации по 120 с
#   ./run_experiments.sh report     таблица и графики в report/
#   ./run_experiments.sh final      бейзлайн и train.py с настройками по умолчанию (итоговая конфигурация)
#                                   по 300 с, BEST=<имя> и BEST2=<имя> добавляют конфигурации из EXPERIMENTS
#   ./run_experiments.sh notebook   выполнить communications.ipynb
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p results/logs

SHORT_SECONDS=${SHORT_SECONDS:-120}

# Порядок по важности, чтобы при нехватке времени главное было сделано первым
SHORT_EXPERIMENTS=(
  baseline
  fa2
  compile
  liger
  fa2_padding_free
  fa2_padding_free_bs64
  fa2_packing
  fa2_padding_free_bs64_liger
  fa2_padding_free_bs64_liger_compile
  fa2_packing_liger_compile
  bs64_ga1
  bs16_ga4
  checkpointing
  offloading
  fa2_padding_free_bs128_liger
  fa2_padding_free_bs256_liger
  compile_dynamic
  fa2_padding_free_bs64_liger_compile_dynamic
  fa2_offloading
  fa2_padding_free_bs128_liger_compile_dynamic
  fa2_padding_free_bs256_liger_compile_dynamic
  fa2_packing_liger_compile_dynamic
  fa2_padding_free_bs64_compile_dynamic
)

run() {  # run <конфигурация> <секунды>, конфигурация final означает train.py без EXPERIMENT
  local name=$1 seconds=$2 experiment=$1
  if [ "$name" = final ]; then experiment=""; fi
  local out="results/${name}_${seconds}s"
  if [ -f "$out/summary.json" ]; then
    echo "== $out уже готов, пропускаю"; return
  fi
  if [ -d "$out" ]; then  # прошлый запуск упал, train.py не перезаписывает папку
    mv "$out" "${out}_failed_$(date +%Y%m%d_%H%M%S)"
  fi
  echo "== $name, $seconds с"
  EXPERIMENT=$experiment TRAIN_SECONDS=$seconds python train.py 2>&1 | tee "results/logs/${name}_${seconds}s.log"
  if [ -f "$out/summary.json" ]; then
    python -c "import json; s = json.load(open('$out/summary.json')); print(f\"== $name: {s['tokens_per_second'] or 0:.0f} ток/с, пик {s['peak_allocated_gib']:.1f} GiB, val loss {s['validation_loss']:.3f}\")"
  else
    echo "== $name: УПАЛ, см. results/logs/${name}_${seconds}s.log"
  fi
}

case "${1:-}" in
  prepare)
    python -c "import train; train.prepare_data()" 2>&1 | tee results/logs/prepare.log
    ;;
  smoke)
    run baseline 30
    run fa2_padding_free_bs64_liger_compile 30
    ;;
  short)
    for name in "${SHORT_EXPERIMENTS[@]}"; do run "$name" "$SHORT_SECONDS"; done
    ;;
  only)
    shift
    for name in "$@"; do run "$name" "$SHORT_SECONDS"; done
    ;;
  final)
    run baseline 300
    run final 300
    if [ -n "${BEST:-}" ]; then run "$BEST" 300; fi
    if [ -n "${BEST2:-}" ]; then run "$BEST2" 300; fi
    ;;
  notebook)
    jupyter nbconvert --to notebook --execute --inplace communications.ipynb
    ;;
  report)
    python report.py
    ;;
  *)
    sed -n '2,10p' "$0"; exit 1
    ;;
esac
