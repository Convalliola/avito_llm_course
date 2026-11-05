#!/bin/bash
# ./run.sh GPU NAME [NAME ...], base это оценка исходной модели
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=$1 HF_HOME=${HF_HOME:-$PWD/hf} TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
shift
mkdir -p logs
for name in "$@"; do
    if [ "$name" = base ]; then venv/bin/python eval_model.py > logs/base.log 2>&1
    else venv/bin/python train.py "$name" > "logs/$name.log" 2>&1; fi
    status=$?
    echo "$(date +%H:%M) $name exit $status"
done
