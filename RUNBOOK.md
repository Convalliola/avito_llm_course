# Как запустить на VM

## 1. Подготовка

```bash
nvidia-smi                        # A100 80GB, драйвер с поддержкой CUDA 11.8+
git clone <repo> && cd <repo>/hw1 # или scp -r hw1 vm:~/
df -h .                           # нужно ~60 GB: parquet + arrow-кэш + чекпоинты (2 модели по ~2 GB на ран)

docker build -t llm-hw1 .
# flash-attn 2.7.3 может собираться из исходников (долго). Если сборка падает по памяти —
# в Dockerfile перед pip install flash-attn добавить ENV MAX_JOBS=8.

tmux new -s hw1                   # чтобы обучение пережило разрыв ssh
docker run --gpus all --ipc=host --rm -it \
  -v "$PWD":/app \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  llm-hw1 bash
```

`-v "$PWD":/app` монтирует код и `output_dir/` с хоста: результаты сохранятся после выхода из контейнера, а правки
кода не требуют пересборки образа. `--ipc=host` нужен воркерам DataLoader (shared memory).

## 2. Стадии (внутри контейнера)

| # | команда | время | что делает |
|---|---|---|---|
| 0 | `./run_experiments.sh prepare` | 20–40 мин | токенизация ruwiki → `output_dir/000NN.parquet` (один раз) |
| 1 | `./run_experiments.sh check` | 2 мин | padded vs padding-free: логиты должны совпасть (до шума bf16), «без границ» — нет |
| 2 | `./run_experiments.sh speed` | ~35 мин | пропускная способность: padding-free, compile, batch size, оптимизатор, dtype |
| 3 | `./run_experiments.sh base` | 18 мин | бейзлайн с гиперпараметрами шаблона |
| 4 | `./run_experiments.sh lr` | 5 × 18 мин | LR sweep (bs из `BEST_BS`, по умолчанию 16) |
| 5 | `BEST_LR=... ./run_experiments.sh sched` | 5 × 18 мин | формы расписания LR |
| 6 | `BEST_LR=... ./run_experiments.sh bs` | 3 × 18 мин | batch size |
| 7 | `BEST_LR=... ./run_experiments.sh system` | 3 × 18 мин | fp32 master weights, padded, без compile |
| 8 | `BEST_LR=... ./run_experiments.sh plots` | 1 мин | графики и таблица в `report/` |

После стадии 2 посмотреть `output_dir/benchmarks.jsonl`: если bs32 заметно быстрее по токенам/с, а память позволяет —
запускать стадии 4+ с `BEST_BS=32`. Если `check` или `speed default` падают на padding-free/compile —
отключить через `--set padding_free=false` / `--set torch_compile=false` (оба режима поддержаны).

Если времени на VM мало, минимальный набор: 0 → 1 → 2 → 3 → `lr` (3e-4, 1e-3, 2e-3) → `sched_cosine`, `sched_constant` → `sys_fp32master` → 8.

Ран с уже существующим `run_summary.json` пропускается, поэтому стадию можно безопасно перезапустить после обрыва.

## 3. Результаты

- `output_dir/runs/<run>/run_summary.json` — конфиг, финальный `eval_loss` (полные 5000 сэмплов), perplexity,
  число шагов и токенов, пропускная способность, генерации;
- `output_dir/runs/<run>/trainer_state.json` — история лоссов для графиков;
- `output_dir/runs/<run>/final/` — веса + токенизатор: `python generate.py output_dir/runs/<run>/final "Свой промпт"`;
- `output_dir/logs/*.log` — полные логи.

Перед коммитом в GitHub скопировать `report/` (графики + `results.md`) — `output_dir/` в `.gitignore`.
Чекпоинты (`checkpoint-*`) можно удалять: `final/` содержит те же веса.
