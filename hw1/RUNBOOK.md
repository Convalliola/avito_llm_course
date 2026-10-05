# Как воспроизвести результаты

Все команды выполняются из папки с `your_solution.py`. Результаты, описанные в [README.md](README.md), получены
командами `prepare` → `quick` → `extra` из раздела 3.

## 1. Окружение

Нужен один GPU с поддержкой bf16 и FlashAttention-2 (Ampere и новее) и не меньше ~24 GB памяти: опорная
конфигурация (bs 16) занимает ~17.5 GB, вариант с fp32 master weights — ~24 GB. Версии библиотек — как в
[Dockerfile](Dockerfile): `transformers==4.52.0`, `datasets==3.6.0`, `accelerate==1.6.0`.

**Вариант А — Docker** (образ курса: torch 2.6 + CUDA 11.8, flash-attn 2.7.3):

```bash
docker build -t llm-hw1 .
docker run --gpus all --ipc=host --rm -it -v "$PWD":/app \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface llm-hw1 bash
```

**Вариант Б — без Docker.** Сборку torch и flash-attn выбираем под драйвер (`nvidia-smi` → «Driver Version»).
Для flash-attn берём готовую сборку: компиляция из исходников идёт часами и требует `nvcc`.

Старый драйвер (CUDA 11.x, например 470) — так запускалось на A100, Python 3.10:

```bash
python3 -m venv ~/venv-hw1 && source ~/venv-hw1/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install 'transformers==4.52.0' 'datasets==3.6.0' 'accelerate==1.6.0' matplotlib
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# torch.compile: Triton собирает ядра ptxas из CUDA 12, которые драйвер 470 не загружает
# («device kernel image is invalid»). Подменяем ptxas на версию из CUDA 11.8:
pip install "nvidia-cuda-nvcc-cu11==11.8.89"
echo "export TRITON_PTXAS_PATH=$(python -c 'import nvidia.cuda_nvcc as m; print(m.__path__[0])')/bin/ptxas" \
  >> ~/venv-hw1/bin/activate
```

Новый драйвер (CUDA 12.8+) — так запускалось на RTX 5090, Python 3.12:

```bash
conda create -y -n hw1 python=3.12 && conda activate hw1
export PYTHONNOUSERSITE=1   # см. примечание ниже; нужен и при установке, и при запуске
pip install torch==2.8.0 'transformers==4.52.0' 'datasets==3.6.0' 'accelerate==1.6.0' matplotlib \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

Примечание про `PYTHONNOUSERSITE=1`: если в `~/.local` уже стоят Python-пакеты, pip считает их зависимости
установленными и не кладёт их в окружение — потом `import datasets` падает. Переменная прячет `~/.local` от pip
и Python.

Проверка (должно напечатать `True`, имя GPU и форму тензора):

```bash
python -c "
import torch; from flash_attn import flash_attn_func
q = torch.randn(1, 128, 8, 64, device='cuda', dtype=torch.bfloat16)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0), flash_attn_func(q, q, q, causal=True).shape)"
python -c "import torch; f = torch.compile(lambda x: torch.sin(x) + 1); print(f(torch.randn(8, device='cuda')))"
```

Если вторая команда падает, `torch.compile` на этой машине не работает: выключите его через
`--set torch_compile=false` (обучение будет ~30% медленнее).

## 2. Данные (один раз, 20–40 минут, только CPU)

```bash
./run_experiments.sh prepare
ls output_dir/*.parquet | wc -l     # 32 шарда, 1 945 063 документа
```

Пиковое место на диске — ~30 GB (исходный датасет и кэши Hugging Face); готовые шарды занимают 2.5 GB.

## 3. Эксперименты

Обучение длится часы — запускайте в `tmux` или через `nohup`. Ран, у которого уже есть `run_summary.json`,
пропускается, так что прерванную стадию можно просто перезапустить.

```bash
./run_experiments.sh quick    # ~80 мин: замер скорости, бейзлайн шаблона, lr 6e-4 / 1e-3 / 2e-3, графики
./run_experiments.sh extra    # ~2 ч: lr 3e-4, cosine по времени, cosine HF, grad accumulation ×2 / ×4,
                              #       fp32 master weights, Adafactor, графики
```

Отдельный ран. Значения по умолчанию в `your_solution.py` совпадают с лучшей конфигурацией (lr 6e-4, bs 16 ×
gradient accumulation 4, WSD по времени), так что без параметров воспроизводится лучший ран. Любой ключ
`TrainingArguments` или `SOLUTION_CONFIG` переопределяется через `--set` (значение разбирается как JSON):

```bash
python your_solution.py train --name best                                   # лучшая конфигурация
python your_solution.py train --name my_run --set learning_rate=1e-3 --set gradient_accumulation_steps=1
```

Стадии `run_experiments.sh` явно ставят `gradient_accumulation_steps=1` (так шли все раны, кроме `bs_16x2` и
`bs_16x4`), поэтому они воспроизводят эксперименты из отчёта, а не новые значения по умолчанию.

Каждый ран длится ровно 15 минут обучения (`TimeoutCallback`), плюс ~2 минуты на загрузку, финальный eval,
сохранение и генерации.

## 4. Где результаты

| путь | что внутри |
|---|---|
| `output_dir/runs/<ран>/run_summary.json` | конфиг, `eval_loss`, perplexity, шаги, токены, скорость, генерации |
| `output_dir/runs/<ран>/trainer_state.json` | история train/eval loss и LR для графиков |
| `output_dir/runs/<ран>/final/` | веса модели и токенизатор (~2 GB) |
| `output_dir/logs/` | полные логи ранов |
| `report/` | графики и таблица результатов (`python plot_losses.py`, стадии строят их сами) |

`output_dir/` в `.gitignore`. Для репозитория `run_summary.json`, `trainer_state.json` и логи скопированы в
[results/](results/).

## 5. Генерации

```bash
python generate.py output_dir/runs/bs_16x4/final "Река Волга —" "Кукрыниксы —" --max-new-tokens 100 --out samples.json
```

На каждый промпт печатаются greedy и сэмплирование. Не оставляйте пробел в конце промпта: одиночный пробел в
конце модели почти не встречался, и генерация ломается (см. README, раздел «Генерации»).

## 6. Диагностика

```bash
python benchmark.py speed --name bs32 --set per_device_train_batch_size=32   # скорость → output_dir/benchmarks.jsonl
python benchmark.py check                                                    # padded vs padding-free: прямой проход
```

Разбор NaN в режиме padding-free (README, раздел 3): обучение останавливается на первом NaN и сохраняет батч и
веса, `replay` проигрывает батч в разных форматах под `torch.autograd.detect_anomaly`:

```bash
python your_solution.py train --name dbg_nan --set padding_free=true --set debug_nan=true \
  --set learning_rate=6e-4 --set max_steps=300 --set torch_compile=false --set generate_after_training=false
python benchmark.py replay output_dir/runs/dbg_nan/nan_dump
```
