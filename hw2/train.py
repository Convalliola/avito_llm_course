import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from transformers.utils import ModelOutput
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


DATA: Path = Path("data")
# Длительность и конфигурация задаются переменными окружения TRAIN_SECONDS и EXPERIMENT
TRAIN_SECONDS: float = float(os.environ.get("TRAIN_SECONDS", 300))
# по умолчанию итоговая конфигурация fa2_padding_free_bs256_compile_dynamic, лучшая по токенам в секунду
ATTENTION_IMPLEMENTATION: str = "flash_attention_2"
# Для torch.compile длина батча сразу помечается динамичсекой, а при padding free границы документов
# считаются в коллаторе, иначе каждая новая длина батча вызывает перекомпиляцию
COMPILE_DYNAMIC: bool = True

MODEL_CONFIG: dict[str, Any] = dict(
    hidden_size=2048,
    num_hidden_layers=12,
    num_attention_heads=16,
    num_key_value_heads=8,
    intermediate_size=8192,
    head_dim=128,
    use_cache=False,
)

TRAINING_CONFIG: dict[str, Any] = dict(
    output_dir=f"results/final_{TRAIN_SECONDS:g}s",
    # Настройки для экспериментов.
    per_device_train_batch_size=256,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    torch_compile=True,
    use_liger_kernel=False,
    activation_offloading=False,
    packing=False,
    padding_free=True,
    # Общие параметры обучения и замера.
    max_steps=1_000_000,
    learning_rate=3e-4,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=200,
    weight_decay=0.01,
    optim="adamw_torch_fused",
    bf16=True,
    tf32=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    packing_strategy="bfd",
    max_length=512,
    completion_only_loss=False,
    loss_type="nll",
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    logging_steps=10,
    logging_nan_inf_filter=False,
    report_to="none",
    save_strategy="no",
    eval_strategy="no",
    seed=42,
    data_seed=42,
    disable_tqdm=True,
)

# Настройки скорости исходного шаблона. Конфигурации из EXPERIMENTS задаются отличиями от них,
# attn_implementation задает ATTENTION_IMPLEMENTATION, compile_dynamic задает COMPILE_DYNAMIC
TEMPLATE_SPEED_CONFIG: dict[str, Any] = dict(
    attn_implementation="sdpa",
    compile_dynamic=False,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    gradient_checkpointing=False,
    torch_compile=False,
    use_liger_kernel=False,
    activation_offloading=False,
    packing=False,
    padding_free=False,
)

# Padding-free и packing корректно работают только с FlashAttention2, с SDPA документы
# в склеенном батче видят друг друга.
# PF64 это padding-free с микробатчем 64 без accumulation, то есть те же 64 примера на шаг, что в шаблоне
FA2 = dict(attn_implementation="flash_attention_2")
PF64 = dict(FA2, padding_free=True, per_device_train_batch_size=64, gradient_accumulation_steps=1)
COMPILE = dict(torch_compile=True, compile_dynamic=True)
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    # Отдельные механизмы
    "fa2": dict(FA2),
    "liger": dict(use_liger_kernel=True),
    "compile": dict(torch_compile=True),
    "compile_dynamic": dict(COMPILE),
    "bs64_ga1": dict(per_device_train_batch_size=64, gradient_accumulation_steps=1),
    "bs16_ga4": dict(per_device_train_batch_size=16, gradient_accumulation_steps=4),
    "checkpointing": dict(gradient_checkpointing=True),
    "offloading": dict(activation_offloading=True),
    # С SDPA offloading падает на выравнивании маски внимания, у FlashAttention2 маски-bias нет
    "fa2_offloading": dict(FA2, activation_offloading=True),
    # Padding-free и packing
    "fa2_padding_free": dict(FA2, padding_free=True),
    "fa2_padding_free_bs64": dict(PF64),
    "fa2_packing": dict(FA2, packing=True),
    # Сочетания с Liger
    "fa2_padding_free_bs64_liger": dict(PF64, use_liger_kernel=True),
    "fa2_padding_free_bs128_liger": dict(PF64, use_liger_kernel=True, per_device_train_batch_size=128),
    "fa2_padding_free_bs256_liger": dict(PF64, use_liger_kernel=True, per_device_train_batch_size=256),
    # Сочетания с torch.compile
    "fa2_padding_free_bs64_liger_compile": dict(PF64, use_liger_kernel=True, torch_compile=True),
    "fa2_packing_liger_compile": dict(FA2, packing=True, use_liger_kernel=True, torch_compile=True),
    "fa2_padding_free_bs64_liger_compile_dynamic": dict(PF64, **COMPILE, use_liger_kernel=True),
    "fa2_padding_free_bs128_liger_compile_dynamic": dict(
        PF64, **COMPILE, use_liger_kernel=True, per_device_train_batch_size=128
    ),
    "fa2_padding_free_bs256_liger_compile_dynamic": dict(
        PF64, **COMPILE, use_liger_kernel=True, per_device_train_batch_size=256
    ),
    "fa2_packing_liger_compile_dynamic": dict(FA2, **COMPILE, packing=True, use_liger_kernel=True),
    "fa2_padding_free_bs64_compile_dynamic": dict(PF64, **COMPILE),
    "fa2_padding_free_bs128_compile_dynamic": dict(PF64, **COMPILE, per_device_train_batch_size=128),
    # Итоговая конфигурация, она же значения по умолчанию
    "fa2_padding_free_bs256_compile_dynamic": dict(PF64, **COMPILE, per_device_train_batch_size=256),
}
EXPERIMENT: str | None = os.environ.get("EXPERIMENT") or None


def apply_experiment() -> None:
    global ATTENTION_IMPLEMENTATION, COMPILE_DYNAMIC
    if EXPERIMENT is None:
        return
    overrides = dict(TEMPLATE_SPEED_CONFIG, **EXPERIMENTS[EXPERIMENT])
    ATTENTION_IMPLEMENTATION = overrides.pop("attn_implementation", ATTENTION_IMPLEMENTATION)
    COMPILE_DYNAMIC = overrides.pop("compile_dynamic", COMPILE_DYNAMIC)
    TRAINING_CONFIG.update(overrides)
    TRAINING_CONFIG["output_dir"] = f"results/{EXPERIMENT}_{TRAIN_SECONDS:g}s"


def prepare_data() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "ai-forever/rugpt3small_based_on_gpt2",
        revision="a9307e696cd3c5b7f953ff4cb19d76a4d81821d5",
    )
    tokenizer.pad_token = tokenizer.eos_token
    source = load_dataset(
        "wikimedia/wikipedia",
        "20231101.ru",
        split="train",
        streaming=True,
        revision="b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    )
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for index, article in enumerate(source.take(50_500)):
        split = "validation" if index < 500 else "train"
        paragraphs = [
            p.strip() for p in article["text"].splitlines() if len(p.strip()) >= 80
        ][:8]
        if paragraphs:
            encoded = tokenizer(
                paragraphs,
                add_special_tokens=False,
                truncation=True,
                max_length=511,
                return_attention_mask=False,
            )["input_ids"]
            for paragraph_index, tokens in enumerate(encoded):
                if len(tokens) >= 32:
                    rows[split].append(
                        dict(
                            input_ids=tokens + [tokenizer.eos_token_id],
                            article_id=article["id"],
                            paragraph_index=paragraph_index,
                        )
                    )
        if (index + 1) % 5000 == 0:
            print(f"Подготовлено {index + 1}/50500 статей", flush=True)
    DatasetDict(
        {name: Dataset.from_list(items) for name, items in rows.items()}
    ).save_to_disk(str(DATA / "dataset"))
    tokenizer.save_pretrained(DATA / "tokenizer")


class CausalCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = super().torch_call(examples)
        batch["labels"][:, 0] = -100
        return batch


class VarlenCollator(DataCollatorForLanguageModeling):
    """Padding-free коллатор TRL, который сразу передает FlashAttention2 границы документов."""

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        position_ids = batch["position_ids"].view(-1)
        starts = (position_ids == 0).nonzero().view(-1)
        cu_seq_lens = torch.cat((starts, torch.tensor([position_ids.numel()]))).to(torch.int32)
        batch["cu_seq_lens_q"] = batch["cu_seq_lens_k"] = cu_seq_lens
        batch["max_length_q"] = batch["max_length_k"] = int(cu_seq_lens.diff().max())
        return batch


class BenchmarkTrainer(SFTTrainer):
    target_tokens: torch.Tensor

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ModelOutput]:
        if COMPILE_DYNAMIC:
            for key in ("input_ids", "labels", "attention_mask", "position_ids"):
                if key in inputs:
                    torch._dynamo.maybe_mark_dynamic(inputs[key], 1)
            for key in ("cu_seq_lens_q", "cu_seq_lens_k"):
                if key in inputs:
                    torch._dynamo.maybe_mark_dynamic(inputs[key], 0)
        return Trainer.compute_loss(
            self,
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.target_tokens += inputs["labels"].ne(-100).sum()
        return super().training_step(model, inputs, num_items_in_batch)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        for key in ("loss", "grad_norm", "train_loss"):
            if key in logs and not math.isfinite(logs[key]):
                raise FloatingPointError(f"Значение {key} не валидное")
        logs.pop("train_samples_per_second", None)
        logs.pop("train_steps_per_second", None)
        super().log(logs, start_time)


class Benchmark(TrainerCallback):
    start: float
    previous_end: float
    previous_tokens: int

    def __init__(self, trainer: BenchmarkTrainer, seconds: float) -> None:
        self.trainer, self.seconds = trainer, seconds
        self.steps: list[dict[str, float | int | bool]] = []

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start = self.previous_end = time.perf_counter()
        self.previous_tokens = 0

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        torch.cuda.synchronize()
        now = time.perf_counter()
        tokens = self.trainer.target_tokens.item()
        record: dict[str, float | int | bool] = dict(
            step=state.global_step,
            elapsed_seconds=now - self.start,
            step_seconds=now - self.previous_end,
            target_tokens=tokens - self.previous_tokens,
            warmup=state.global_step <= 10,
        )
        self.steps.append(record)
        if state.global_step % 10 == 0:
            print(json.dumps(record), flush=True)
        self.previous_end, self.previous_tokens = now, tokens
        if now - self.start >= self.seconds:
            control.should_training_stop = True

    def summary(self) -> dict[str, float | int | None]:
        measured = self.steps[10:]
        duration = sum(step["step_seconds"] for step in measured)
        tokens = sum(step["target_tokens"] for step in measured)
        elapsed = self.previous_end - self.start
        return dict(
            train_seconds=elapsed,
            optimizer_steps=len(self.steps),
            target_tokens=self.previous_tokens,
            tokens_per_second_including_warmup=self.previous_tokens / elapsed,
            warmup_steps=10,
            measured_steps=len(measured),
            measured_seconds=duration,
            tokens_per_second=tokens / duration if measured else None,
            seconds_per_optimizer_step=duration / len(measured) if measured else None,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, float | int]:
    loader = DataLoader(
        dataset.select_columns(["input_ids"]),
        batch_size=8,
        collate_fn=CausalCollator(tokenizer.pad_token_id, completion_only_loss=False),
    )
    model.eval()
    loss_sum, tokens = 0.0, 0
    for batch in loader:
        batch = {key: value.cuda() for key, value in batch.items()}
        count = batch["labels"].ne(-100).sum().item()
        loss_sum += model(**batch).loss.item() * count
        tokens += count
    loss = loss_sum / tokens

    return dict(validation_loss=loss, validation_target_tokens=tokens)


def main() -> None:
    apply_experiment()
    training_args = SFTConfig(**TRAINING_CONFIG)

    output = Path(training_args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(
        experiment=EXPERIMENT,
        train_seconds=TRAIN_SECONDS,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        compile_dynamic=COMPILE_DYNAMIC,
        training_config=TRAINING_CONFIG,
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if not (DATA / "dataset").exists():
        prepare_data()
    tokenizer = AutoTokenizer.from_pretrained(DATA / "tokenizer")
    dataset = load_from_disk(str(DATA / "dataset"))

    set_seed(training_args.seed)
    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG,
    )
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        torch_dtype=torch.bfloat16,
    )

    padding_free = training_args.packing or training_args.padding_free
    trainer = BenchmarkTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
        data_collator=None if padding_free else CausalCollator(tokenizer.pad_token_id),
    )
    if padding_free and COMPILE_DYNAMIC:
        # TRL не принимает свой коллатор при padding-free, поэтому он подменяется после создания тренера
        trainer.data_collator = VarlenCollator(
            tokenizer.pad_token_id,
            completion_only_loss=False,
            padding_free=True,
            pad_to_multiple_of=training_args.pad_to_multiple_of,
        )
    trainer.target_tokens = torch.zeros((), dtype=torch.int64, device="cuda")
    benchmark = Benchmark(trainer, TRAIN_SECONDS)
    trainer.add_callback(benchmark)

    trainer.train()
    summary = benchmark.summary()
    summary.update(
        evaluate(trainer.model, dataset["validation"].select(range(512)), tokenizer)
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
