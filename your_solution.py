import argparse
import json
import math
import os
import time

from datasets import Features, Sequence, Value, load_dataset
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
import torch


# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 15
MAX_LENGTH = 512
INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
LABELS = 'labels'
IGNORE_INDEX = -100

# Don't change these parameters
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000


# Hugging Face TrainingArguments. Any key can be overridden from the CLI: --set learning_rate=3e-4
TRAINING_CONFIG = {
    'output_dir': f'{OUTPUT_DIR}/gpt2-1b-russian',
    'optim': 'adamw_torch_fused',
    'num_train_epochs': 1,
    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 50,  # 5000 and 1000 are divisible by 50 -> every eval batch has equal weight
    'gradient_accumulation_steps': 1,
    'learning_rate': 1e-3,
    # Used only when time_lr_schedule is None; otherwise the time-based schedule below replaces it.
    'lr_scheduler_type': 'cosine',
    'warmup_steps': 100,
    'weight_decay': 0.1,
    'adam_beta1': 0.9,
    'adam_beta2': 0.95,
    'max_grad_norm': 1.0,
    'logging_steps': 5,
    # Evaluation is triggered by wall-clock time (PeriodicEvalCallback), so curves of runs with different
    # batch sizes are sampled at the same moments and the eval cost per run is the same.
    'eval_strategy': 'no',
    # Checkpointing a 1B model every N steps eats into the 15-minute budget. The loss is still decreasing at the
    # end of the run, so the last weights are the best ones; TimeoutCallback saves them once training stops.
    'save_strategy': 'no',
    'save_only_model': True,
    'load_best_model_at_end': False,
    'bf16': True,
    'tf32': True,
    'gradient_checkpointing': False,
    'dataloader_num_workers': 4,
    'dataloader_pin_memory': True,
    'dataloader_drop_last': True,
    'torch_compile': True,
    'report_to': 'none',
    'seed': 42,
}

# Options of this solution that are not TrainingArguments. Also overridable via --set.
SOLUTION_CONFIG = {
    # Strip padding and pack the whole batch into one sequence; FlashAttention-2 varlen keeps documents separate.
    # The loss is mathematically identical to the padded batch, but no compute is spent on pad tokens.
    'padding_free': True,
    'pad_to_multiple_of': 64,
    # Shorter packed batches are padded up to this length: torch.compile's matmul padding pass guards on
    # `2048 <= total_len`, and a batch below it triggers a ~40 s recompilation in the middle of training.
    'min_batch_tokens': 2048,
    # Keep fp32 master weights (+ bf16 autocast) instead of training pure bf16 weights.
    'fp32_master_weights': False,
    # LR schedule driven by wall-clock time, so the LR is annealed exactly when the time budget runs out
    # whatever the throughput is: 'wsd' | 'cosine' | 'linear' | 'constant' | None (use lr_scheduler_type + max_steps).
    'time_lr_schedule': 'wsd',
    'decay_fraction': 0.25,  # wsd: share of the budget spent on the final linear decay
    'min_lr_ratio': 0.02,
    'eval_subset_size': 1000,  # intermediate evals use the first N validation samples, the final one uses all 5000
    'eval_every_seconds': 150,
    'generate_after_training': True,
}

GENERATION_PROMPTS = [
    "Москва — столица",
    "Александр Сергеевич Пушкин родился",
    "Математика — это",
    "В 1812 году",
    "Река Волга",
    "Искусственный интеллект",
]


class TimeoutCallback(TrainerCallback):
    """Callback to stop training after a specified timeout."""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                control.should_training_stop = True
                # Include the final weights in best-checkpoint selection.
                control.should_evaluate = True
                control.should_save = True
                print(f"Training stopped after {elapsed:.2f} seconds")
        return control


class PeriodicEvalCallback(TrainerCallback):
    """Requests an evaluation every `every_seconds` of wall-clock training time."""
    def __init__(self, every_seconds):
        self.every_seconds = every_seconds
        self.next_eval = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.next_eval = time.time() + self.every_seconds

    def on_step_end(self, args, state, control, **kwargs):
        if self.next_eval is not None and time.time() >= self.next_eval:
            control.should_evaluate = True
            self.next_eval = time.time() + self.every_seconds
        return control


class TimeBudgetTrainer(Trainer):
    """Trainer with a wall-clock LR schedule and extra logging (elapsed time, number of trained tokens)."""

    def __init__(self, *args, solution_config, time_budget, **kwargs):
        super().__init__(*args, **kwargs)
        self.solution_config = solution_config
        self.time_budget = time_budget
        self.train_start = None
        self.tokens_seen = None

    def elapsed(self):
        return 0.0 if self.train_start is None else time.time() - self.train_start

    def train(self, *args, **kwargs):
        self.train_start = time.time()
        return super().train(*args, **kwargs)

    def _time_lr_lambda(self, step):
        cfg = self.solution_config
        warmup = self.args.warmup_steps
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, self.elapsed() / self.time_budget)
        min_ratio = cfg['min_lr_ratio']
        shape = cfg['time_lr_schedule']
        if shape == 'constant':
            return 1.0
        if shape == 'linear':
            return 1.0 - (1.0 - min_ratio) * progress
        if shape == 'cosine':
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        if shape == 'wsd':
            decay_start = 1.0 - cfg['decay_fraction']
            if progress <= decay_start:
                return 1.0
            return 1.0 - (1.0 - min_ratio) * (progress - decay_start) / cfg['decay_fraction']
        raise ValueError(f"Unknown time_lr_schedule: {shape}")

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None and self.solution_config['time_lr_schedule'] is not None:
            self.lr_scheduler = LambdaLR(optimizer if optimizer is not None else self.optimizer, self._time_lr_lambda)
            self._created_lr_scheduler = True
        return super().create_scheduler(num_training_steps, optimizer)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.args.torch_compile and 'cu_seq_lens_q' in inputs:
            # Packed batches change length every step: compile for a dynamic length right away instead of
            # compiling for the first batch's exact shape and recompiling on the second one.
            for key in (INPUT_IDS, LABELS, 'position_ids'):
                torch._dynamo.maybe_mark_dynamic(inputs[key], 1)
            for key in ('cu_seq_lens_q', 'cu_seq_lens_k'):
                torch._dynamo.maybe_mark_dynamic(inputs[key], 0)
        if model.training:
            # Kept on the GPU to avoid a host sync every step; it is read only when logging.
            n_tokens = (inputs[LABELS] != IGNORE_INDEX).sum()
            self.tokens_seen = n_tokens if self.tokens_seen is None else self.tokens_seen + n_tokens
        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

    def log(self, logs, start_time=None):
        logs['elapsed_sec'] = round(self.elapsed(), 1)
        logs['train_tokens'] = 0 if self.tokens_seen is None else int(self.tokens_seen)
        super().log(logs, start_time)


def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_function(examples, tokenizer):
    """
    Tokenizes texts into fixed MAX_LENGTH blocks (truncation + right padding).

    Labels are input_ids with IGNORE_INDEX on positions that must not contribute to the loss:
    - padding, except the first pad token: pad == eos, so it teaches the model where a document ends;
    - the first token, which has no context to be predicted from (HF drops it anyway when shifting labels,
      masking it explicitly makes the token count used for loss normalization exact).
    """
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
    )
    labels = []
    for ids, mask in zip(tokenized[INPUT_IDS], tokenized[ATTENTION_MASK]):
        n_real = sum(mask)
        n_scored = min(n_real + 1, MAX_LENGTH)  # real tokens + the eos that closes the document
        labels.append([IGNORE_INDEX] + ids[1:n_scored] + [IGNORE_INDEX] * (MAX_LENGTH - n_scored))
    return {
        LABELS: labels,
        INPUT_IDS: tokenized[INPUT_IDS],
        ATTENTION_MASK: tokenized[ATTENTION_MASK],
    }


def save_as_parquets(ds, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    os.makedirs(output_dir, exist_ok=True)
    for index in range(num_shards):
        # contiguous=True keeps the original order, so the first VALIDATION_SIZE samples stay the validation set.
        shard = ds.shard(num_shards=num_shards, index=index, contiguous=True)
        shard.to_parquet(f"{output_dir}/{index:05d}.parquet")


def prepare_dataset():
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")
    tokenizer = prepare_tokenizer()
    features = Features({
        LABELS: Sequence(Value('int32')),
        INPUT_IDS: Sequence(Value('int32')),
        ATTENTION_MASK: Sequence(Value('int8')),
    })
    tokenized = dataset.map(
        tokenize_function,
        fn_kwargs={'tokenizer': tokenizer},
        batched=True,
        num_proc=min(32, os.cpu_count() or 1),
        remove_columns=dataset.column_names,
        features=features,
        desc="Tokenizing",
    )
    save_as_parquets(tokenized)


def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.parquet'))
    data_files = [os.path.join(data_dir, f) for f in files]
    return load_dataset('parquet', data_files=data_files)['train']


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    dataset_size = len(dataset)
    train_dataset = dataset.select(range(validation_size, dataset_size))
    eval_dataset = dataset.select(range(validation_size))

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def create_model(tokenizer):
    # Don't change this parameter
    MODEL_CONFIG = {
        'hidden_size': 2048,
        'num_hidden_layers': 12,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 8192,
        'head_dim': 128,
        'hidden_act': 'silu',
        'initializer_range': 0.02,
        'scale_attn_weights': True,
        'use_cache': True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG
    )

    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16
    )

    print(f"Model pad token id: {model.config.pad_token_id}")

    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")

    return model


class PaddedCollator:
    """Stacks the fixed-length padded samples as they are stored."""

    def __call__(self, features):
        batch = {key: torch.tensor([f[key] for f in features], dtype=torch.long)
                 for key in (INPUT_IDS, ATTENTION_MASK, LABELS)}
        batch['use_cache'] = False  # no KV cache during training/eval forward passes
        return batch


class PaddingFreeCollator:
    """
    Removes padding and concatenates the batch into a single [1, total_len] sequence.

    position_ids restart from 0 for every document, and cu_seq_lens tell FlashAttention-2 (varlen kernel) where the
    documents start, so tokens never attend across documents. Each document keeps its tokens plus the closing eos, i.e.
    exactly the positions that have a label in the padded format, so the loss is the same as for the padded batch.
    max_length is the fixed upper bound MAX_LENGTH, which keeps it constant for torch.compile.
    """

    def __init__(self, pad_token_id, pad_to_multiple_of=None, min_batch_tokens=0):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.min_batch_tokens = min_batch_tokens

    def __call__(self, features):
        input_ids, labels, position_ids, cu_seq_lens = [], [], [], [0]
        for f in features:
            n_real = sum(f[ATTENTION_MASK])
            n_keep = min(n_real + 1, len(f[INPUT_IDS]))
            input_ids.extend(f[INPUT_IDS][:n_keep])
            labels.extend(f[LABELS][:n_keep])
            position_ids.extend(range(n_keep))
            cu_seq_lens.append(cu_seq_lens[-1] + n_keep)

        # Tail padding: fully-ignored "documents" of at most MAX_LENGTH tokens (max_length must stay an upper bound).
        n_pad = max(self.min_batch_tokens - len(input_ids), 0)
        if self.pad_to_multiple_of:
            n_pad += -(len(input_ids) + n_pad) % self.pad_to_multiple_of
        while n_pad > 0:
            chunk = min(n_pad, MAX_LENGTH)
            input_ids.extend([self.pad_token_id] * chunk)
            labels.extend([IGNORE_INDEX] * chunk)
            position_ids.extend(range(chunk))
            cu_seq_lens.append(cu_seq_lens[-1] + chunk)
            n_pad -= chunk

        cu_seq_lens = torch.tensor(cu_seq_lens, dtype=torch.int32)
        return {
            INPUT_IDS: torch.tensor([input_ids], dtype=torch.long),
            LABELS: torch.tensor([labels], dtype=torch.long),
            'position_ids': torch.tensor([position_ids], dtype=torch.long),
            'cu_seq_lens_q': cu_seq_lens,
            'cu_seq_lens_k': cu_seq_lens,
            'max_length_q': MAX_LENGTH,
            'max_length_k': MAX_LENGTH,
            'use_cache': False,
        }


def make_collator(solution_config, tokenizer):
    if solution_config['padding_free']:
        return PaddingFreeCollator(
            tokenizer.pad_token_id, solution_config['pad_to_multiple_of'], solution_config['min_batch_tokens'])
    return PaddedCollator()


def parse_overrides(pairs):
    """['learning_rate=1e-3', 'time_lr_schedule=null'] -> {'learning_rate': 0.001, 'time_lr_schedule': None}"""
    overrides = {}
    for pair in pairs or []:
        key, _, raw = pair.partition('=')
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        overrides[key.strip()] = value
    return overrides


def build_configs(run_name=None, overrides=None):
    training_config = dict(TRAINING_CONFIG)
    solution_config = dict(SOLUTION_CONFIG)
    if run_name:
        training_config['output_dir'] = f'{OUTPUT_DIR}/runs/{run_name}'
        training_config['run_name'] = run_name
    for key, value in (overrides or {}).items():
        if key in solution_config:
            solution_config[key] = value
        else:
            training_config[key] = value
    return training_config, solution_config


@torch.no_grad()
def generate_samples(model, tokenizer, prompts=GENERATION_PROMPTS, max_new_tokens=80):
    model.eval()
    device = next(model.parameters()).device
    modes = {
        'greedy': dict(do_sample=False, repetition_penalty=1.2),
        'sampling': dict(do_sample=True, temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.2),
    }
    samples = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        for mode, kwargs in modes.items():
            torch.manual_seed(0)
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=20,  # an undertrained model often emits eos right away
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                **kwargs,
            )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            samples.append({'prompt': prompt, 'mode': mode, 'text': text})
            print(f"[{mode}] {text}\n")
    return samples


def train_model(run_name=None, overrides=None):
    training_config, solution_config = build_configs(run_name, overrides)
    output_dir = training_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"TrainingArguments: {json.dumps(training_config, indent=2)}")
    print(f"Solution config: {json.dumps(solution_config, indent=2)}")

    tokenizer = prepare_tokenizer()
    train_dataset, eval_dataset = split_dataset(load_tokenized_dataset())
    eval_subset = eval_dataset.select(range(solution_config['eval_subset_size']))

    model = create_model(tokenizer)
    if solution_config['fp32_master_weights']:
        # bf16 compute still comes from autocast (bf16=True); only weights, grads and optimizer state become fp32.
        model = model.to(torch.float32)

    training_args = TrainingArguments(**training_config)
    trainer = TimeBudgetTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # Intermediate evals (logged as eval_subset_loss) run on a subset to save training time.
        eval_dataset={'subset': eval_subset},
        data_collator=make_collator(solution_config, tokenizer),
        solution_config=solution_config,
        time_budget=MAX_TRAINING_TIME_SECONDS,
        callbacks=[TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS)] # dont change
        )
    trainer.add_callback(PeriodicEvalCallback(solution_config['eval_every_seconds']))
    train_result = trainer.train()
    train_elapsed = trainer.elapsed()
    print("Running final evaluation...")
    eval_results = trainer.evaluate(eval_dataset=eval_dataset)  # full 5000-sample validation set -> eval_loss
    print(f"Final evaluation results: {eval_results}")
    trainer.save_state()  # writes trainer_state.json with log_history for the loss plots

    model = trainer.model.to(torch.bfloat16)
    final_dir = os.path.join(output_dir, 'final')
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    samples = []
    if solution_config['generate_after_training']:
        samples = generate_samples(model, tokenizer)

    summary = {
        'run_name': run_name,
        'training_config': training_config,
        'solution_config': solution_config,
        'eval_loss': eval_results['eval_loss'],
        'eval_perplexity': math.exp(eval_results['eval_loss']),
        'global_step': trainer.state.global_step,
        'train_seconds': train_elapsed,
        'train_tokens': 0 if trainer.tokens_seen is None else int(trainer.tokens_seen),
        'train_loss_avg': train_result.training_loss,
        'peak_memory_gb': torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        'generations': samples,
    }
    summary['train_tokens_per_sec'] = summary['train_tokens'] / max(train_elapsed, 1e-9)
    with open(os.path.join(output_dir, 'run_summary.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"eval_loss={summary['eval_loss']:.4f} ppl={summary['eval_perplexity']:.1f} "
          f"steps={summary['global_step']} tokens={summary['train_tokens']:,}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Mini-pretrain of a 1B Qwen3 on Russian Wikipedia")
    parser.add_argument('command', nargs='?', default='train', choices=['prepare', 'train'])
    parser.add_argument('--name', default=None, help="run name; outputs go to output_dir/runs/<name>")
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help="override a TRAINING_CONFIG or SOLUTION_CONFIG entry (value parsed as JSON)")
    return parser.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    if cli.command == 'prepare':
        # Step 1: Prepare the dataset (run once)
        prepare_dataset()
    else:
        # Step 2: Train the model
        train_model(cli.name, parse_overrides(cli.set))
