"""
Short runs for the system-level part of the study (run on the GPU machine).

  python benchmark.py check
      Compares the loss of the same batch in padded and padding-free format (must match up to bf16 noise).

  python benchmark.py speed --name bs32 --set per_device_train_batch_size=32
      Trains for `--warmup + --steps` steps and reports steady-state throughput (trained tokens/s, i.e. tokens that
      have a label, padding excluded), time of the first step (includes torch.compile) and peak memory.
      Results are appended to output_dir/benchmarks.jsonl.
"""
import argparse
import json
import os
import time

import torch
from transformers import TrainerCallback, TrainingArguments

import your_solution as S


class ThroughputCallback(TrainerCallback):
    def __init__(self, warmup_steps):
        self.warmup_steps = warmup_steps
        self.trainer = None
        self.train_begin = None
        self.first_step_sec = None
        self.mark = None
        self.result = {}

    def _tokens(self):
        return 0 if self.trainer.tokens_seen is None else int(self.trainer.tokens_seen)

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_begin = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        now = time.time()
        if state.global_step == 1:
            self.first_step_sec = now - self.train_begin
        if state.global_step == self.warmup_steps:
            self.mark = (now, self._tokens(), state.global_step)
        elif state.global_step == args.max_steps and self.mark is not None:
            t0, tok0, step0 = self.mark
            dt, steps = now - t0, state.global_step - step0
            self.result = {
                'tokens_per_sec': (self._tokens() - tok0) / dt,
                'sec_per_step': dt / steps,
                'samples_per_sec': steps * args.per_device_train_batch_size * args.gradient_accumulation_steps / dt,
            }


def run_speed(name, overrides, warmup, steps):
    training_config, solution_config = S.build_configs(f'bench_{name}', overrides)
    training_config.update(max_steps=warmup + steps, logging_steps=warmup + steps, save_strategy='no')
    tokenizer = S.prepare_tokenizer()
    train_dataset, _ = S.split_dataset(S.load_tokenized_dataset())

    model = S.create_model(tokenizer)
    if solution_config['fp32_master_weights']:
        model = model.to(torch.float32)

    timer = ThroughputCallback(warmup)
    trainer = S.TimeBudgetTrainer(
        model=model,
        args=TrainingArguments(**training_config),
        train_dataset=train_dataset,
        data_collator=S.make_collator(solution_config, tokenizer),
        solution_config=solution_config,
        time_budget=S.MAX_TRAINING_TIME_SECONDS,
        callbacks=[timer],
    )
    timer.trainer = trainer
    record = {'name': name, 'overrides': overrides}
    try:
        trainer.train()
        record.update(timer.result)
        record['first_step_sec'] = timer.first_step_sec
        record['last_logged_loss'] = next(h['loss'] for h in reversed(trainer.state.log_history) if 'loss' in h)
        record['peak_memory_gb'] = torch.cuda.max_memory_allocated() / 2**30
    except torch.cuda.OutOfMemoryError:
        record['error'] = 'OOM'
    print(json.dumps(record, indent=2))
    with open(os.path.join(S.OUTPUT_DIR, 'benchmarks.jsonl'), 'a') as f:
        f.write(json.dumps(record) + '\n')


@torch.no_grad()
def run_check(n_samples=16):
    tokenizer = S.prepare_tokenizer()
    train_dataset, _ = S.split_dataset(S.load_tokenized_dataset())
    rows = [train_dataset[i] for i in range(n_samples)]
    model = S.create_model(tokenizer).cuda().eval()

    def forward(batch):
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(**batch)
        return out.loss.item(), out.logits.float()

    padded = S.PaddedCollator()(rows)
    flat = S.PaddingFreeCollator(tokenizer.pad_token_id, pad_to_multiple_of=64)(rows)
    flat_position_ids_only = {k: v for k, v in flat.items() if not k.startswith(('cu_seq_lens', 'max_length'))}
    # Without document boundaries tokens attend to the previous documents -> logits must differ.
    no_boundaries = dict(flat_position_ids_only)
    no_boundaries['position_ids'] = torch.arange(flat['input_ids'].shape[1]).unsqueeze(0)

    n_real = sum(sum(r[S.ATTENTION_MASK]) for r in rows)
    print(f"real tokens: {n_real} / {n_samples * S.MAX_LENGTH} "
          f"({100 * (1 - n_real / (n_samples * S.MAX_LENGTH)):.1f}% of the padded batch is padding)")

    # A random-init model has ~uniform loss, so compare logits of the real tokens rather than only the loss.
    padded_loss, padded_logits = forward(padded)
    starts = flat['cu_seq_lens_q'].tolist()
    reference = torch.cat([padded_logits[i, :starts[i + 1] - starts[i]] for i in range(n_samples)])
    print(f"{'variant':<34}{'loss':>10}{'max |dlogit|':>16}{'mean |dlogit|':>16}")
    print(f"{'padded (reference)':<34}{padded_loss:>10.5f}")
    for title, batch in [('padding-free, cu_seq_lens', flat),
                         ('padding-free, position_ids only', flat_position_ids_only),
                         ('packed WITHOUT boundaries (bad)', no_boundaries)]:
        loss, logits = forward(batch)
        diff = (logits[0, :starts[n_samples]] - reference).abs()
        print(f"{title:<34}{loss:>10.5f}{diff.max().item():>16.4f}{diff.mean().item():>16.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['check', 'speed'])
    parser.add_argument('--name', default='default')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--warmup', type=int, default=15, help="steps excluded from the measurement (compile etc.)")
    parser.add_argument('--steps', type=int, default=40, help="measured steps")
    args = parser.parse_args()
    if args.command == 'check':
        run_check()
    else:
        run_speed(args.name, S.parse_overrides(args.set), args.warmup, args.steps)


if __name__ == '__main__':
    main()
