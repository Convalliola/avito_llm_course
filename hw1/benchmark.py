"""
Short runs for the system-level part of the study (run on the GPU machine).

  python benchmark.py check
      Compares the loss of the same batch in padded and padding-free format (must match up to bf16 noise).

  python benchmark.py speed --name bs32 --set per_device_train_batch_size=32
      Trains for `--warmup + --steps` steps and reports steady-state throughput (trained tokens/s, i.e. tokens that
      have a label, padding excluded), time of the first step (includes torch.compile) and peak memory.
      Results are appended to output_dir/benchmarks.jsonl.

  python benchmark.py replay output_dir/runs/<run>/nan_dump
      Replays the batch saved by `--set debug_nan=true` on the saved weights in several formats (packed as in training,
      without cu_seq_lens, without tail padding, padded) and reports which ones give non-finite gradients, then reruns
      the packed one under autograd anomaly detection to name the operation that produced the NaN.
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, TrainerCallback, TrainingArguments

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


def unpack_documents(batch, n_docs):
    """Splits a packed batch back into its document segments (the tail padding segments come after them)."""
    cu = batch['cu_seq_lens_q'].tolist()
    return [(batch[S.INPUT_IDS][0, a:b].tolist(), batch[S.LABELS][0, a:b].tolist())
            for a, b in zip(cu[:n_docs], cu[1:n_docs + 1])]


def to_padded(docs, pad_token_id):
    """Rebuilds the padded batch: the appended eos is a pad position (attention mask 0) that still carries a label."""
    rows = []
    for ids, labels in docs:
        n_real = len(ids) - 1 if ids[-1] == pad_token_id else len(ids)
        rows.append({
            S.INPUT_IDS: ids + [pad_token_id] * (S.MAX_LENGTH - len(ids)),
            S.ATTENTION_MASK: [1] * n_real + [0] * (S.MAX_LENGTH - n_real),
            S.LABELS: labels + [S.IGNORE_INDEX] * (S.MAX_LENGTH - len(labels)),
        })
    return S.PaddedCollator()(rows)


def run_replay(dump_dir):
    dump = torch.load(os.path.join(dump_dir, 'batch.pt'))
    batch, n_docs = dump['batch'], dump['n_docs']
    on_gpu = torch.cuda.is_available()
    device = 'cuda' if on_gpu else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(dump_dir, 'model'), torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2' if on_gpu else 'sdpa').to(device)
    model.train()
    pad_id = model.config.pad_token_id

    lengths = (batch['cu_seq_lens_q'][1:] - batch['cu_seq_lens_q'][:-1]).tolist()
    print(f"step {dump['step']}, logged loss {dump['loss']}, {len(dump['bad_grads'])} params with bad grads: "
          f"{dump['bad_grads'][:8]}")
    print(f"packed length {batch[S.INPUT_IDS].shape[1]}, {n_docs} documents, segment lengths: {lengths}")

    docs = unpack_documents(batch, n_docs)
    n_doc_tokens = sum(len(ids) for ids, _ in docs)
    no_tail = {k: (v[:, :n_doc_tokens] if k in (S.INPUT_IDS, S.LABELS, 'position_ids') else v)
               for k, v in batch.items()}
    no_tail['cu_seq_lens_q'] = no_tail['cu_seq_lens_k'] = batch['cu_seq_lens_q'][:n_docs + 1]
    variants = {
        'packed (as in training)': batch,
        'packed, position_ids only': {k: v for k, v in batch.items()
                                      if not k.startswith(('cu_seq_lens', 'max_length'))},
        'packed, no tail padding': no_tail,
        'padded (same documents)': to_padded(docs, pad_id),
    }

    def forward_backward(inputs):
        model.zero_grad(set_to_none=True)
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        with torch.autocast(device, dtype=torch.bfloat16, enabled=on_gpu):
            loss = model(**inputs).loss
        loss.backward()
        bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        return loss.item(), bad

    for title, inputs in variants.items():
        loss, bad = forward_backward(inputs)
        print(f"{title:<28} loss {loss:9.4f}   params with non-finite grads: {len(bad)}  {bad[:3]}")

    print("\nAnomaly detection on the packed batch (names the backward op that produced NaN):")
    try:
        with torch.autograd.detect_anomaly():
            forward_backward(batch)
        print("no anomaly detected")
    except RuntimeError as error:
        print(str(error)[:1500])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['check', 'speed', 'replay'])
    parser.add_argument('dump_dir', nargs='?', help="replay: directory with the NaN dump")
    parser.add_argument('--name', default='default')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--warmup', type=int, default=15, help="steps excluded from the measurement (compile etc.)")
    parser.add_argument('--steps', type=int, default=40, help="measured steps")
    args = parser.parse_args()
    if args.command == 'check':
        run_check()
    elif args.command == 'replay':
        run_replay(args.dump_dir)
    else:
        run_speed(args.name, S.parse_overrides(args.set), args.warmup, args.steps)


if __name__ == '__main__':
    main()
