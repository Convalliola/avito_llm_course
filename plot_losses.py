"""
Builds loss plots and a results table from output_dir/runs/*/{trainer_state.json, run_summary.json}.

Runs are grouped by the prefix of their name before the first '_' (base_, lr_, sched_, bs_, dtype_ ...), one figure
per group: train loss / eval loss (1000-sample validation subset) / learning rate against wall-clock minutes.
Extra groups can be given explicitly, e.g. to compare schedulers against the best run of the LR sweep:
    python plot_losses.py --group sched=lr_1e-3,sched_cosine,sched_constant
Usage: python plot_losses.py [--runs-dir output_dir/runs] [--out report] [--group NAME=run1,run2 ...]
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Categorical slots in fixed order (validated default palette); a group never has more than 8 runs.
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3dc'


def load_runs(runs_dir):
    runs = {}
    for name in sorted(os.listdir(runs_dir)):
        state_path = os.path.join(runs_dir, name, 'trainer_state.json')
        if not os.path.exists(state_path):
            continue
        with open(state_path) as f:
            history = json.load(f)['log_history']
        summary_path = os.path.join(runs_dir, name, 'run_summary.json')
        summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
        runs[name] = {'history': history, 'summary': summary}
    return runs


def ema(values, alpha=0.1):
    smoothed, current = [], None
    for v in values:
        current = v if current is None else alpha * v + (1 - alpha) * current
        smoothed.append(current)
    return smoothed


def series(history, key):
    points = [(h['elapsed_sec'] / 60, h[key]) for h in history if key in h and 'elapsed_sec' in h]
    return [p[0] for p in points], [p[1] for p in points]


def style(ax, title, ylabel):
    ax.set_title(title, loc='left', fontsize=11, color=INK)
    ax.set_xlabel('training time, min', color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)


def plot_group(group, names, runs, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for color, name in zip(COLORS, names):
        history = runs[name]['history']
        eval_loss = runs[name]['summary'].get('eval_loss')
        label = name if eval_loss is None else f'{name} (eval_loss {eval_loss:.3f})'

        x, y = series(history, 'loss')
        axes[0].plot(x, y, color=color, linewidth=0.8, alpha=0.25)
        axes[0].plot(x, ema(y), color=color, linewidth=2, label=label)

        x, y = series(history, 'eval_subset_loss')
        axes[1].plot(x, y, color=color, linewidth=2, marker='o', markersize=5, label=label)

        x, y = series(history, 'learning_rate')
        axes[2].plot(x, y, color=color, linewidth=2, label=label)

    style(axes[0], 'Train loss (EMA; raw faint)', 'loss')
    style(axes[1], 'Eval loss, 1000-sample validation subset', 'loss')
    style(axes[2], 'Learning rate', 'lr')
    # Early losses (~11) would squash the interesting part of the curves.
    for ax in axes[:2]:
        lows = [l.get_ydata().min() for l in ax.get_lines() if len(l.get_ydata())]
        if lows:
            ax.set_ylim(min(lows) - 0.2, min(lows) + 4)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(f'Experiment group: {group}', x=0.01, ha='left', fontsize=13, color=INK)
    fig.tight_layout()
    path = os.path.join(out_dir, f'loss_{group}.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_final(runs, out_dir):
    items = sorted(((r['summary']['eval_loss'], n) for n, r in runs.items() if 'eval_loss' in r['summary']),
                   reverse=True)
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(items) + 1.2))
    values = [v for v, _ in items]
    ax.barh(range(len(items)), values, color=COLORS[0], height=0.6)
    ax.set_yticks(range(len(items)), [n for _, n in items])
    for i, v in enumerate(values):
        ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9, color=INK)
    ax.set_xlim(min(values) - 0.5, max(values) + 0.4)
    style(ax, 'Final eval_loss on the full 5000-sample validation set (lower is better)', '')
    ax.set_xlabel('eval_loss', color=MUTED)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    path = os.path.join(out_dir, 'final_eval_loss.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def results_table(runs):
    rows = ['| run | eval_loss | ppl | steps | trained tokens, M | tokens/s | peak mem, GB |',
            '|---|---|---|---|---|---|---|']
    for name, run in sorted(runs.items(), key=lambda kv: kv[1]['summary'].get('eval_loss', 1e9)):
        s = run['summary']
        if 'eval_loss' not in s:
            continue
        mem = s.get('peak_memory_gb')
        rows.append(f"| {name} | {s['eval_loss']:.4f} | {s['eval_perplexity']:.1f} | {s['global_step']} | "
                    f"{s['train_tokens'] / 1e6:.1f} | {s['train_tokens_per_sec']:,.0f} | "
                    f"{'' if mem is None else f'{mem:.1f}'} |")
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', default='output_dir/runs')
    parser.add_argument('--out', default='report')
    parser.add_argument('--group', action='append', default=[], metavar='NAME=run1,run2')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = load_runs(args.runs_dir)
    groups = defaultdict(list)
    for name in runs:
        groups[name.split('_')[0]].append(name)
    for spec in args.group:
        group, _, names = spec.partition('=')
        groups[group] = [n for n in names.split(',') if n in runs]
    for group, names in groups.items():
        for chunk_start in range(0, len(names), len(COLORS)):
            suffix = '' if chunk_start == 0 else f'_{chunk_start // len(COLORS) + 1}'
            print(plot_group(group + suffix, names[chunk_start:chunk_start + len(COLORS)], runs, args.out))
    print(plot_final(runs, args.out))

    table = results_table(runs)
    with open(os.path.join(args.out, 'results.md'), 'w') as f:
        f.write(table + '\n')
    print(table)


if __name__ == '__main__':
    main()
