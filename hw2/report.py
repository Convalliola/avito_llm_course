"""Собирает results/*/summary.json и config.json в таблицу report/results.md и графики report/*.png."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results")
REPORT = Path("report")
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def describe(config: dict) -> str:
    """Короткое описание настроек, отличающих конфигурацию от шаблона."""
    tc = config["training_config"]
    parts = [config["attn_implementation"], f"bs {tc['per_device_train_batch_size']}×{tc['gradient_accumulation_steps']}"]
    flags = {
        "packing": "packing",
        "padding_free": "padding-free",
        "use_liger_kernel": "liger",
        "torch_compile": "compile",
        "gradient_checkpointing": "checkpointing",
        "activation_offloading": "offloading",
    }
    parts += [label for key, label in flags.items() if tc.get(key)]
    if config.get("compile_dynamic"):
        parts.append("dynamic")
    return ", ".join(parts)


def load_runs() -> tuple[list[dict], list[str]]:
    runs, failed = [], []
    for directory in sorted(RESULTS.iterdir()):
        # Папки _failed_ это прошлые попытки, перезапущенные под тем же именем
        if not directory.is_dir() or directory.name == "logs" or "_failed_" in directory.name:
            continue
        summary_path, config_path = directory / "summary.json", directory / "config.json"
        if not summary_path.exists() or not config_path.exists():
            failed.append(directory.name)
            continue
        config = json.loads(config_path.read_text())
        runs.append(dict(
            name=config["experiment"] or "final",
            seconds=int(config["train_seconds"]),
            setup=describe(config),
            **json.loads(summary_path.read_text()),
        ))
    return runs, failed


def last_error(name: str) -> str:
    base = name.split("_failed_")[0]
    log = RESULTS / "logs" / f"{base}.log"
    if not log.exists():
        return ""
    lines = [line for line in log.read_text(errors="replace").splitlines() if "Error" in line or "error" in line]
    return lines[-1].strip()[:160] if lines else ""


def table(runs: list[dict]) -> str:
    baselines = {r["seconds"]: r["tokens_per_second"] for r in runs if r["name"] == "baseline"}
    rows = [
        "| конфигурация | настройки | длительность, с | токенов/с | ускорение к бейзлайну | пик allocated, GiB | "
        "пик reserved, GiB | val loss | шагов оптимизатора | с/шаг |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(runs, key=lambda r: (r["seconds"], -(r["tokens_per_second"] or 0))):
        base = baselines.get(r["seconds"])
        speedup = f"{r['tokens_per_second'] / base:.2f}×" if base and r["tokens_per_second"] else ""
        # Без замера скорости, если за отведенное время прошло не больше 10 шагов разогрева
        measured = r["tokens_per_second"] is not None
        tokens = f"{r['tokens_per_second']:,.0f}" if measured else "нет замера"
        step = f"{r['seconds_per_optimizer_step']:.2f}" if measured else ""
        rows.append(
            f"| {r['name']} | {r['setup']} | {r['seconds']} | {tokens} | {speedup} | "
            f"{r['peak_allocated_gib']:.1f} | {r['peak_reserved_gib']:.1f} | {r['validation_loss']:.3f} | "
            f"{r['optimizer_steps']} | {step} |"
        )
    return "\n".join(rows)


def style(ax, title: str, xlabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def bar_chart(runs: list[dict], key: str, title: str, xlabel: str, path: Path, fmt: str) -> None:
    runs = sorted((r for r in runs if r[key] is not None), key=lambda r: r[key])
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(runs) + 1.3))
    values = [r[key] for r in runs]
    colors = [ORANGE if r["name"] == "baseline" else BLUE for r in runs]
    ax.barh(range(len(runs)), values, color=colors, height=0.6)
    ax.set_yticks(range(len(runs)), [r["name"] for r in runs])
    for i, v in enumerate(values):
        ax.text(v, i, " " + format(v, fmt), va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(values) * 1.15)
    style(ax, title, xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    REPORT.mkdir(exist_ok=True)
    runs, failed = load_runs()
    lines = ["# Результаты экспериментов", "", table(runs), ""]

    # S считается для итогового запуска train.py с настройками по умолчанию, а без него для лучшего по 300 с
    final = [r for r in runs if r["seconds"] == 300]
    base = next((r for r in final if r["name"] == "baseline"), None)
    best = next((r for r in final if r["name"] == "final"), None) or max(
        (r for r in final if r["name"] != "baseline"), key=lambda r: r["tokens_per_second"] or 0, default=None
    )
    if base and best:
        speedup = best["tokens_per_second"] / base["tokens_per_second"]
        lines += [f"**S = {best['tokens_per_second']:,.0f} / {base['tokens_per_second']:,.0f} = {speedup:.2f}** "
                  f"(итоговый запуск {best['name']} против бейзлайна, по 300 с)", ""]

    if failed:
        lines += ["## Упавшие запуски", ""]
        lines += [f"- {name}: {last_error(name)}" for name in failed]
        lines.append("")

    short_seconds = max((r["seconds"] for r in runs if r["seconds"] < 300), default=None)
    short = [r for r in runs if r["seconds"] == short_seconds]
    if short:
        bar_chart(short, "tokens_per_second", f"Полезные токены в секунду, короткие запуски по {short_seconds} с",
                  "токенов/с", REPORT / "tokens_per_second.png", ",.0f")
        bar_chart(short, "peak_allocated_gib", f"Пиковая память GPU, короткие запуски по {short_seconds} с",
                  "GiB", REPORT / "peak_memory.png", ".1f")
        lines += ["![tokens per second](tokens_per_second.png)", "", "![peak memory](peak_memory.png)", ""]

    (REPORT / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
