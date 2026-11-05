import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

# этот файл писал клод для воспроизводимости картинок и табличек

RES = Path(__file__).parent / "results"
PLOTS = RES / "plots"
TARGET_NAMES = {"all": "все линейные", "attn": "q k v o", "qv": "q v", "mlp": "gate up down"}
GROUPS = {
    "r": ["lora_r4", "lora_r8", "lora_base", "lora_r32", "lora_r64", "lora_r128"],
    "alpha": ["lora_alpha8", "lora_alpha16", "lora_base", "lora_alpha64", "lora_alpha128"],
    "dropout": ["lora_dropout0.0", "lora_base", "lora_dropout0.1", "lora_dropout0.2"],
    "target_modules": ["lora_qv", "lora_attn", "lora_mlp", "lora_base"],
}
# цвета проверены на различимость, на точечном графике их не больше трех
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.8,
    "lines.linewidth": 1.8, "lines.markersize": 6, "font.size": 10, "figure.facecolor": "white",
})
FMT = {
    "обучаемых, млн": "{:.2f}", "доля, %": "{:.2f}", "EM": "{:.2f}", "F1": "{:.2f}", "NLL ответа": "{:.4f}",
    "MMLU": "{:.2f}", "dev loss": "{:.4f}", "длина общих ответов": "{:.0f}", "обучение, мин": "{:.1f}", "ток/с": "{:.0f}",
    "пик памяти, ГБ": "{:.1f}", "lr": "{:g}", "r": "{:.0f}", "alpha": "{:.0f}", "dropout": "{:g}",
}


class Comma(ScalarFormatter):
    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def comma(x, f="{:g}"):
    return f.format(x).replace(".", ",")


def short_lr(lr):
    m, e = f"{lr:.0e}".split("e")
    return f"{m}e{int(e)}"


def general_len(r):
    # средняя длина ответа на 4 обычные просьбы
    out = [m["output"] for m in r["manual"] if m["kind"] == "general"]
    return sum(map(len, out)) / len(out)


def row(r):
    lora = r["method"] == "lora"
    dev = [h["eval_loss"] for h in r.get("log_history", []) if "eval_loss" in h]
    return {
        "эксперимент": r["name"], "r": r.get("r") if lora else None, "alpha": r.get("alpha") if lora else None,
        "dropout": r.get("dropout") if lora else None, "модули": TARGET_NAMES.get(r.get("targets")) if lora else None,
        "lr": r.get("lr"), "обучаемых, млн": r["trainable_params"] / 1e6,
        "доля, %": 100 * r["trainable_params"] / r["total_params"],
        "EM": 100 * r["sberquad_em"], "F1": 100 * r["sberquad_f1"], "NLL ответа": r["sberquad_nll"],
        "MMLU": 100 * r["mmlu_acc"], "dev loss": dev[-1] if dev else None, "длина общих ответов": general_len(r),
        "обучение, мин": r["train_time"] / 60 if "train_time" in r else None,
        "ток/с": r.get("tokens_per_second"), "пик памяти, ГБ": r.get("peak_mem_allocated_gb"),
    }


def md(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, rw in df.iterrows():
        cells = []
        for col, v in rw.items():
            if v is None or (isinstance(v, float) and pd.isna(v)):
                cells.append("")
            elif col in FMT and isinstance(v, (int, float)):
                cells.append(comma(v, FMT[col]))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_dev_loss(res):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, color in [("full_ft", BLUE), ("lora_base", ORANGE)]:
        h = [x for x in res[name]["log_history"] if "eval_loss" in x]
        steps, loss = [x["step"] for x in h], [x["eval_loss"] for x in h]
        ax.plot(steps, loss, marker="o", color=color, label=name)
        ax.annotate(comma(loss[-1], "{:.4f}"), (steps[-1], loss[-1]), textcoords="offset points",
                    xytext=(6, -10 if name == "full_ft" else 4), color=INK, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("шаг")
    ax.set_ylabel("loss на dev, лог. шкала")
    ax.set_title("Loss на dev во время обучения", color=INK)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS / "dev_loss_full_vs_lora.png", dpi=130)
    plt.close(fig)


def plot_groups(res):
    # по столбцам параметры, по строкам F1, NLL и MMLU, пунктир это full FT с lr 1e-5
    metrics = [("sberquad_f1", 100, "F1"), ("sberquad_nll", 1, "NLL ответа"), ("mmlu_acc", 100, "MMLU")]
    cols = {"r": "r", "alpha": "alpha", "dropout": "dropout", "target_modules": "модули"}
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharey="row")
    for j, (group, names) in enumerate(GROUPS.items()):
        names = [n for n in names if n in res]
        labels = [row(res[n])[cols[group]] for n in names]
        labels = [comma(v) if isinstance(v, (int, float)) else (v.replace(" ", "\n", 1) if len(v) > 8 else v) for v in labels]
        for i, (key, scale, title) in enumerate(metrics):
            ax = axes[i, j]
            ax.plot(labels, [scale * res[n][key] for n in names], marker="o", color=BLUE)
            ref = scale * res["full_ft"][key]
            ax.axhline(ref, color=MUTED, linestyle="--", linewidth=1)
            ax.yaxis.set_major_formatter(Comma())
            if j == 0:
                ax.set_ylabel(title)
                ax.annotate("Full FT", (0, ref), textcoords="offset points", xytext=(0, 4), color=MUTED, fontsize=9)
            if i == 0:
                ax.set_title(group, color=INK)
    fig.tight_layout()
    fig.savefig(PLOTS / "lora_params.png", dpi=130)
    plt.close(fig)


def plot_tradeoff(res):
    pt = lambda n: (res[n]["sberquad_nll"], 100 * res[n]["mmlu_acc"])
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    full = sorted([n for n in res if res[n]["method"] == "full"], key=lambda n: res[n]["lr"])
    step = [n for n in ["lora_alpha8", "lora_alpha16", "lora_base", "lora_alpha64", "lora_alpha128"] if n in res]
    lrs = [n for n in ["lora_lr5e-5", "lora_lr8e-4"] if n in res]
    other = [n for n in res if res[n]["method"] == "lora" and n not in step + lrs]
    ax.plot(*zip(*[pt(n) for n in full]), marker="o", color=BLUE, label="Full FT, lr от 2e-6 до 2e-5")
    ax.plot(*zip(*[pt(n) for n in step]), marker="o", color=ORANGE, label="LoRA r 16, alpha от 8 до 128")
    if lrs:
        ax.scatter(*zip(*[pt(n) for n in lrs]), s=60, facecolors="white", edgecolors=ORANGE, linewidths=1.8,
                   label="LoRA r 16, alpha 32, lr 5e-5 и 8e-4", zorder=3)
    ax.scatter(*zip(*[pt(n) for n in other]), s=36, color=GREEN, edgecolors="white", linewidths=1,
               label="LoRA, другие r, dropout, модули, сиды", zorder=3)
    labels = {n: f"lr {short_lr(res[n]['lr'])}" for n in full + lrs}
    labels |= {n: f"alpha {res[n]['alpha']}" for n in step}
    labels |= {"lora_qv": "q v", "lora_attn": "q k v o", "lora_mlp": "gate up down", "lora_qv_r128": "q v, r 128",
               "lora_r128_alpha256": "r 128, alpha 256"}
    # сдвиги подписей подобраны вручную, чтобы не налезали друг на друга
    shift = {"full_ft": (-8, -15), "lora_alpha64": (7, 2), "lora_alpha16": (4, -14), "lora_mlp": (-30, 8),
             "lora_alpha8": (4, -14), "lora_attn": (-10, 7), "lora_alpha128": (7, -3)}
    for n, text in labels.items():
        if n in res:
            ax.annotate(text, pt(n), textcoords="offset points", xytext=shift.get(n, (6, 4)), color=INK, fontsize=8.5)
    base = 100 * res["base"]["mmlu_acc"]
    ax.axhline(base, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate(f"исходная модель, MMLU {comma(base, '{:.1f}')} при NLL 0,92", (ax.get_xlim()[0], base),
                textcoords="offset points", xytext=(4, 4), color=MUTED, fontsize=9)
    ax.xaxis.set_major_formatter(Comma())
    ax.yaxis.set_major_formatter(Comma())
    ax.set_xlabel("NLL ответа на тесте SberQuAD (меньше значит лучше выучена задача)")
    ax.set_ylabel("MMLU, % (выше значит меньше забыто)")
    ax.set_title("Выученное против забытого", color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS / "tradeoff.png", dpi=130)
    plt.close(fig)


def manual_md(res, names):
    blocks = []
    for i, item in enumerate(res[names[0]]["manual"]):
        if item["kind"] == "sberquad":
            head = f"**{i + 1}. SberQuAD {item['id']}.** {item['question']}\n\nэталон `{item['answer']}`"
        else:
            head = f"**{i + 1}. Вне задачи.** {item['prompt']}"
        lines = [head, ""] + [f"- {n}: " + res[n]["manual"][i]["output"].replace("\n", " ") for n in names]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


every = {p.stem: json.loads(p.read_text()) for p in sorted(RES.glob("*.json"))}
res = {n: r for n, r in every.items() if not r.get("profile_only")}
PLOTS.mkdir(parents=True, exist_ok=True)
main_names = [n for n in ["base", "full_ft", "lora_base"] if n in res]
out = ["## Base, Full FT и LoRA", md(pd.DataFrame([row(res[n]) for n in main_names]))]
for group, names in GROUPS.items():
    out += [f"## LoRA, {group}", md(pd.DataFrame([row(res[n]) for n in names if n in res]))]

prof = [{"эксперимент": n, "gradient checkpointing": "да" if every[n].get("gradient_checkpointing") else "нет",
         "обучаемых, млн": every[n]["trainable_params"] / 1e6, "ток/с": every[n]["tokens_per_second"],
         "пик памяти, ГБ": every[n]["peak_mem_allocated_gb"]}
        for n in ["full_ft", "full_ft_gc", "lora_base", "lora_base_gc"] if n in every]
seeds = pd.DataFrame([row(res[n]) for n in ["lora_base", "lora_base_seed1", "lora_base_seed2"] if n in res])
seeds = seeds[["эксперимент", "EM", "F1", "NLL ответа", "MMLU", "dev loss"]]
seeds.loc[len(seeds)] = ["разброс max-min"] + [seeds[c].max() - seeds[c].min() for c in seeds.columns[1:]]
out += ["## Память и скорость", md(pd.DataFrame(prof)), "## Шум между сидами", md(seeds)]

done = {n for names in GROUPS.values() for n in names} | set(main_names) | {"lora_base_seed1", "lora_base_seed2"}
others = [n for n in res if n not in done]
if others:
    out += ["## Остальные", md(pd.DataFrame([row(res[n]) for n in others]))]
(RES / "tables.md").write_text("\n\n".join(out) + "\n")
print("\n\n".join(out))

plot_dev_loss(res)
plot_groups(res)
plot_tradeoff(res)
(RES / "manual_eval.md").write_text(manual_md(res, main_names) + "\n")
