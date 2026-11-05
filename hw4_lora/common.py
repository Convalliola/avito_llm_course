import json
import random
import re
from collections import Counter
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
OUTPUTS_DIR = ROOT / "outputs"

PROMPT = "Прочитай текст и ответь на вопрос. Ответ должен быть точным фрагментом текста, без пояснений.\n\nТекст: {context}\n\nВопрос: {question}"


def prompt_messages(row):
    return [{"role": "user", "content": PROMPT.format(context=row["context"], question=row["question"])}]


def prepare_splits():
    from datasets import concatenate_datasets, load_dataset

    raw = load_dataset("kuznetsoffandrey/sberquad")
    # тексты из validation есть и в train, поэтому делим заново по абзацам
    by_ctx = {}
    for r in concatenate_datasets([raw["train"], raw["validation"]]):
        by_ctx.setdefault(r["context"], []).append({"id": r["id"], "context": r["context"], "question": r["question"], "answer": r["answers"]["text"][0]})
    contexts = sorted(by_ctx)
    random.Random(SEED).shuffle(contexts)
    splits = {"test": [], "dev": [], "train": []}
    for c in contexts:
        name = "test" if len(splits["test"]) < 5000 else "dev" if len(splits["dev"]) < 500 else "train"
        splits[name] += by_ctx[c]
    splits["train"] = random.Random(SEED).sample(splits["train"], 20000)
    DATA_DIR.mkdir(exist_ok=True)
    for name, rows in splits.items():
        with open(DATA_DIR / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (DATA_DIR / "split_ids.json").write_text(json.dumps({k: [r["id"] for r in v] for k, v in splits.items()}))
    return splits


def load_split(name):
    return [json.loads(l) for l in open(DATA_DIR / f"{name}.jsonl")]


def normalize_answer(s):
    s = re.sub(r"[^\w\s]", " ", s.lower().replace("ё", "е"))
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


if __name__ == "__main__":
    for name, rows in prepare_splits().items():
        print(name, len(rows), "вопросов", len({r["context"] for r in rows}), "абзацев")
