import json
import os
import random
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODEL_NAME, RESULTS_DIR, SEED, exact_match, f1_score, load_split, prompt_messages

# SMOKE=1 быстрый прогон на маленьких выборках, чтобы проверить код
SMOKE = os.environ.get("SMOKE") == "1"
MMLU_PROMPT = "The following is a multiple choice question about {subject}.\n\n{question}\n{choices}\n\nAnswer with the letter of the correct option."
GENERAL_PROMPTS = [
    "Объясни простыми словами, что такое фотосинтез.",
    "Напиши короткое поздравление коллеге с днем рождения.",
    "У Маши было 5 яблок, она отдала 2 брату и купила еще 4. Сколько яблок у Маши? Объясни решение.",
    "Give three tips for learning a new language.",
]


@torch.inference_mode()
def generate_batch(model, tok, convs, max_new_tokens):
    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
    inputs = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
    # в generation_config модели включен sampling, выключаем
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
    n = inputs["input_ids"].shape[1]
    return [tok.decode(o[n:], skip_special_tokens=True).strip() for o in out]


def sberquad_predictions(model, tok, rows, bs=128):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["context"]) + len(rows[i]["question"]), reverse=True)
    preds = [None] * len(rows)
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        for i, o in zip(idx, generate_batch(model, tok, [prompt_messages(rows[i]) for i in idx], 64)):
            preds[i] = o
    return preds


@torch.inference_mode()
def answer_nll(model, tok, rows, bs=16):
    # loss на токенах правильного ответа как при обучении
    ex = []
    for r in rows:
        prompt = tok.apply_chat_template(prompt_messages(r), add_generation_prompt=True)
        full = tok.apply_chat_template(prompt_messages(r) + [{"role": "assistant", "content": r["answer"]}])
        assert full[:len(prompt)] == prompt
        ex.append((full, len(prompt)))
    ex.sort(key=lambda e: len(e[0]))
    total, count = 0.0, 0
    for k in range(0, len(ex), bs):
        batch = ex[k:k + bs]
        w = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), w), tok.pad_token_id)
        attn = torch.zeros((len(batch), w), dtype=torch.long)
        labels = torch.full((len(batch), w), -100)
        for i, (ids, start) in enumerate(batch):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            attn[i, :len(ids)] = 1
            labels[i, start:len(ids)] = torch.tensor(ids[start:])
        logits = model(input_ids=input_ids.cuda(), attention_mask=attn.cuda()).logits[:, :-1]
        y = labels[:, 1:].cuda()
        m = y != -100
        total += torch.nn.functional.cross_entropy(logits[m].float(), y[m], reduction="sum").item()
        count += m.sum().item()
    return total / count


@torch.inference_mode()
def mmlu_accuracy(model, tok, bs=128):
    data = load_dataset("cais/mmlu", "all", split="test")
    if SMOKE:
        data = data.shuffle(seed=SEED).select(range(500))
    letters = [tok.convert_tokens_to_ids(l) for l in "ABCD"]
    prompts = []
    for r in data:
        choices = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", r["choices"]))
        text = MMLU_PROMPT.format(subject=r["subject"].replace("_", " "), question=r["question"].strip(), choices=choices)
        prompts.append(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True))
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    answers = data["answer"]
    correct = [None] * len(prompts)
    # ответ это буква с самым большим логитом, без генерации
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        inputs = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, padding_side="left").to(model.device)
        pred = model(**inputs, logits_to_keep=1).logits[:, -1, letters].argmax(-1).tolist()
        for i, p in zip(idx, pred):
            correct[i] = p == answers[i]
    by_subject = {}
    for s, ok in zip(data["subject"], correct):
        by_subject.setdefault(s, []).append(ok)
    return sum(correct) / len(correct), {s: sum(v) / len(v) for s, v in sorted(by_subject.items())}


def manual_examples(model, tok):
    rows = random.Random(SEED).sample(load_split("test"), 6)
    qa = generate_batch(model, tok, [prompt_messages(r) for r in rows], 64)
    general = generate_batch(model, tok, [[{"role": "user", "content": p}] for p in GENERAL_PROMPTS], 256)
    return ([{"kind": "sberquad", "id": r["id"], "question": r["question"], "answer": r["answer"], "output": o}
             for r, o in zip(rows, qa)]
            + [{"kind": "general", "prompt": p, "output": o} for p, o in zip(GENERAL_PROMPTS, general)])


def evaluate_model(model, tok):
    model.eval()
    test = load_split("test")
    if SMOKE:
        test = test[:200]
    t = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        t0 = time.time()
        preds = sberquad_predictions(model, tok, test)
        t["sberquad_generate"] = time.time() - t0
        t0 = time.time()
        nll = answer_nll(model, tok, test)
        t["sberquad_nll"] = time.time() - t0
        t0 = time.time()
        mmlu, mmlu_by_subject = mmlu_accuracy(model, tok)
        t["mmlu"] = time.time() - t0
        manual = manual_examples(model, tok)
    m = {
        "sberquad_em": sum(exact_match(p, r["answer"]) for p, r in zip(preds, test)) / len(test),
        "sberquad_f1": sum(f1_score(p, r["answer"]) for p, r in zip(preds, test)) / len(test),
        "sberquad_nll": nll, "mmlu_acc": mmlu, "mmlu_by_subject": mmlu_by_subject, "eval_timings": t, "manual": manual,
    }
    print("EM", m["sberquad_em"], "F1", m["sberquad_f1"], "NLL", nll, "MMLU", mmlu, t)
    return m, [{"id": r["id"], "prediction": p} for r, p in zip(test, preds)]


def save_result(name, info, metrics, preds):
    out = RESULTS_DIR / "smoke" if SMOKE else RESULTS_DIR
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(info | metrics, ensure_ascii=False, indent=1))
    if preds:
        with open(out / "predictions" / f"{name}.jsonl", "w") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # оценка исходной модели
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
    info = {"name": "base", "method": "base", "trainable_params": 0, "total_params": sum(p.numel() for p in model.parameters())}
    save_result("base", info, *evaluate_model(model, tok))
