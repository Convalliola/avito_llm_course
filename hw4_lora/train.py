import gc
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from common import MODEL_NAME, OUTPUTS_DIR, SEED, load_split, prompt_messages
from eval_model import SMOKE, evaluate_model, save_result

TARGETS = {
    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qv": ["q_proj", "v_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
}
FULL = dict(method="full", lr=1e-5, seed=SEED)
LORA = dict(method="lora", lr=2e-4, r=16, alpha=32, dropout=0.05, targets="all", seed=SEED)
EXPERIMENTS = {
    "full_ft": FULL,
    "full_ft_lr2e-6": FULL | {"lr": 2e-6},
    "full_ft_lr5e-6": FULL | {"lr": 5e-6},
    "full_ft_lr2e-5": FULL | {"lr": 2e-5},
    "lora_base": LORA,
    "lora_base_seed1": LORA | {"seed": 1},
    "lora_base_seed2": LORA | {"seed": 2},
    **{f"lora_r{r}": LORA | {"r": r} for r in [4, 8, 32, 64, 128]},
    **{f"lora_alpha{a}": LORA | {"alpha": a} for a in [8, 16, 64, 128]},
    **{f"lora_dropout{d}": LORA | {"dropout": d} for d in [0.0, 0.1, 0.2]},
    **{f"lora_{t}": LORA | {"targets": t} for t in ["qv", "attn", "mlp"]},
    "lora_qv_r128": LORA | {"targets": "qv", "r": 128},
    "lora_r128_alpha256": LORA | {"r": 128, "alpha": 256},
    "lora_lr5e-5": LORA | {"lr": 5e-5},
    "lora_lr8e-4": LORA | {"lr": 8e-4},
    # только замер памяти с gradient checkpointing, 100 шагов без оценки
    "full_ft_gc": FULL | {"gradient_checkpointing": True, "max_steps": 100, "profile_only": True},
    "lora_base_gc": LORA | {"gradient_checkpointing": True, "max_steps": 100, "profile_only": True},
}


class AnswerLogitsTrainer(SFTTrainer):
    # логиты по словарю считаем только на токенах ответа, TRL по умолчанию считает их для всего батча
    train_tokens = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        y = inputs["labels"][0, 1:]
        pos = (y != -100).nonzero().squeeze(-1)
        out = model(input_ids=inputs["input_ids"], position_ids=inputs["position_ids"], logits_to_keep=pos, use_cache=False)
        loss = torch.nn.functional.cross_entropy(out.logits[0].float(), y[pos], reduction="sum")
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = loss / num_items_in_batch
        else:
            loss = loss / len(pos)
        if model.training:
            self.train_tokens += inputs["input_ids"].shape[1]
        return (loss, out) if return_outputs else loss


def main(name):
    c = EXPERIMENTS[name]
    set_seed(c["seed"])
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    # full FT учит fp32 копию весов, в чистом bf16 мелкие шаги Adam теряются при округлении
    dtype = torch.float32 if c["method"] == "full" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype, attn_implementation="flash_attention_2").cuda()
    if c["method"] == "lora":
        model = get_peft_model(model, LoraConfig(r=c["r"], lora_alpha=c["alpha"], lora_dropout=c["dropout"],
                                                 target_modules=TARGETS[c["targets"]], task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    to_ds = lambda rows: Dataset.from_list(
        [{"prompt": prompt_messages(r), "completion": [{"role": "assistant", "content": r["answer"]}]} for r in rows])
    args = SFTConfig(
        output_dir=str(OUTPUTS_DIR / name), num_train_epochs=1, max_steps=30 if SMOKE else c.get("max_steps", -1),
        per_device_train_batch_size=32, per_device_eval_batch_size=32, learning_rate=c["lr"],
        lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0,
        max_length=2048,  # по умолчанию 1024, длинные тексты обрезались бы вместе с ответом
        padding_free=True, bf16=True,
        gradient_checkpointing=c.get("gradient_checkpointing", False),  # в TRL он по умолчанию включен
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, eval_strategy="steps", eval_steps=10 if SMOKE else 100, eval_on_start=True,
        save_strategy="no", report_to="none", seed=c["seed"], dataset_num_proc=8,
    )
    trainer = AnswerLogitsTrainer(model=model, args=args, processing_class=tok,
                                  train_dataset=to_ds(load_split("train")), eval_dataset=to_ds(load_split("dev")))
    torch.cuda.reset_peak_memory_stats()
    res = trainer.train()
    hist = trainer.state.log_history
    # из времени обучения вычитаем замеры loss на dev
    eval_time = sum(h["eval_runtime"] for h in hist if "eval_runtime" in h)
    train_time = res.metrics["train_runtime"] - eval_time
    info = {
        "name": name, **c, "target_modules": TARGETS.get(c.get("targets")),
        "trainable_params": trainable, "total_params": total, "trainable_pct": 100 * trainable / total,
        "train_examples": len(trainer.train_dataset), "steps": trainer.state.global_step,
        "train_runtime": res.metrics["train_runtime"], "dev_eval_time": eval_time, "train_time": train_time,
        "train_tokens": trainer.train_tokens, "tokens_per_second": trainer.train_tokens / train_time,
        "peak_mem_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_mem_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "train_loss": res.metrics["train_loss"], "log_history": hist,
    }
    print(name, round(train_time), "с", round(info["tokens_per_second"]), "ток/с", round(info["peak_mem_allocated_gb"], 1), "ГБ")
    if c.get("profile_only"):
        save_result(name, info, {}, [])
        return
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    if c["method"] == "full":
        model.to(torch.bfloat16)
    if not SMOKE:
        model.save_pretrained(OUTPUTS_DIR / name)
        tok.save_pretrained(OUTPUTS_DIR / name)
    save_result(name, info, *evaluate_model(model, tok))


main(sys.argv[1])
