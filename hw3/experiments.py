"""Короткие эксперименты с обучением для таблицы в блоке 4. Код берется из ячеек ноутбука, меняется только SFTConfig.

CUDA_VISIBLE_DEVICES=1 python experiments.py NAME LR MASTER(fp32|bf16) [EPOCHS], результат в exp_results/NAME.json
"""
import json
import sys
import time
from pathlib import Path

NAME, LR, MASTER = sys.argv[1], float(sys.argv[2]), sys.argv[3]
EPOCHS = float(sys.argv[4]) if len(sys.argv) > 4 else 3

nb = json.loads(Path("sft_function_calling_hw.ipynb").read_text())
sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def run(prefix):
    matches = [s for s in sources if s.lstrip().startswith(prefix)]
    assert len(matches) == 1, prefix
    exec(matches[0], globals())


run("# Окружение")
run("def generate(")
run("def read_jsonl(")
run("# YOUR CODE HERE\n# Меняется только ветка assistant")
run("def drop_nulls(")
run("TOOL_CALL_PATTERN")

if MASTER == "fp32":
    model.float()
config = SFTConfig(
    output_dir=f"exp_out/{NAME}",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=16,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_length=MAX_LENGTH,
    assistant_only_loss=True,
    bf16=True,
    gradient_checkpointing=False,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="no",
    report_to="none",
    seed=SEED,
)
trainer = SFTTrainer(model=model, args=config, train_dataset=train_dataset, eval_dataset=validation_dataset,
                     processing_class=tokenizer)
trainer.evaluate()
started = time.time()
result = trainer.train()
model.to(torch.bfloat16)
model.eval()
results = evaluate(model, tokenizer, read_jsonl("data/test_cases.jsonl"))
summary = summarize(results)
out = {
    "name": NAME, "lr": LR, "master": MASTER, "epochs": EPOCHS,
    "train_runtime": result.metrics["train_runtime"],
    "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
    "log_history": trainer.state.log_history,
    "test": summary.reset_index().to_dict("records"),
    "test_overall": {"tool_set_accuracy": results["tools_ok"].mean(), "exact_match": results["args_ok"].mean()},
    "failures": results[~results["args_ok"]][["id", "output"]].to_dict("records"),
}
Path("exp_results").mkdir(exist_ok=True)
Path(f"exp_results/{NAME}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
evals = [h["eval_loss"] for h in trainer.state.log_history if "eval_loss" in h]
print(NAME, f"eval_loss final {evals[-1]:.4f} min {min(evals):.4f}", out["test_overall"],
      f"{out['train_runtime']:.0f}s {out['peak_mem_gb']:.1f}GB")
