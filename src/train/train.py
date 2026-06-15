# src/train/train.py
"""QLoRA fine-tune Qwen2.5-Coder-1.5B on grounded (prompt,target) jsonl. Run on Colab T4."""
from __future__ import annotations
import json, yaml, sys, os, glob
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

def load_records(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def main(cfg_path="configs/qlora.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    tok = AutoTokenizer.from_pretrained(cfg["base"])
    tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(cfg["base"], quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))

    def to_text(rec):
        msgs = [{"role": "user", "content": rec["prompt"]},
                {"role": "assistant", "content": rec["target"]}]
        return tok.apply_chat_template(msgs, tokenize=False)

    recs = load_records(cfg["train_file"])
    ds = Dataset.from_list([{"text": to_text(r)} for r in recs])
    ds = ds.map(lambda e: tok(e["text"], truncation=True, max_length=cfg["max_len"]),
                remove_columns=["text"])

    args = TrainingArguments(
        output_dir=cfg["output_dir"], num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"], gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"], fp16=True, logging_steps=10, save_steps=cfg["save_steps"],
        save_total_limit=2, gradient_checkpointing=True, report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    # Resume from the last checkpoint if one survived a previous (e.g. disconnected) run.
    has_ckpt = bool(glob.glob(os.path.join(cfg["output_dir"], "checkpoint-*")))
    if has_ckpt:
        print(f"resuming from checkpoint in {cfg['output_dir']}")
    trainer.train(resume_from_checkpoint=has_ckpt)
    model.save_pretrained(cfg["output_dir"])
    print(f"saved adapter -> {cfg['output_dir']}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/qlora.yaml")
