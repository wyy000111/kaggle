"""Kaggle T4 单卡 QLoRA 微调 7B：自检 -> 训 -> 存 adapter 到 /kaggle/working -> webhook。"""
import os, json, urllib.request
from datetime import datetime, timezone

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
OUT_DIR = "/kaggle/working/lora-adapter"

def notify(t):
    if not WEBHOOK_URL: return
    payload = json.dumps({"msgtype": "text", "text": {"content": t}}).encode()
    req = urllib.request.Request(WEBHOOK_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    try: urllib.request.urlopen(req, timeout=10)
    except Exception: pass

def main():
    started = datetime.now(timezone.utc)
    notify(f"[Kaggle] QLoRA 开始 {started:%Y-%m-%d %H:%M} UTC")
    assert torch.cuda.is_available(), "没分到 GPU，检查 Accelerator 设置"
    notify(f"[Kaggle] GPU={torch.cuda.get_device_name(0)}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # 1) NF4 4-bit 量化底座
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,   # T4 无 BF16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.float16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 2) LoRA：全线性层都挂（QLoRA 论文建议，掉点更少）
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()   # 约 0.4%~0.8%

    # 3) 数据：换成你自己的 jsonl（{"text": "..."} 或对话列）
    ds = load_dataset("json", data_files="/kaggle/input/你的数据集/train.jsonl")["train"]

    # 4) 训练参数：T4 上 batch=1 + 累积，省显存
    args = TrainingArguments(
        output_dir="/kaggle/working/ckpt",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",        # 省时间/省盘，只在结尾存一次
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         tokenizer=tok, max_seq_length=1024)
    trainer.train()

    # 5) 只存 adapter（几百 MB），别存合并后的整模
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    notify(f"[Kaggle] 完成，adapter 已存 {OUT_DIR}，用时 "
           f"{(datetime.now(timezone.utc)-started).total_seconds()/60:.1f} 分钟")

if __name__ == "__main__":
    main()
