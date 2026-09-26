#!/usr/bin/env python3
"""
Шаг 1 обучения: SFT (обучение на примерах) с LoRA-адаптером для Qwen3.8-27B.

Модель учится ПОДРАЖАТЬ хорошим решениям: как разбирать задачу, цитировать источники,
считать, признавать нехватку данных. Меняются только маленькие LoRA-матрицы (~0.5–1% весов),
сама модель остаётся нетронутой — поэтому общие способности почти не страдают.

Данные: JSONL в формате prompt/completion (см. training/data/sft_example.jsonl) —
их делает eval_skills.py --save-passing из ответов модели-учителя.

Запуск (1× H200/B200 для BF16-LoRA; для 48–80 ГБ — флаг --qlora):
    pip install "trl==1.14.*" "peft==0.21.*" "transformers>=5.8" datasets accelerate \\
        flash-linear-attention causal-conv1d bitsandbytes
    python3 training/sft_lora.py --data data/sft_from_teacher.jsonl --general-data data/general.jsonl \\
        --out runs/analyst-sft

Результат — папка с адаптером. Подключение в vLLM:
    vllm serve Qwen/Qwen3.8-27B-FP8 ... --enable-lora --max-lora-rank 64 \\
        --lora-modules analyst=runs/analyst-sft
    (в запросах model="analyst")
"""
import argparse

import torch
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# Слои полного внимания и MLP. Слои DeltaNet (linear_attn.*) по умолчанию не трогаем:
# поддержка LoRA для них в vLLM пока частичная.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
GDN_TARGETS = ["in_proj_qkv", "in_proj_z", "out_proj"]


def main():
    p = argparse.ArgumentParser(description="SFT + LoRA для Qwen3.8-27B")
    p.add_argument("--model", default="Qwen/Qwen3.8-27B", help="базовая модель в BF16 (не FP8!)")
    p.add_argument("--data", nargs="+", required=True, help="JSONL с примерами навыка")
    p.add_argument("--general-data", nargs="*", default=[],
                   help="общие примеры (диалоги, код, задачи) — против «забывания»")
    p.add_argument("--general-share", type=float, default=0.25, help="доля общих примеров в смеси")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--max-length", type=int, default=16384, help="макс. длина примера в токенах")
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--qlora", action="store_true", help="4-битная база (влезает в 48–80 ГБ, медленнее)")
    p.add_argument("--lora-gdn", action="store_true", help="добавить LoRA и на слои DeltaNet")
    args = p.parse_args()

    skill = load_dataset("json", data_files=args.data, split="train")
    parts = [skill]
    if args.general_data:
        general = load_dataset("json", data_files=args.general_data, split="train").shuffle(seed=0)
        n = int(len(skill) * args.general_share / (1 - args.general_share))
        parts.append(general.select(range(min(n, len(general)))))
    data = concatenate_datasets(parts).shuffle(seed=0).train_test_split(test_size=0.05, seed=0)
    print(f"примеров: обучение {len(data['train'])}, проверка {len(data['test'])}")

    quant = None
    if args.qlora:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    # AutoModelForCausalLM грузит только языковую часть (без vision-энкодера).
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 quantization_config=quant, device_map="auto")

    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=LORA_TARGETS + (GDN_TARGETS if args.lora_gdn else []))
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,            # доля шагов на разогрев (float < 1 = доля)
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        max_length=args.max_length,   # длинные примеры обрезаются — следите, чтобы их было мало
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        report_to="none",
        # формат prompt/completion: ошибка считается только по ответу ассистента
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=data["train"],
                         eval_dataset=data["test"], peft_config=lora)
    trainer.train()
    trainer.save_model(args.out)
    print(f"Адаптер сохранён: {args.out}")


if __name__ == "__main__":
    main()
