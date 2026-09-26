#!/usr/bin/env python3
"""
Шаг 2 обучения: GRPO (обучение с подкреплением на проверяемых наградах).

SFT учит подражать. GRPO учит ДОБИВАТЬСЯ результата: на каждую задачу модель генерирует
группу из N ответов, автоматические проверки из skills_format.py их оценивают
(верный ответ, настоящие цитаты, формат, честная уверенность), и модель сдвигается
к ответам лучше среднего по группе. Эталонные решения не нужны — нужен эталонный ОТВЕТ
и возможность его проверить. Так выращивают навыки, которым трудно научить примерами:
не выдумывать цитаты, пересчитывать числа, отказываться при нехватке данных.

Генерация идёт через vLLM. Для 27B удобнее режим server: vLLM на отдельной GPU.
    # терминал 1 (GPU 0): генерация (обёртка TRL над `vllm serve` с флагами для синхронизации весов;
    # в TRL 1.14 помечена устаревшей, при запуске печатает эквивалентную команду vllm serve)
    CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3.8-27B
    # терминал 2 (GPU 1): обучение
    CUDA_VISIBLE_DEVICES=1 python3 training/grpo_train.py --tasks data/rl_tasks.jsonl \\
        --init-adapter runs/analyst-sft --out runs/analyst-grpo

Сначала отладьте всё на маленькой модели (--model Qwen/Qwen3.5-4B или похожей):
в десятки раз дешевле и быстрее, ошибки те же.
"""
import argparse
import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skills_format import build_messages, load_tasks, score_all  # noqa: E402


def reward_total(completions, task, **kwargs):
    """Итоговая оценка 0..1 за каждый ответ (см. score_all)."""
    return [score_all(c, json.loads(t))["total"] for c, t in zip(completions, task)]


def reward_no_fake_citations(completions, task, **kwargs):
    """Отдельный сильный штраф за выдуманные цитаты — главный порок «аналитиков»."""
    out = []
    for c, t in zip(completions, task):
        s = score_all(c, json.loads(t))["citations"]
        out.append(-1.0 if s < 0.5 else 0.0)
    return out


def main():
    p = argparse.ArgumentParser(description="GRPO для навыков анализа")
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--tasks", required=True, help="JSONL задач С ЭТАЛОННЫМИ ответами (answer_type != free)")
    p.add_argument("--init-adapter", default=None, help="продолжить с SFT-адаптера (рекомендуется)")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--generations", type=int, default=8, help="ответов на задачу в группе")
    p.add_argument("--max-completion", type=int, default=4096)
    p.add_argument("--vllm-mode", default="server", choices=["server", "colocate"])
    args = p.parse_args()

    tasks = [t for t in load_tasks(args.tasks) if t.get("answer_type", "free") != "free"]
    data = Dataset.from_list([{"prompt": build_messages(t), "task": json.dumps(t, ensure_ascii=False)}
                              for t in tasks])
    print(f"задач для RL: {len(data)}")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    peft_config = None
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(r=32, lora_alpha=64, task_type="CAUSAL_LM",
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                 "gate_proj", "up_proj", "down_proj"])

    cfg = GRPOConfig(
        output_dir=args.out,
        max_steps=args.steps,
        learning_rate=args.lr,
        num_generations=args.generations,
        per_device_train_batch_size=args.generations,   # одна задача (группа) на шаг микробатча
        gradient_accumulation_steps=4,                  # → 4 задачи на шаг оптимизатора
        max_completion_length=args.max_completion,
        temperature=1.0,                                # разнообразие внутри группы обязательно
        chat_template_kwargs={"reasoning_effort": "low"},  # короче размышления → дешевле генерация
        mask_truncated_completions=True,                # обрезанные ответы не учат плохому
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=1,
        log_completions=True,
        save_steps=50,
        report_to="none",
    )
    trainer = GRPOTrainer(model=model, reward_funcs=[reward_total, reward_no_fake_citations],
                          args=cfg, train_dataset=data, peft_config=peft_config)
    trainer.train()
    trainer.save_model(args.out)
    print(f"Адаптер сохранён: {args.out}")


if __name__ == "__main__":
    main()
