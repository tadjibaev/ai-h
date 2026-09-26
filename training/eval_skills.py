#!/usr/bin/env python3
"""
Экзамен навыков исследования/анализа + сбор обучающих данных у «учителя».

Два режима одного скрипта:
  1. ЭКЗАМЕН: прогнать набор задач через модель (базовую или дообученную) и получить оценки
     по навыкам — правильность, настоящие ли цитаты, формат, калибровка уверенности.
  2. ДИСТИЛЛЯЦИЯ: прогнать те же задачи через сильную модель-учителя (любой OpenAI-совместимый
     API) по несколько раз и СОХРАНИТЬ только ответы, прошедшие проверки (--save-passing).
     Это и есть качественные данные для SFT — «отбор с отказом» (rejection sampling).

Примеры:
    # экзамен базовой модели на своём vLLM
    python3 training/eval_skills.py --tasks training/data/tasks_example.jsonl

    # сбор данных у учителя: 4 попытки на задачу, сохраняем ответы с оценкой >= 0.9
    python3 training/eval_skills.py --tasks my_tasks.jsonl --base-url https://openrouter.ai/api/v1 \\
        --api-key $KEY --model <сильная-модель> --samples 4 --save-passing data/sft_from_teacher.jsonl

Зависимость: httpx.
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
from collections import defaultdict

try:
    import httpx
except ImportError:
    sys.exit("Нужен пакет httpx:  pip install httpx")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skills_format import build_messages, load_tasks, score_all  # noqa: E402


async def ask(client, cfg, messages):
    body = {"model": cfg.model, "messages": messages, "max_tokens": cfg.max_tokens}
    if cfg.temperature is not None:
        body["temperature"] = cfg.temperature
    if cfg.thinking == "off":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif cfg.thinking in ("low", "medium", "xhigh"):
        body["chat_template_kwargs"] = {"reasoning_effort": cfg.thinking}
    for attempt in range(3):
        try:
            r = await client.post(f"{cfg.base_url}/chat/completions", json=body)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                await asyncio.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            return msg.get("content") or "", msg.get("reasoning") or msg.get("reasoning_content") or ""
        except httpx.TransportError:
            if attempt == 2:
                raise
            await asyncio.sleep(3 * (attempt + 1))
    raise RuntimeError("не удалось получить ответ")


async def main():
    p = argparse.ArgumentParser(description="Экзамен навыков анализа и сбор данных у учителя")
    p.add_argument("--tasks", required=True, help="JSONL с задачами (формат — skills_format.py)")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model", default=None, help="по умолчанию — первая модель из /v1/models")
    p.add_argument("--samples", type=int, default=1, help="попыток на задачу (для отбора данных — 4–8)")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--thinking", default="default", choices=["default", "off", "low", "medium", "xhigh"])
    p.add_argument("--save-passing", default=None, help="куда сохранить ответы, прошедшие проверку (JSONL для SFT)")
    p.add_argument("--min-score", type=float, default=0.9, help="порог итоговой оценки для сохранения")
    p.add_argument("--report", default=None, help="сохранить все ответы и оценки в JSON")
    cfg = p.parse_args()

    tasks = load_tasks(cfg.tasks)
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    async with httpx.AsyncClient(timeout=3600, headers=headers) as client:
        if not cfg.model:
            r = await client.get(f"{cfg.base_url}/models")
            r.raise_for_status()
            cfg.model = r.json()["data"][0]["id"]
        sem = asyncio.Semaphore(cfg.concurrency)

        async def run(task, k):
            async with sem:
                try:
                    content, reasoning = await ask(client, cfg, build_messages(task))
                except Exception as e:
                    return {"task": task, "k": k, "error": str(e)[:200]}
                return {"task": task, "k": k, "content": content, "reasoning": reasoning,
                        "scores": score_all(content, task)}

        results = await asyncio.gather(*(run(t, k) for t in tasks for k in range(cfg.samples)))

    ok = [r for r in results if "scores" in r]
    by_skill = defaultdict(list)
    for r in ok:
        by_skill[r["task"].get("skill", "?")].append(r["scores"])
    print(f"Модель: {cfg.model} | задач: {len(tasks)} × {cfg.samples} | ошибок запросов: {len(results) - len(ok)}\n")
    print(f"{'навык':<15} | {'n':>3} | {'итог':>5} | {'верно':>5} | {'цитаты':>6} | {'формат':>6} | {'калибр.':>7}")
    print("-" * 64)

    def avg(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return f"{statistics.mean(vals):.2f}" if vals else "  —"

    for skill, rows in sorted(by_skill.items()) + [("ВСЕГО", [s for r in ok for s in [r["scores"]]])]:
        print(f"{skill:<15} | {len(rows):>3} | {avg(rows, 'total'):>5} | {avg(rows, 'answer'):>5} | "
              f"{avg(rows, 'citations'):>6} | {avg(rows, 'format'):>6} | {avg(rows, 'calibration'):>7}")

    if cfg.save_passing:
        kept = [r for r in ok if r["scores"]["total"] >= cfg.min_score]
        with open(cfg.save_passing, "w", encoding="utf-8") as f:
            for r in kept:
                completion = {"role": "assistant", "content": r["content"]}
                if r["reasoning"]:
                    completion["reasoning_content"] = r["reasoning"]
                f.write(json.dumps({"prompt": build_messages(r["task"]), "completion": [completion]},
                                   ensure_ascii=False) + "\n")
        print(f"\nСохранено для SFT: {len(kept)} из {len(ok)} ответов (порог {cfg.min_score}) → {cfg.save_passing}")
    if cfg.report:
        with open(cfg.report, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
