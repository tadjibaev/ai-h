#!/usr/bin/env python3
"""
Минимальный «рой агентов»: планировщик → N параллельных исполнителей → сборщик.
Работает с любым OpenAI-совместимым сервером (vLLM, SGLang, llama.cpp, облачный API).

Режимы:
  shared — ВСЕ агенты видят один и тот же большой контекст (файлы проекта, документы),
           но решают разные подзадачи. Контекст стоит в самом начале промпта и одинаков
           до байта → движок считает его ОДИН раз, остальные агенты берут его из кэша
           префиксов. Это главный приём экономии при больших контекстах.
  map    — файлы делятся между агентами (у каждого своя часть), потом сборщик объединяет
           выводы. Нужен, когда материалов больше, чем влезает в один контекст (262K).

Примеры:
    python3 scripts/swarm_demo.py --mode shared --context ./my_project \\
        --task "Найди главные риски и ошибки в проекте" --agents 10
    python3 scripts/swarm_demo.py --mode map --context ./docs --task "Сделай сводку" --agents 10

Зависимость: httpx (pip install httpx).
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("Нужен пакет httpx:  pip install httpx")

TEXT_EXT = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt", ".c", ".h", ".cpp",
            ".hpp", ".cs", ".php", ".rb", ".swift", ".sql", ".sh", ".yaml", ".yml", ".toml", ".json",
            ".ini", ".cfg", ".md", ".txt", ".rst", ".html", ".css", ".csv", ".xml"}
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".idea", ".mypy_cache"}

SYSTEM_SHARED = ("Ты — участник команды аналитиков-агентов. Ниже — общий контекст (материалы), "
                 "одинаковый для всей команды. Опирайся только на него, ссылайся на файлы и строки. "
                 "Отвечай по-русски, по делу.")


def load_files(paths, max_file_bytes):
    files = []
    for raw in paths:
        p = Path(raw)
        candidates = [p] if p.is_file() else sorted(
            f for f in p.rglob("*") if f.is_file() and not (set(f.parts) & SKIP_DIRS))
        for f in candidates:
            if f.suffix.lower() not in TEXT_EXT or f.stat().st_size > max_file_bytes:
                continue
            try:
                files.append((str(f), f.read_text(encoding="utf-8")))
            except (UnicodeDecodeError, OSError):
                continue
    # Сортировка = детерминированный порядок = одинаковый префикс между запусками (кэш!).
    return sorted(files)


def render_files(files):
    return "\n\n".join(f"### Файл: {path}\n```\n{text}\n```" for path, text in files)


def split_balanced(files, n):
    """Раскладываем файлы по n группам примерно равного размера (жадно, от больших к малым)."""
    groups = [[] for _ in range(n)]
    sizes = [0] * n
    for path, text in sorted(files, key=lambda f: -len(f[1])):
        i = sizes.index(min(sizes))
        groups[i].append((path, text))
        sizes[i] += len(text)
    return [sorted(g) for g in groups if g]


class LLM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = httpx.AsyncClient(
            timeout=cfg.timeout, headers={"Authorization": f"Bearer {cfg.api_key}"},
            limits=httpx.Limits(max_connections=cfg.agents + 4))
        self.stats = {"requests": 0, "prompt": 0, "cached": 0, "completion": 0}
        self.max_model_len = None

    async def init(self):
        r = await self.client.get(f"{self.cfg.base_url}/models")
        r.raise_for_status()
        data = r.json()["data"][0]
        self.cfg.model = self.cfg.model or data["id"]
        self.max_model_len = data.get("max_model_len")

    async def count_tokens(self, text):
        """Точный подсчёт через /tokenize (vLLM); иначе грубая оценка ~3.5 символа на токен."""
        root = re.sub(r"/v1/?$", "", self.cfg.base_url.rstrip("/"))
        try:
            r = await self.client.post(f"{root}/tokenize", json={"model": self.cfg.model, "prompt": text})
            r.raise_for_status()
            return r.json()["count"]
        except Exception:
            return int(len(text) / 3.5)

    async def chat(self, messages, max_tokens, retries=3):
        body = {"model": self.cfg.model, "messages": messages, "max_tokens": max_tokens}
        if self.cfg.temperature is not None:
            body["temperature"] = self.cfg.temperature
        if self.cfg.thinking == "off":
            body["chat_template_kwargs"] = {"enable_thinking": False}
        elif self.cfg.thinking in ("low", "medium", "xhigh"):
            body["chat_template_kwargs"] = {"reasoning_effort": self.cfg.thinking}  # только Qwen3.8
        for attempt in range(retries):
            try:
                r = await self.client.post(f"{self.cfg.base_url}/chat/completions", json=body)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt * 2)
                    continue
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
                data = r.json()
                break
            except httpx.TransportError:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt * 2)
        usage = data.get("usage") or {}
        self.stats["requests"] += 1
        self.stats["prompt"] += usage.get("prompt_tokens") or 0
        self.stats["cached"] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        self.stats["completion"] += usage.get("completion_tokens") or 0
        msg = data["choices"][0]["message"]
        return (msg.get("content") or "").strip()


def parse_subtasks(text, n):
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            items = [str(x).strip() for x in json.loads(m.group(0)) if str(x).strip()]
            if items:
                return items[:n]
        except json.JSONDecodeError:
            pass
    lines = [re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", ln).strip() for ln in text.splitlines()]
    return [ln for ln in lines if len(ln) > 10][:n]


async def run_shared(llm, cfg, files, log):
    context = render_files(files)
    shared = [{"role": "system", "content": f"{SYSTEM_SHARED}\n\n# ОБЩИЙ КОНТЕКСТ\n\n{context}"}]
    n_tok = await llm.count_tokens(shared[0]["content"])
    log(f"Общий контекст: файлов — {len(files)}, ≈{n_tok:,} ток.")
    if llm.max_model_len and n_tok + cfg.max_tokens > llm.max_model_len:
        sys.exit(f"Контекст ({n_tok:,}) + ответ ({cfg.max_tokens:,}) > max_model_len ({llm.max_model_len:,}). "
                 "Используйте --mode map или уменьшите набор файлов.")

    # 1) Прогрев: один раз считаем общий префикс, дальше все берут его из кэша.
    t = time.perf_counter()
    await llm.chat(shared + [{"role": "user", "content": "Ответь одним словом: готов"}], 1)
    log(f"[1/4] прогрев кэша префиксов: {time.perf_counter() - t:.1f} с")

    # 2) Планировщик делит задачу на подзадачи (тоже поверх общего префикса — почти бесплатно).
    if cfg.subtasks:
        subtasks = [s.strip() for s in Path(cfg.subtasks).read_text(encoding="utf-8").splitlines() if s.strip()]
    else:
        t = time.perf_counter()
        plan = await llm.chat(shared + [{"role": "user", "content": (
            f"Общая задача: {cfg.task}\n\nРаздели её ровно на {cfg.agents} независимых подзадач для "
            "параллельной работы агентов (без пересечений). Ответь ТОЛЬКО JSON-массивом строк.")}],
            cfg.plan_tokens)
        subtasks = parse_subtasks(plan, cfg.agents)
        log(f"[2/4] план готов (подзадач: {len(subtasks)}) за {time.perf_counter() - t:.1f} с")
    if not subtasks:
        sys.exit("Планировщик не вернул подзадачи — задайте их файлом через --subtasks")

    # 3) Исполнители работают параллельно.
    answers = await run_workers(llm, cfg, [
        shared + [{"role": "user", "content": (
            f"Общая задача команды: {cfg.task}\n\nТВОЯ подзадача (#{i + 1}): {sub}\n\n"
            "Сделай только свою подзадачу. Дай конкретные выводы со ссылками на файлы.")}]
        for i, sub in enumerate(subtasks)], subtasks, log)

    # 4) Сборщик тоже видит общий контекст (из кэша) и может перепроверить выводы.
    return await reduce(llm, cfg, shared, subtasks, answers, log), subtasks, answers


async def run_map(llm, cfg, files, log):
    groups = split_balanced(files, cfg.agents)
    log(f"Режим map: файлов — {len(files)}, групп (по одной на агента) — {len(groups)}")
    subtasks = [f"часть {i + 1}: " + ", ".join(p for p, _ in g[:5]) + (" …" if len(g) > 5 else "")
                for i, g in enumerate(groups)]
    system = {"role": "system", "content": SYSTEM_SHARED}  # одинаковый короткий префикс
    answers = await run_workers(llm, cfg, [
        [system, {"role": "user", "content": (
            f"Общая задача команды: {cfg.task}\n\nТвоя часть материалов:\n\n{render_files(g)}\n\n"
            "Выполни задачу только по своей части. Дай конкретные выводы со ссылками на файлы.")}]
        for g in groups], subtasks, log)
    return await reduce(llm, cfg, [system], subtasks, answers, log), subtasks, answers


async def run_workers(llm, cfg, prompts, labels, log):
    sem = asyncio.Semaphore(cfg.agents)
    t0 = time.perf_counter()

    async def one(i, messages):
        async with sem:
            t = time.perf_counter()
            try:
                ans = await llm.chat(messages, cfg.max_tokens)
            except Exception as e:
                ans = f"(ошибка агента: {e})"
            log(f"   агент {i + 1:>2} готов за {time.perf_counter() - t:5.1f} с — {labels[i][:70]}")
            return ans

    answers = await asyncio.gather(*(one(i, m) for i, m in enumerate(prompts)))
    log(f"[3/4] исполнители (агентов: {len(prompts)}) отработали за {time.perf_counter() - t0:.1f} с")
    return answers


async def reduce(llm, cfg, prefix, subtasks, answers, log):
    t = time.perf_counter()
    notes = "\n\n".join(f"## Агент {i + 1}: {s}\n{a}" for i, (s, a) in enumerate(zip(subtasks, answers)))
    report = await llm.chat(prefix + [{"role": "user", "content": (
        f"Общая задача: {cfg.task}\n\nНиже — выводы агентов команды.\n\n{notes}\n\n"
        "Собери из них единый итоговый отчёт: убери повторы, отметь противоречия, "
        "расставь приоритеты, в конце — список конкретных действий.")}], cfg.max_tokens)
    log(f"[4/4] сборка отчёта: {time.perf_counter() - t:.1f} с")
    return report


async def main():
    p = argparse.ArgumentParser(description="Рой агентов: планировщик → N исполнителей → сборщик")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model", default=None, help="по умолчанию — первая модель из /v1/models")
    p.add_argument("--mode", choices=["shared", "map"], default="shared")
    p.add_argument("--context", nargs="+", required=True, help="файлы и/или папки с материалами")
    p.add_argument("--task", required=True, help="общая задача для роя")
    p.add_argument("--subtasks", default=None, help="файл с подзадачами (по одной в строке) вместо планировщика")
    p.add_argument("--agents", type=int, default=10, help="сколько агентов работают одновременно")
    p.add_argument("--max-tokens", type=int, default=8192, help="лимит ответа агента (с размышлениями нужно больше)")
    p.add_argument("--plan-tokens", type=int, default=4096)
    p.add_argument("--thinking", default="default", choices=["default", "off", "low", "medium", "xhigh"],
                   help="размышления: off — быстрее всего; low/medium/xhigh — reasoning_effort Qwen3.8 "
                        "(одинаковый для всех агентов, иначе общий префикс не переиспользуется)")
    p.add_argument("--temperature", type=float, default=None, help="по умолчанию — из generation_config модели")
    p.add_argument("--max-file-kb", type=int, default=512, help="пропускать файлы крупнее")
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--out", default="swarm_report.md")
    cfg = p.parse_args()

    files = load_files(cfg.context, cfg.max_file_kb * 1024)
    if not files:
        sys.exit("Не нашёл текстовых файлов в --context")

    t0 = time.perf_counter()

    def log(msg):
        print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)

    llm = LLM(cfg)
    try:
        await llm.init()
        log(f"Сервер {cfg.base_url}, модель {cfg.model}, max_model_len={llm.max_model_len}")
        runner = run_shared if cfg.mode == "shared" else run_map
        report, subtasks, answers = await runner(llm, cfg, files, log)
    finally:
        await llm.client.aclose()

    appendix = "\n\n".join(f"### Агент {i + 1}: {s}\n\n{a}" for i, (s, a) in enumerate(zip(subtasks, answers)))
    Path(cfg.out).write_text(f"# Итоговый отчёт роя\n\nЗадача: {cfg.task}\n\n{report}\n\n---\n\n"
                             f"## Приложение: ответы агентов\n\n{appendix}\n", encoding="utf-8")
    s = llm.stats
    share = 100 * s["cached"] / s["prompt"] if s["prompt"] else 0
    log(f"Готово → {cfg.out}")
    print(f"Запросов: {s['requests']}, токенов промпта: {s['prompt']:,} (из кэша: {s['cached']:,} = {share:.0f}%), "
          f"сгенерировано: {s['completion']:,}")


if __name__ == "__main__":
    asyncio.run(main())
