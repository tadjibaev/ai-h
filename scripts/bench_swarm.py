#!/usr/bin/env python3
"""
Нагрузочный тест «роя агентов» для OpenAI-совместимого сервера (vLLM / SGLang / llama.cpp).

Имитирует N агентов, которые работают ПАРАЛЛЕЛЬНО и в несколько ходов:
  * общий префикс (system + «общий документ») — одинаковый у всех агентов;
  * уникальный контекст у каждого агента;
  * на каждом ходу агент получает «результат инструмента» (новые токены) и отвечает.

Что меряем:
  * TTFT — время до первого токена (≈ время префилла + ожидание в очереди);
  * скорость декода на один поток (ток/с) и общую пропускную способность сервера;
  * сколько токенов промпта взято из кэша префиксов (если сервер это сообщает);
  * по /metrics (vLLM): попадания в кэш префиксов, приёмку MTP, вытеснения (preemptions).

Примеры:
    # быстрый прогон (проверить, что всё работает)
    python3 scripts/bench_swarm.py --agents 10

    # сценарий «10 агентов × ~220K контекста»
    python3 scripts/bench_swarm.py --agents 10 --shared-tokens 50000 \\
        --unique-tokens 150000 --turns 3 --turn-tokens 5000 --output-tokens 1000 --warmup

Зависимость: httpx (pip install httpx).
"""
import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time

try:
    import httpx
except ImportError:
    sys.exit("Нужен пакет httpx:  pip install httpx")

# Частые английские слова: у токенизатора Qwen почти каждое такое слово с пробелом = 1 токен.
VOCAB = """the of and to in is was for on that with as by at from his her this which or be are
an not have has had were but they their one all been would there more can if will about when
who so what up out time into only some them other than then its also two over after first new
most these made may such where many any like work year could used through between world city
system state water data model memory cache layer token agent report value result number point
area group level order power market process service support problem program government public
research analysis design control energy company history method project policy example question
""".split()


def make_text(n_tokens: int, rng: random.Random) -> str:
    """Синтетический текст длиной ~n_tokens (строки по 16 слов)."""
    words = [rng.choice(VOCAB) for _ in range(max(1, n_tokens))]
    return "\n".join(" ".join(words[i:i + 16]) for i in range(0, len(words), 16))


def pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def root_url(base_url: str) -> str:
    return re.sub(r"/v1/?$", "", base_url.rstrip("/"))


async def detect_model(client, base_url):
    r = await client.get(f"{base_url}/models")
    r.raise_for_status()
    data = r.json()["data"][0]
    return data["id"], data.get("max_model_len")


async def calibrate(client, cfg):
    """Сколько реальных токенов даёт make_text(1000) — через /tokenize (есть в vLLM)."""
    sample = make_text(1000, random.Random(0))
    try:
        r = await client.post(f"{root_url(cfg.base_url)}/tokenize",
                              json={"model": cfg.model, "prompt": sample}, timeout=30)
        r.raise_for_status()
        return r.json()["count"] / 1000.0
    except Exception:
        return None


async def scrape_metrics(client, cfg):
    """Счётчики Prometheus из /metrics (сумма по всем меткам). Пусто, если недоступно."""
    try:
        r = await client.get(f"{root_url(cfg.base_url)}/metrics", timeout=10)
        r.raise_for_status()
    except Exception:
        return {}
    out = {}
    for line in r.text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([-+0-9.eE]+|NaN|\+Inf)$", line)
        if m:
            try:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
            except ValueError:
                pass
    return out


def metric_delta(before, after, needle):
    names = [k for k in after if needle in k and not k.endswith(("_created", "_bucket"))]
    if not names:
        return None
    name = min(names, key=len)  # предпочитаем «…_total», а не «…_sum» и т.п.
    return after[name] - before.get(name, 0.0)


async def chat_stream(client, cfg, messages, max_tokens):
    body = {
        "model": cfg.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": cfg.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if cfg.ignore_eos:
        body["ignore_eos"] = True  # vLLM/SGLang: генерировать ровно max_tokens
    if cfg.thinking == "off":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif cfg.thinking in ("low", "medium", "high", "xhigh"):
        body["chat_template_kwargs"] = {"reasoning_effort": cfg.thinking}

    t0 = time.perf_counter()
    t_first = None
    text, usage = [], {}
    async with client.stream("POST", f"{cfg.base_url}/chat/completions", json=body) as r:
        if r.status_code != 200:
            err = (await r.aread()).decode(errors="replace")[:300]
            raise RuntimeError(f"HTTP {r.status_code}: {err}")
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if (piece or thought) and t_first is None:
                    t_first = time.perf_counter()
                text.append(piece)
    t_end = time.perf_counter()
    t_first = t_first or t_end
    completion = usage.get("completion_tokens") or 0
    details = usage.get("prompt_tokens_details") or {}
    decode_time = t_end - t_first
    return {
        "ttft": t_first - t0,
        "e2e": t_end - t0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "completion_tokens": completion,
        "decode_tps": (completion - 1) / decode_time if completion > 1 and decode_time > 0 else None,
        "text": "".join(text),
    }


async def run_agent(idx, client, cfg, shared_prefix, results, scale):
    rng = random.Random(cfg.seed * 1000 + idx)
    await asyncio.sleep(idx * cfg.stagger)
    messages = [
        {"role": "system", "content": shared_prefix},
        {"role": "user", "content": (
            f"You are agent #{idx}. Private context follows.\n"
            f"{make_text(int(cfg.unique_tokens / scale), rng)}\n\n"
            "Task: extract the key facts from the context above.")},
    ]
    for turn in range(cfg.turns):
        try:
            r = await chat_stream(client, cfg, messages, cfg.output_tokens)
        except Exception as e:  # ошибку фиксируем и прекращаем этого агента
            results.append({"agent": idx, "turn": turn, "error": str(e)[:300]})
            return
        r.update(agent=idx, turn=turn)
        results.append(r)
        print(f"  агент {idx:>2} ход {turn}: prompt={r['prompt_tokens']} "
              f"cached={r['cached_tokens']} TTFT={r['ttft']:.2f}s "
              f"декод={r['decode_tps'] or 0:.1f} ток/с", flush=True)
        messages.append({"role": "assistant", "content": r.pop("text") or "ok"})
        messages.append({"role": "user", "content": (
            f"Tool result:\n{make_text(int(cfg.turn_tokens / scale), rng)}\n\nContinue the task.")})


def report(results, wall, cfg, mdelta):
    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    print("\n" + "=" * 78)
    print(f"Агентов: {cfg.agents}, ходов: {cfg.turns}, успешных запросов: {len(ok)}, ошибок: {len(errors)}")
    for e in errors[:5]:
        print(f"  ошибка (агент {e['agent']}, ход {e['turn']}): {e['error']}")
    if not ok:
        return
    print(f"\n{'ход':>3} | {'prompt ток.':>11} | {'из кэша':>7} | {'TTFT p50':>8} | {'TTFT p90':>8} "
          f"| {'декод ток/с/поток':>17} | {'e2e p50':>7}")
    for turn in range(cfg.turns):
        rs = [r for r in ok if r["turn"] == turn]
        if not rs:
            continue
        prompt = statistics.mean(r["prompt_tokens"] or 0 for r in rs)
        cached = [r["cached_tokens"] for r in rs if r["cached_tokens"] is not None]
        cache_share = f"{100 * sum(cached) / max(1, sum(r['prompt_tokens'] or 0 for r in rs)):.0f}%" if cached else "н/д"
        dec = [r["decode_tps"] for r in rs if r["decode_tps"]]
        print(f"{turn:>3} | {prompt:>11,.0f} | {cache_share:>7} | {pct([r['ttft'] for r in rs], 50):>7.2f}s "
              f"| {pct([r['ttft'] for r in rs], 90):>7.2f}s | {statistics.mean(dec) if dec else 0:>17.1f} "
              f"| {pct([r['e2e'] for r in rs], 50):>6.1f}s")

    out_tok = sum(r["completion_tokens"] for r in ok)
    prompt_tok = sum(r["prompt_tokens"] or 0 for r in ok)
    cached_tok = sum(r["cached_tokens"] or 0 for r in ok)
    print(f"\nВремя теста: {wall:.1f} с")
    print(f"Сгенерировано: {out_tok:,} ток. → {out_tok / wall:,.0f} ток/с суммарно по серверу")
    print(f"Обработано промпта: {prompt_tok:,} ток., из них из кэша: {cached_tok:,} "
          f"→ реально посчитано {(prompt_tok - cached_tok) / wall:,.0f} ток/с префилла")

    if mdelta:
        print("\nМетрики сервера (/metrics, разница за время теста):")
        q, h = mdelta.get("prefix_cache_queries"), mdelta.get("prefix_cache_hits")
        if q:
            print(f"  кэш префиксов: {100 * (h or 0) / q:.1f}% токенов взято из кэша")
        acc, drf = mdelta.get("spec_decode_num_accepted_tokens"), mdelta.get("spec_decode_num_draft_tokens")
        if drf:
            print(f"  MTP/спекуляция: принято {100 * (acc or 0) / drf:.1f}% черновых токенов")
            n_drafts = mdelta.get("spec_decode_num_drafts")
            if n_drafts:
                print(f"  средняя длина принятия: {1 + (acc or 0) / n_drafts:.2f} ток. за шаг")
        if mdelta.get("num_preemptions"):
            print(f"  ВНИМАНИЕ: вытеснений (preemptions) = {mdelta['num_preemptions']:.0f} — "
                  "не хватает KV-кэша, уменьшите число агентов/контекст или добавьте памяти")


async def main():
    p = argparse.ArgumentParser(description="Нагрузочный тест роя агентов (OpenAI-совместимый API)")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model", default=None, help="по умолчанию — первая модель из /v1/models")
    p.add_argument("--agents", type=int, default=10, help="сколько агентов работают параллельно")
    p.add_argument("--shared-tokens", type=int, default=8000, help="общий префикс (одинаковый у всех)")
    p.add_argument("--unique-tokens", type=int, default=24000, help="личный контекст каждого агента")
    p.add_argument("--turns", type=int, default=3, help="ходов у каждого агента")
    p.add_argument("--turn-tokens", type=int, default=2000, help="новых токенов на каждом ходу")
    p.add_argument("--output-tokens", type=int, default=256, help="токенов ответа на ход")
    p.add_argument("--thinking", default="off", choices=["off", "on", "low", "medium", "high", "xhigh"],
                   help="режим размышлений Qwen (off — предсказуемое время)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false",
                   help="не заставлять генерировать ровно --output-tokens")
    p.add_argument("--warmup", action="store_true",
                   help="сначала один раз прогреть общий префикс, потом запускать агентов")
    p.add_argument("--stagger", type=float, default=0.0, help="задержка между стартами агентов, с")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--json-out", default=None, help="сохранить сырые результаты в JSON")
    cfg = p.parse_args()

    limits = httpx.Limits(max_connections=cfg.agents + 4, max_keepalive_connections=cfg.agents + 4)
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    async with httpx.AsyncClient(timeout=cfg.timeout, limits=limits, headers=headers) as client:
        max_len = None
        if not cfg.model:
            cfg.model, max_len = await detect_model(client, cfg.base_url)
        ratio = await calibrate(client, cfg)
        scale = ratio or 1.0
        final_ctx = cfg.shared_tokens + cfg.unique_tokens + cfg.turns * (cfg.turn_tokens + cfg.output_tokens)
        print(f"Сервер: {cfg.base_url}  модель: {cfg.model}  max_model_len: {max_len or '?'}")
        print(f"Калибровка токенизатора: {'%.3f ток./слово' % ratio if ratio else 'нет /tokenize, считаем 1 слово = 1 ток.'}")
        print(f"План: агентов={cfg.agents}, ходов={cfg.turns}, контекст к концу ≈ {final_ctx:,} ток./агента, "
              f"всего в KV-кэше до ≈ {cfg.agents * final_ctx - (cfg.agents - 1) * cfg.shared_tokens:,} ток.")
        if max_len and final_ctx > max_len:
            print(f"ВНИМАНИЕ: контекст агента ({final_ctx:,}) больше max_model_len ({max_len:,}) — будут ошибки 400")

        shared = ("You are a helpful analyst in a team of agents. Shared project knowledge follows.\n"
                  + make_text(int(cfg.shared_tokens / scale), random.Random(cfg.seed)))
        m_before = await scrape_metrics(client, cfg)
        if cfg.warmup:
            t = time.perf_counter()
            await chat_stream(client, cfg, [{"role": "system", "content": shared},
                                            {"role": "user", "content": "Reply with: ok"}], 1)
            print(f"Прогрев общего префикса: {time.perf_counter() - t:.1f} с")

        results = []
        t0 = time.perf_counter()
        await asyncio.gather(*(run_agent(i, client, cfg, shared, results, scale) for i in range(cfg.agents)))
        wall = time.perf_counter() - t0
        m_after = await scrape_metrics(client, cfg)

    mdelta = {}
    for key in ("prefix_cache_queries", "prefix_cache_hits", "spec_decode_num_accepted_tokens",
                "spec_decode_num_draft_tokens", "spec_decode_num_drafts", "num_preemptions"):
        d = metric_delta(m_before, m_after, key)
        if d is not None:
            mdelta[key] = d
    report(results, wall, cfg, mdelta)
    if cfg.json_out:
        with open(cfg.json_out, "w") as f:
            json.dump({"config": vars(cfg), "wall_s": wall, "metrics_delta": mdelta, "requests": results},
                      f, indent=2, ensure_ascii=False)
        print(f"\nСырые результаты: {cfg.json_out}")


if __name__ == "__main__":
    asyncio.run(main())
