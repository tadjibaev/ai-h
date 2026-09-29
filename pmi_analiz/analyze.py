#!/usr/bin/env python3
"""
Анализ проекта по его шагам (этапам) для карточки ИИ на портале PMI.

Идея: выводы делаются из ДАННЫХ проекта — шагов с плановыми и фактическими датами,
подтверждающих документов, переносов сроков, переписки и паспорта проекта. Фото не
используются как доказательство выполнения (только их дата — как след события).

Два слоя:
  1. Правила (детерминированно): считают факты и баллы риска, каждый балл — с объяснением.
     Модель не может «придумать» просрочку или её отсутствие — цифры приходят отсюда.
  2. Тексты карточки: по шаблонам (по умолчанию) или LLM через OpenAI-совместимый API
     (свой vLLM с Qwen, см. docs/04-zapusk.md). LLM получает только посчитанные факты
     и тексты переписки и обязана ссылаться на шаги.

Примеры:
    python3 pmi_analiz/analyze.py pmi_analiz/example_project.json --today 2026-09-29
    python3 pmi_analiz/analyze.py projects/*.json --llm http://localhost:8000/v1 --model qwen3.8-27b

Формат входа — pmi_analiz/README.md. Зависимости: только стандартная библиотека
(httpx — лишь для режима --llm).
"""
import argparse
import datetime as dt
import json
import re
import sys

LEVELS = [(75, "Критический"), (50, "Высокий"), (25, "Средний"), (0, "Низкий")]
# шаги, которые физически не делаются за пару дней: если по плану на них отведено меньше
# MIN_DAYS после предыдущего шага — график нереалистичен
HEAVY = {r"смет|psd|псд|loyiha-smeta|smeta": 30, r"moliyalash|финансир": 30,
         r"qurilish|строит|монтаж|montaj": 60, r"uskuna|оборуд": 30, r"ekspertiza|эксперт": 20}
DONE, ACTIVE = "done", "in_progress"


def d(s):
    return dt.date.fromisoformat(s) if s else None


def fmt(x):
    return x.strftime("%d.%m.%Y") if x else "—"


def short(name, n=45):
    return name if len(name) <= n else name[: n - 1] + "…"


def analyze(p, today):
    steps = p.get("steps", [])
    for i, s in enumerate(steps):
        s["_i"], s["_plan"], s["_fact"] = i + 1, d(s.get("plan_date")), d(s.get("fact_date"))
    facts, risks = [], []  # risks: (баллы, код, текст)

    def risk(points, code, text):
        risks.append((points, code, text))

    done = [s for s in steps if s.get("status") == DONE]
    open_ = [s for s in steps if s.get("status") != DONE]
    nxt = min(open_, key=lambda s: s["_plan"] or dt.date.max) if open_ else None

    # 1. Просроченные шаги
    over = [s for s in open_ if s["_plan"] and s["_plan"] < today]
    for s in over:
        late = (today - s["_plan"]).days
        risk(min(20, 15 + 5 * (late > 60)), "overdue",
             f"Шаг {s['_i']} «{short(s['name'])}» просрочен на {late} дн. (план {fmt(s['_plan'])}).")

    # 2. Ошибки данных: «выполнено» с будущей датой, выполнено без даты, порядок шагов
    for s in done:
        if s["_fact"] and s["_fact"] > today:
            risk(10, "data", f"Шаг {s['_i']} «{short(s['name'])}» отмечен выполненным с датой в будущем "
                             f"({fmt(s['_fact'])}) — отметка недостоверна, шаг считается невыполненным.")
            s["status"] = "suspect"
        elif not s["_fact"]:
            risk(5, "data", f"Шаг {s['_i']} отмечен выполненным без фактической даты.")
    done = [s for s in steps if s.get("status") == DONE]
    for a, b in zip(steps, steps[1:]):
        if b.get("status") == DONE and a.get("status") != DONE:
            risk(5, "order", f"Шаг {b['_i']} выполнен раньше предыдущего шага {a['_i']} — нарушена последовательность.")

    # 3. Выполнено без подтверждающего документа (фото документом не считается)
    nodoc = [s for s in done if not s.get("documents")]
    if nodoc:
        risk(min(15, 5 * len(nodoc)), "nodoc", "Нет подтверждающего документа по выполненным шагам: "
             + ", ".join(f"{s['_i']}" for s in nodoc) + (" (есть только фото)." if any(s.get("photos") for s in nodoc) else "."))

    # 4. Сдвиги: выполнено с опозданием и переносы плановых дат
    slips = [(s, (s["_fact"] - s["_plan"]).days) for s in done if s["_fact"] and s["_plan"] and s["_fact"] > s["_plan"]]
    if slips:
        facts.append("Выполнено с опозданием: " + ", ".join(f"шаг {s['_i']} на {n} дн." for s, n in slips) + ".")
        risk(min(10, 3 * len(slips)), "slip", f"{len(slips)} шаг(ов) выполнено позже плана.")
    moves = sum(max(0, len(s.get("plan_history", [])) - 1) for s in steps)
    if moves:
        risk(min(20, 4 * moves), "reschedule", f"Сроки шагов переносились {moves} раз(а).")

    # 5. Застой: сколько дней нет продвижения по шагам (не по переписке!)
    last_fact = max((s["_fact"] for s in done if s["_fact"]), default=d(p.get("created_date")))
    stall = (today - last_fact).days if last_fact else None
    if stall is not None:
        facts.append(f"Последний выполненный шаг — {fmt(last_fact)}, без продвижения {stall} дн.")
        if stall > 120:
            risk(20, "stall", f"{stall} дн. нет ни одного выполненного шага.")
        elif stall > 60:
            risk(10, "stall", f"{stall} дн. нет ни одного выполненного шага.")

    # 6. Нереалистичный график: тяжёлый шаг запланирован почти сразу после предыдущего
    for a, b in zip(steps, steps[1:]):
        if a["_plan"] and b["_plan"] and b.get("status") != DONE:
            gap = (b["_plan"] - a["_plan"]).days
            need = next((n for pat, n in HEAVY.items() if re.search(pat, b["name"].lower())), 0)
            if need and gap < need:
                risk(10, "schedule", f"На шаг {b['_i']} «{short(b['name'])}» отведено {gap} дн. после шага {a['_i']} "
                                     f"(реалистично не меньше {need}) — график требует пересмотра.")

    # 7. Годовой план: будет ли результат в текущем году
    plan = p.get("year_plan")
    in_year = [s for s in open_ if s["_plan"] and s["_plan"].year == today.year]
    if not plan:
        risk(10, "noplan", f"План на {today.year} год не задан — нечем измерить выполнение.")
    if open_ and not in_year and not over:
        risk(10, "year", f"До конца {today.year} года ни один шаг не запланирован — результата в этом году не будет.")
    if nxt:
        facts.append(f"Следующий шаг {nxt['_i']} «{short(nxt['name'])}» — план {fmt(nxt['_plan'])}"
                     + (f", через {(nxt['_plan'] - today).days} дн." if nxt["_plan"] and nxt["_plan"] >= today else "."))

    # 8. Полнота паспорта — чем меньше данных, тем меньше доверия к «низкому риску»
    missing = [lbl for key, lbl in [("region", "область"), ("cost_usd_mln", "стоимость"),
                                    ("responsible", "ответственный")] if not p.get(key)]
    if missing:
        risk(min(15, 5 * len(missing)), "passport", "Не заполнено в паспорте: " + ", ".join(missing) + ".")
    completeness = round(100 * (1 - (len(missing) + (not plan)) / 4) * (1 - 0.5 * (not steps)))

    # 9. Переписка — дата последнего сообщения и тексты (смысл оценивает LLM, не счётчик)
    msgs = sorted(p.get("messages", []), key=lambda m: m["date"])
    if msgs:
        silence = (today - d(msgs[-1]["date"])).days
        facts.append(f"Переписка: {len(msgs)} сообщ., последнее {fmt(d(msgs[-1]['date']))} ({silence} дн. назад).")
        if silence > 60:
            risk(15, "silence", f"Переписки нет {silence} дн.")
        elif silence > 30:
            risk(10, "silence", f"Переписки нет {silence} дн.")
        if msgs and stall and stall > 60 and silence <= 14:
            facts.append("Переписка идёт, но шаги не закрываются — активность без результата.")
    else:
        risk(10, "silence", "Переписки по проекту нет.")

    score = min(100, sum(r[0] for r in risks))
    level = next(name for lim, name in LEVELS if score >= lim)
    return dict(id=p.get("id"), title=p.get("title"), today=today.isoformat(), score=score, level=level,
                completeness=completeness, missing=missing, done=len(done), total=len(steps),
                next_step=nxt and {"n": nxt["_i"], "name": nxt["name"], "plan": nxt.get("plan_date")},
                facts=facts, risks=[{"points": r[0], "code": r[1], "text": r[2]} for r in sorted(risks, key=lambda r: -r[0])],
                messages=[{"date": m["date"], "from": m.get("from", ""), "text": m.get("text", "")} for m in msgs[-10:]])


def card_template(a):
    """Тексты карточки без LLM — строго из посчитанных фактов."""
    top = [r["text"] for r in a["risks"][:3]]
    state = (f"Выполнено {a['done']} из {a['total']} шагов. " + " ".join(a["facts"][:2])).strip()
    problem = " ".join(top) if top else "По данным шагов проблем не выявлено."
    why = f"Оценка {a['score']}/100 ({a['level'].lower()}); полнота данных {a['completeness']}%."
    if a["completeness"] < 75:
        why += " Данных мало — оценка ненадёжна, сначала заполнить паспорт."
    recs = {"overdue": "назначить новые сроки по просроченным шагам с ответственным",
            "data": "исправить отметки выполнения — ставить «выполнено» только по факту",
            "nodoc": "загрузить подтверждающие документы по выполненным шагам",
            "schedule": "пересмотреть график: разнести сроки шагов реалистично",
            "noplan": "утвердить план на год (какой шаг и результат до 31.12)",
            "year": "определить, что будет сделано по проекту до конца года",
            "passport": "заполнить паспорт проекта: " + ", ".join(a["missing"]),
            "stall": "запросить у инициатора статус и дату следующего шага",
            "silence": "возобновить переписку с инициатором", "reschedule": "выяснить причины переносов сроков",
            "slip": "контролировать сроки следующих шагов", "order": "проверить последовательность шагов"}
    rec = "; ".join(dict.fromkeys(recs[r["code"]] for r in a["risks"][:3])) or "продолжать мониторинг"
    return {"Текущее состояние": state, "Вероятный риск": why, "Проблема": problem, "Рекомендация": rec[:1].upper() + rec[1:] + "."}


SYSTEM = """Ты аналитик портфеля инвестиционных проектов министерства. Пиши по-русски, коротко и по делу.
Правила:
- Используй ТОЛЬКО переданные факты, риски и тексты переписки. Ничего не выдумывай.
- Оценку риска и баллы НЕ меняй — они посчитаны по правилам.
- Каждое утверждение привязывай к шагу («шаг 3 …») или к сообщению с датой.
- Фото не доказывают выполнение шага — не делай по ним выводов.
- Если данных мало — прямо скажи, каких.
- Из переписки извлеки суть: о чём договорились, что обещано и к какому сроку, что блокирует.
Ответ — JSON с ключами: "Текущее состояние", "Вероятный риск", "Проблема", "Рекомендация" (каждое — 1–3 предложения)."""


def card_llm(a, base_url, model, api_key):
    import httpx
    body = {"model": model, "temperature": 0.2, "max_tokens": 700, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": json.dumps(a, ensure_ascii=False)}]}
    r = httpx.post(base_url.rstrip("/") + "/chat/completions", json=body, timeout=120,
                   headers={"Authorization": f"Bearer {api_key}"})
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    card = json.loads(text[text.find("{"): text.rfind("}") + 1])
    keys = ["Текущее состояние", "Вероятный риск", "Проблема", "Рекомендация"]
    if not all(isinstance(card.get(k), str) and card[k].strip() for k in keys):
        raise ValueError("LLM вернула неполную карточку")
    return {k: card[k] for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="JSON проекта (один проект или список проектов)")
    ap.add_argument("--today", default=dt.date.today().isoformat())
    ap.add_argument("--llm", help="base URL OpenAI-совместимого API, напр. http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--json", action="store_true", help="вывести полный разбор в JSON")
    args = ap.parse_args()
    today = d(args.today)
    projects = []
    for f in args.files:
        data = json.load(open(f, encoding="utf-8"))
        projects += data if isinstance(data, list) else [data]
    out = []
    for p in projects:
        a = analyze(p, today)
        try:
            a["card"] = card_llm(a, args.llm, args.model, args.api_key) if args.llm else card_template(a)
            a["card_source"] = "llm" if args.llm else "rules"
        except Exception as e:  # LLM недоступна или ответила мусором — карточка по правилам
            print(f"[{a['id']}] LLM: {e}; карточка по шаблону", file=sys.stderr)
            a["card"], a["card_source"] = card_template(a), "rules"
        out.append(a)
    out.sort(key=lambda a: -a["score"])
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    for a in out:
        print(f"#{a['id']}  {a['title']}\nРиск: {a['level']} {a['score']}/100 · шаги {a['done']}/{a['total']} · "
              f"полнота данных {a['completeness']}%")
        for r in a["risks"]:
            print(f"  +{r['points']:>2}  {r['text']}")
        for k, v in a["card"].items():
            print(f"{k}: {v}")
        print()


if __name__ == "__main__":
    main()
