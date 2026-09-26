"""
Общий «контракт» навыка исследования и анализа: как ставится задача, в каком формате
модель обязана отвечать и как этот ответ проверяется автоматически.

Один и тот же код используют:
  * eval_skills.py — экзамен модели и отбор хороших ответов учителя (данные для SFT);
  * sft_lora.py    — обучение на примерах;
  * grpo_train.py  — обучение с подкреплением (награды = проверки отсюда).

Формат задачи (одна строка JSONL):
{
  "id": "q-001",
  "skill": "grounded_qa | contradiction | numeric | abstain | plan | hypotheses",
  "sources": [{"id": "S1", "text": "..."}, ...],
  "question": "...",
  "answer": "эталонный короткий ответ (может отсутствовать для открытых задач)",
  "answer_type": "exact | number | abstain | free",
  "tolerance": 0.01,          # для number: допустимая относительная ошибка
  "must_cite": true           # требовать ли цитаты из источников
}
"""
import json
import re

SYSTEM_PROMPT = """Ты — исследователь-аналитик. Работаешь ТОЛЬКО с предоставленными источниками.
Правила:
1. Каждое утверждение подкрепляй точной цитатой из источника в формате [S1: "дословный фрагмент"].
2. Числа пересчитывай сам и показывай расчёт.
3. Если источники противоречат друг другу — назови противоречие явно.
4. Если данных недостаточно — так и напиши в ответе: «недостаточно данных». Не выдумывай.
5. Оцени уверенность честно: высокая / средняя / низкая.

Отвечай строго в формате:
## Вывод
<главный вывод в 1–3 предложениях>
## Обоснование
- <утверждение> [S1: "цитата"]
## Уверенность
<высокая | средняя | низкая>
## Ответ
<короткий итоговый ответ: слово, число или «недостаточно данных»>"""

SECTIONS = ("## Вывод", "## Обоснование", "## Уверенность", "## Ответ")
ABSTAIN = "недостаточно данных"
CITATION_RE = re.compile(r'\[(S\d+):\s*"([^"]+)"\]')


def build_messages(task: dict) -> list:
    """Промпт в формате чата: общий system + источники и вопрос."""
    sources = "\n\n".join(f"[{s['id']}]\n{s['text']}" for s in task["sources"])
    user = f"ИСТОЧНИКИ:\n\n{sources}\n\nЗАДАЧА: {task['question']}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def completion_text(completion) -> str:
    """Текст ответа без размышлений. Принимает строку или список сообщений (формат TRL)."""
    if isinstance(completion, list):
        completion = completion[-1].get("content") or ""
    if "</think>" in completion:  # размышления могли попасть в текст
        completion = completion.split("</think>", 1)[1]
    return completion.strip()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("ё", "е")).strip().lower()


def section(text: str, name: str) -> str:
    m = re.search(rf"{re.escape(name)}\s*\n(.*?)(?=\n## |\Z)", text, re.S)
    return m.group(1).strip() if m else ""


# ------------------------------------------------------------------ проверки (0..1)

def score_format(text: str) -> float:
    """Доля обязательных разделов, которые есть в ответе, в правильном порядке."""
    pos = [text.find(s) for s in SECTIONS]
    present = [p for p in pos if p >= 0]
    ordered = present == sorted(present)
    return (len(present) / len(SECTIONS)) * (1.0 if ordered else 0.5)


def score_citations(text: str, sources: list, must_cite: bool = True) -> float:
    """Доля цитат, которые ДОСЛОВНО есть в указанном источнике. Выдуманная цитата = ошибка."""
    by_id = {s["id"]: _norm(s["text"]) for s in sources}
    cites = CITATION_RE.findall(text)
    if not cites:
        return 0.0 if must_cite else 1.0
    ok = sum(1 for sid, quote in cites if sid in by_id and _norm(quote) in by_id[sid])
    return ok / len(cites)


def _first_number(s: str):
    m = re.search(r"-?\d[\d\s]*(?:[.,]\d+)?", s)
    if not m:
        return None
    return float(m.group(0).replace(" ", "").replace(",", "."))


def score_answer(text: str, task: dict):
    """1.0 — верно, 0.0 — неверно, None — у задачи нет эталона (открытый вопрос)."""
    kind, gold = task.get("answer_type", "free"), task.get("answer")
    answer = _norm(section(text, "## Ответ"))
    if kind == "free" or gold is None:
        return None
    if kind == "abstain":  # правильно — отказаться, если данных нет
        return 1.0 if ABSTAIN in answer else 0.0
    if ABSTAIN in answer:
        return 0.0
    if kind == "number":
        got, want = _first_number(answer), float(gold)
        if got is None:
            return 0.0
        tol = float(task.get("tolerance", 0.01))
        return 1.0 if abs(got - want) <= tol * max(1.0, abs(want)) else 0.0
    return 1.0 if _norm(str(gold)) in answer else 0.0


def score_calibration(text: str, correct) -> float:
    """Уверенность должна соответствовать правоте: «высокая» + ошибка — хуже всего."""
    if correct is None:
        return 0.5
    conf = _norm(section(text, "## Уверенность"))
    level = 1.0 if "высок" in conf else 0.5 if "средн" in conf else 0.0 if "низк" in conf else None
    if level is None:
        return 0.0
    return 1.0 - abs(level - correct)


def score_all(completion, task: dict) -> dict:
    text = completion_text(completion)
    ans = score_answer(text, task)
    parts = {
        "format": score_format(text),
        "citations": score_citations(text, task["sources"], task.get("must_cite", True)),
        "answer": ans,
        "calibration": score_calibration(text, ans),
    }
    # Итог: правильность важнее всего, выдуманные цитаты — второе по важности.
    w = {"answer": 0.5, "citations": 0.3, "format": 0.1, "calibration": 0.1}
    if ans is None:
        w = {"citations": 0.6, "format": 0.2, "calibration": 0.2}
    parts["total"] = sum(w[k] * parts[k] for k in w)
    return parts


def load_tasks(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
