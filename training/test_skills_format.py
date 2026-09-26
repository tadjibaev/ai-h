"""Проверки автоматических оценок: python3 training/test_skills_format.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skills_format import load_tasks, score_all  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
tasks = {t["id"]: t for t in load_tasks(os.path.join(HERE, "data", "tasks_example.jsonl"))}
T = tasks["t-numeric-1"]  # выручка 2024 = 400


def answer(body, conf="высокая", final="400"):
    return f"## Вывод\nтекст\n## Обоснование\n{body}\n## Уверенность\n{conf}\n## Ответ\n{final}"


good = answer('- расчёт [S1: "Выручка составила 480 млн сум"]')
s = score_all(good, T)
assert s["answer"] == 1 and s["citations"] == 1 and s["format"] == 1 and s["total"] > 0.95, s

fake = answer('- [S1: "Выручка составила 500 млн сум"]')            # выдуманная цитата
assert score_all(fake, T)["citations"] == 0

wrong = answer('- [S1: "Выручка составила 480 млн сум"]', final="576")  # 480*1.2 — типичная ошибка
s = score_all(wrong, T)
assert s["answer"] == 0 and s["calibration"] == 0, s                  # уверен и неправ — худшее

humble = answer('- [S1: "Выручка составила 480 млн сум"]', conf="низкая", final="576")
assert score_all(humble, T)["calibration"] == 1                       # неправ, но честно сомневался

thought = "<think>размышления</think>\n\n" + good                    # размышления отрезаются
assert score_all(thought, T)["total"] == score_all(good, T)["total"]

msgs = [{"role": "assistant", "content": good}]                       # формат TRL (список сообщений)
assert score_all(msgs, T)["answer"] == 1

abstain = tasks["t-abstain-1"]
assert score_all(answer("- нет данных", final="недостаточно данных"), abstain)["answer"] == 1
assert score_all(answer("- выдумка", final="12%"), abstain)["answer"] == 0

no_format = "Выручка была 400."
assert score_all(no_format, T)["format"] == 0

print("все проверки пройдены")
