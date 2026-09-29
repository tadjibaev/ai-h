"""Проверки правил анализа:  python3 pmi_analiz/test_analyze.py"""
import copy
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import analyze, card_template  # noqa: E402

TODAY = dt.date(2026, 9, 29)
BASE = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_project.json"), encoding="utf-8"))


def codes(p):
    return {r["code"] for r in analyze(copy.deepcopy(p), TODAY)["risks"]}


def test_example_is_not_low_risk():
    a = analyze(copy.deepcopy(BASE), TODAY)
    assert a["level"] != "Низкий", a["score"]
    assert {"stall", "schedule", "noplan", "passport", "nodoc"} <= {r["code"] for r in a["risks"]}
    assert "overdue" not in {r["code"] for r in a["risks"]}
    assert a["next_step"]["n"] == 3
    assert all(card_template(a).values())


def test_overdue_step():
    p = copy.deepcopy(BASE)
    p["steps"][2]["plan_date"] = "2026-06-01"
    a = analyze(p, TODAY)
    assert any(r["code"] == "overdue" and "120 дн." in r["text"] for r in a["risks"])


def test_done_with_future_date_is_flagged():
    p = copy.deepcopy(BASE)
    p["steps"][2].update(status="done", fact_date="2026-12-15")
    a = analyze(p, TODAY)
    assert "data" in {r["code"] for r in a["risks"]} and a["done"] == 2


def test_filled_healthy_project_is_low():
    p = copy.deepcopy(BASE)
    p.update(region="Навоийская область", cost_usd_mln=12, year_plan={"step": 4, "date": "2026-12-16"})
    p["steps"][1]["documents"] = [{"name": "Протокол.pdf", "date": "2026-06-30"}]
    p["steps"][1]["fact_date"] = p["steps"][1]["plan_date"] = "2026-09-15"
    p["steps"][3]["plan_date"], p["steps"][4]["plan_date"] = "2027-02-15", "2027-04-01"
    a = analyze(p, TODAY)
    assert a["level"] == "Низкий", a["risks"]


def test_silence():
    p = copy.deepcopy(BASE)
    p["messages"] = [{"date": "2026-06-01", "text": "…"}]
    assert "silence" in codes(p)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
