from arp.evals.checks import run_check, score


def test_contains_respects_token_boundaries():
    assert run_check({"type": "contains", "value": "3.1"}, "needs >=3.1")[0]
    assert not run_check({"type": "contains", "value": "3.1"}, "needs >=3.12")[0]
    assert not run_check({"type": "contains", "value": "mit"}, "please submit it")[0]


def test_number():
    assert run_check({"type": "number", "value": 1234}, "it has 1,234 releases.")[0]
    assert not run_check({"type": "number", "value": 12}, "version 3.12")[0]


def test_answer_first_uses_answer_line():
    ans = "flask needs 3.9 and fastapi needs 3.10.\nANSWER: FastAPI requires a newer Python than Flask."
    assert run_check({"type": "answer_first", "value": "fastapi", "others": ["flask"]}, ans)[0]
    assert not run_check({"type": "answer_first", "value": "flask", "others": ["fastapi"]}, ans)[0]


def test_yesno():
    assert run_check({"type": "yesno", "value": "no"}, "x\n**ANSWER:** No, typer 0.27 does not.")[0]
    assert not run_check({"type": "yesno", "value": "no"}, "ANSWER: yes")[0]


def test_score_all_or_nothing():
    s = score([{"type": "contains", "value": "a1"}, {"type": "contains", "value": "b2"}], "a1 only")
    assert not s["pass"] and s["check_fraction"] == 0.5
