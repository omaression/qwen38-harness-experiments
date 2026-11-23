"""verdicts: think validity, strict schema validity, the lenient diagnostic, judge_record and the
strict execution verdict (rule values/1)."""

from __future__ import annotations

import json

import pytest

from llm_eval_harness import sandbox, util, verdicts
from llm_eval_harness.records import Check, Task
from llm_eval_harness.sandbox import CallRecord, SandboxResult
from llm_eval_harness.verdicts import BUGFIX_SCHEMA, JUDGE_SCHEMA, ThinkPolicy

from conftest import TINY_TASKS, task_from_dict

JUDGE_OK = '{"verdict": true}'
OPTIONAL = ThinkPolicy()
REQUIRED = ThinkPolicy(required=True)


# ---- THINK -------------------------------------------------------------------------------------

THINK_CASES = [
    # id, content, reasoning, policy, (present, closed, words, valid, source), problems
    ("none", JUDGE_OK, None, OPTIONAL, (False, False, 0, True, "none"), []),
    ("none but required", JUDGE_OK, None, REQUIRED, (False, False, 0, False, "none"), []),
    ("closed block", "<think>check the bound</think>" + JUDGE_OK, None, OPTIONAL, (True, True, 3, True, "content"), []),
    ("leading whitespace", "\n  <think>a b</think>\n" + JUDGE_OK, None, REQUIRED, (True, True, 2, True, "content"), []),
    ("unclosed block", "<think>still thinking when the tokens ran out", None, OPTIONAL, (True, False, 7, False, "content"), []),
    ("closed but no answer", "<think>a b</think>   \n", None, OPTIONAL, (True, False, 2, False, "content"), []),
    ("orphan close tag", "the template opened it</think>" + JUDGE_OK, None, OPTIONAL, (True, True, 4, True, "content"), []),
    ("reasoning field", JUDGE_OK, "one two three", OPTIONAL, (True, True, 3, True, "reasoning"), []),
    ("reasoning field and no answer", "", "one two", OPTIONAL, (True, False, 2, False, "reasoning"), []),
    ("blank reasoning field", JUDGE_OK, "  \n ", OPTIONAL, (False, False, 0, True, "none"), []),
    ("both places", "<think>b</think>" + JUDGE_OK, "a", OPTIONAL, (True, True, 1, False, "reasoning"), [verdicts.BOTH_PLACES]),
    ("two blocks", "<think>a</think>" + JUDGE_OK + "<think>b</think>", None, OPTIONAL, (True, True, 1, False, "content"), [verdicts.MORE_THAN_ONE]),
    ("nested open", "<think>a <think>b</think>" + JUDGE_OK, None, OPTIONAL, (True, True, 2, False, "content"), [verdicts.MORE_THAN_ONE]),
    ("stray close after", "<think>a</think>" + JUDGE_OK + "</think>", None, OPTIONAL, (True, True, 1, False, "content"), [verdicts.MORE_THAN_ONE]),
    ("late open tag", "Sure. <think>a</think>" + JUDGE_OK, None, OPTIONAL, (True, True, 2, False, "content"), [verdicts.LATE_OPEN]),
    ("empty block", "<think>\n\n</think>\n" + JUDGE_OK, None, OPTIONAL, (False, False, 0, True, "none"), []),
    ("empty block, required", "<think></think>" + JUDGE_OK, None, REQUIRED, (False, False, 0, False, "none"), []),
    ("empty block then another", "<think></think>" + JUDGE_OK + "<think>x</think>", None, OPTIONAL, (False, False, 0, False, "none"), [verdicts.MORE_THAN_ONE]),
    ("at the word limit", "<think>" + "w " * 1200 + "</think>" + JUDGE_OK, None, OPTIONAL, (True, True, 1200, True, "content"), []),
    ("over the word limit", "<think>" + "w " * 1201 + "</think>" + JUDGE_OK, None, OPTIONAL, (True, True, 1201, False, "content"), []),
    ("under a minimum", "<think>a b c</think>" + JUDGE_OK, None, ThinkPolicy(min_words=5), (True, True, 3, False, "content"), []),
]


@pytest.mark.parametrize("content, reasoning, policy, expected, problems", [c[1:] for c in THINK_CASES], ids=[c[0] for c in THINK_CASES])
def test_check_think(content: str, reasoning: str | None, policy: ThinkPolicy, expected: tuple, problems: list[str]) -> None:
    check = verdicts.check_think(content, reasoning, policy)
    assert (check.present, check.closed, check.words, check.valid, check.source) == expected
    assert check.problems == problems
    assert check.in_range == (policy.min_words <= check.words <= policy.max_words)


@pytest.mark.parametrize("content, reasoning, thinking, answer", [
    (JUDGE_OK, None, None, JUDGE_OK),
    ("<think>t</think> " + JUDGE_OK, None, "t", " " + JUDGE_OK),
    ("<think>cut off", None, "cut off", ""),
    ("implied</think>" + JUDGE_OK, None, "implied", JUDGE_OK),
    (JUDGE_OK, "separate", "separate", JUDGE_OK),
    ("", None, None, ""),
])
def test_split_reply(content: str, reasoning: str | None, thinking: str | None, answer: str) -> None:
    got_thinking, got_answer, _problems = verdicts.split_reply(content, reasoning)
    assert (got_thinking, got_answer) == (thinking, answer)


def test_count_words() -> None:
    assert verdicts.count_words("  one\ttwo\nthree  ") == 3
    assert verdicts.count_words("") == 0


# ---- SCHEMA ------------------------------------------------------------------------------------

BUGFIX_OK = {"explanation": "max() was given lo twice", "patched_code": "def f():\n    return 1\n"}
GOOD = json.dumps(BUGFIX_OK)

SCHEMA_CASES = [
    # id, answer, schema, finish_reason, valid, lenient_valid, a problem fragment
    ("valid bugfix", GOOD, BUGFIX_SCHEMA, "stop", True, True, None),
    ("valid with surrounding whitespace", "\n  " + GOOD + "\n", BUGFIX_SCHEMA, None, True, True, None),
    ("valid judge", JUDGE_OK, JUDGE_SCHEMA, "stop", True, True, None),
    ("judge with optional reason", '{"verdict": false, "reason": "misses x == 0"}', JUDGE_SCHEMA, None, True, True, None),
    ("code fence", "```json\n" + GOOD + "\n```", BUGFIX_SCHEMA, None, False, True, "not exactly one JSON object"),
    ("prose before", "Here is the fix: " + GOOD, BUGFIX_SCHEMA, None, False, True, "not exactly one JSON object"),
    ("prose after", GOOD + "\nHope this helps!", BUGFIX_SCHEMA, None, False, True, "Extra data"),
    ("two objects", JUDGE_OK + JUDGE_OK, JUDGE_SCHEMA, None, False, False, "Extra data"),
    ("duplicate keys", '{"verdict": true, "verdict": false}', JUDGE_SCHEMA, None, False, True, "duplicate key 'verdict'"),
    ("NaN", '{"verdict": NaN}', JUDGE_SCHEMA, None, False, False, "NaN is not JSON"),
    ("Infinity", '{"verdict": true, "reason": Infinity}', JUDGE_SCHEMA, None, False, False, "Infinity is not JSON"),
    ("extra key", '{"verdict": true, "confidence": 0.9}', JUDGE_SCHEMA, None, False, False, "unexpected key 'confidence'"),
    ("missing key", '{"explanation": "x"}', BUGFIX_SCHEMA, None, False, False, "missing key 'patched_code'"),
    ("wrong type", '{"verdict": "true"}', JUDGE_SCHEMA, None, False, False, "'verdict' is a string, not a boolean"),
    ("int for bool", '{"verdict": 1}', JUDGE_SCHEMA, None, False, False, "'verdict' is an integer, not a boolean"),
    ("null for bool", '{"verdict": null}', JUDGE_SCHEMA, None, False, False, "'verdict' is null, not a boolean"),
    ("blank code", '{"explanation": "x", "patched_code": "  \\n"}', BUGFIX_SCHEMA, None, False, False, "'patched_code' is blank"),
    ("blank explanation", '{"explanation": "", "patched_code": "def f(): pass"}', BUGFIX_SCHEMA, None, False, False, "'explanation' is blank"),
    ("an array", "[" + JUDGE_OK + "]", JUDGE_SCHEMA, None, False, True, "an array, not a JSON object"),
    ("empty answer", "   ", JUDGE_SCHEMA, None, False, False, "no answer"),
    ("truncated", GOOD, BUGFIX_SCHEMA, "length", False, True, "truncated at max_tokens"),
    ("nested too deeply", "[" * 100_000 + "]" * 100_000, JUDGE_SCHEMA, None, False, False, "not exactly one JSON object"),
]


@pytest.mark.parametrize("answer, schema, finish, valid, lenient_valid, fragment", [c[1:] for c in SCHEMA_CASES], ids=[c[0] for c in SCHEMA_CASES])
def test_check_schema(answer: str, schema: verdicts.JsonSchema, finish: str | None, valid: bool, lenient_valid: bool, fragment: str | None) -> None:
    check = verdicts.check_schema(answer, schema, finish_reason=finish)
    assert check.valid is valid
    assert check.lenient_valid is lenient_valid
    if valid:
        assert check.problems == [] and check.value == json.loads(answer)
    else:
        assert check.value is None
        assert any(fragment in problem for problem in check.problems), check.problems


def test_truncation_is_reported_with_the_other_problems() -> None:
    check = verdicts.check_schema('{"verdict": 1}', JUDGE_SCHEMA, finish_reason="length")
    assert check.problems == ["truncated at max_tokens", "'verdict' is an integer, not a boolean"]


def test_problems_list_every_shape_fault() -> None:
    check = verdicts.check_schema('{"patched_code": 3, "extra": 1}', BUGFIX_SCHEMA)
    assert check.problems == ["missing key 'explanation'", "unexpected key 'extra'", "'patched_code' is an integer, not a string"]


# ---- the lenient diagnostic --------------------------------------------------------------------


@pytest.mark.parametrize("answer, expected", [
    (JUDGE_OK, {"verdict": True}),
    ("Sure!\n```json\n" + JUDGE_OK + "\n```\nDone.", {"verdict": True}),
    ("```\n" + JUDGE_OK + "```", {"verdict": True}),
    ("The answer is " + JUDGE_OK + ", I think.", {"verdict": True}),
    ('{"verdict": true, "verdict": false}', {"verdict": False}),
    ("no object at all", None),
    ("[1, 2]", None),
    ('{"verdict": NaN}', None),
    ("{ not json }", None),
    ("", None),
])
def test_lenient_json(answer: str, expected: dict | None) -> None:
    assert verdicts.lenient_json(answer) == expected


def test_lenient_value_is_kept_for_scanning_but_never_valid() -> None:
    check = verdicts.check_schema("```python\n" + GOOD + "\n```", BUGFIX_SCHEMA)
    assert not check.valid and check.value is None
    assert check.lenient_valid and check.lenient_value == BUGFIX_OK


# ---- judge_record ------------------------------------------------------------------------------


def value(index: int, text: str) -> CallRecord:
    return CallRecord(index=index, ok=True, repr=text)


def raised(index: int, *mro: str, msg: str = "") -> CallRecord:
    return CallRecord(index=index, ok=False, exc=mro[0], mro=tuple(mro), msg=msg)


JUDGE_CASES = [
    ("equal", value(0, "3"), Check("f()", expected="3"), ("passed", "")),
    ("equal list", value(0, "[1, 3, 6]"), Check("f()", expected="[1, 3, 6]"), ("passed", "")),
    ("tuple is not a list", value(0, "(1, 3, 6)"), Check("f()", expected="[1, 3, 6]"), ("failed", "got (1, 3, 6), expected [1, 3, 6]")),
    ("none", value(0, "None"), Check("f()", expected="None"), ("passed", "")),
    ("unicode string", value(0, "'é'"), Check("f()", expected="'é'"), ("passed", "")),
    ("wrong value", value(0, "0"), Check("f()", expected="3"), ("failed", "got 0, expected 3")),
    ("approx", value(0, "0.30000000000000004"), Check("f()", expected="0.3", approx=1e-9), ("passed", "")),
    ("approx needed", value(0, "0.30000000000000004"), Check("f()", expected="0.3"), ("failed", "got 0.30000000000000004, expected 0.3")),
    ("approx too tight", value(0, "0.31"), Check("f()", expected="0.3", approx=1e-9), ("failed", "got 0.31, expected 0.3")),
    ("not a literal", value(0, "<object object at 0x7f>"), Check("f()", expected="3"), ("failed", "got <object object at 0x7f>, which is not a literal")),
    ("repr failed", value(0, "<repr() raised ValueError>"), Check("f()", expected="3"), ("failed", "got <repr() raised ValueError>, which is not a literal")),
    ("code in a repr is not run", value(0, "__import__('os').getpid()"), Check("f()", expected="3"), ("failed", "got __import__('os').getpid(), which is not a literal")),
    ("exception for a value", raised(0, "ZeroDivisionError", "ArithmeticError", "Exception", "BaseException", msg="division by zero"), Check("f()", expected="3"), ("error", "raised ZeroDivisionError: division by zero")),
    ("expected exception", raised(0, "ValueError", "Exception", "BaseException"), Check("f()", raises="ValueError"), ("passed", "")),
    ("subclass passes", raised(0, "KeyError", "LookupError", "Exception", "BaseException"), Check("f()", raises="LookupError"), ("passed", "")),
    ("other exception", raised(0, "TypeError", "Exception", "BaseException"), Check("f()", raises="ValueError"), ("failed", "raised TypeError, expected ValueError")),
    ("value for an exception", value(0, "0"), Check("f()", raises="ValueError"), ("failed", "returned 0, expected ValueError")),
    ("system exit is not a value", raised(0, "SystemExit", "BaseException"), Check("f()", expected="0"), ("error", "raised SystemExit")),
    ("no record", None, Check("f()", expected="3"), ("missing", "no result: the process ended before this test reported")),
    ("no record at the deadline", None, Check("f()", raises="ValueError"), ("timeout", "no result before the deadline")),
    ("a test with nothing to compare", value(0, "3"), Check("f()"), ("failed", "the test names neither an expected value nor an exception")),
    ("a bad expected literal", value(0, "3"), Check("f()", expected="three"), ("failed", "the expected value three is not a literal")),
]


@pytest.mark.parametrize("record, check, expected", [c[1:] for c in JUDGE_CASES], ids=[c[0] for c in JUDGE_CASES])
def test_judge_record(record: CallRecord | None, check: Check, expected: tuple[str, str]) -> None:
    timed_out = expected[0] == "timeout"
    assert verdicts.judge_record(record, check, timed_out=timed_out) == expected


def test_judge_record_refuses_an_oversized_repr() -> None:
    big = "[" + "1, " * sandbox.MAX_REPR_CHARS + "1]"
    outcome, detail = verdicts.judge_record(value(0, big), Check("f()", expected="[1]"))
    assert outcome == "failed" and "not a literal" in detail and len(detail) < 120


# ---- exec_verdict on the tiny tasks --------------------------------------------------------------


@pytest.mark.parametrize("task_json", TINY_TASKS, ids=[t["id"] for t in TINY_TASKS])
def test_exec_verdict_on_the_tiny_tasks(task_json: dict) -> None:
    task = task_from_dict(task_json)
    buggy = verdicts.exec_verdict(task, task.buggy_code, timeout_s=5)
    reference = verdicts.exec_verdict(task, task.reference_code, timeout_s=5)
    wrong = verdicts.exec_verdict(task, task.wrong_code, timeout_s=5)
    assert not buggy.shown_passed and not buggy.passed
    assert reference.passed and reference.shown_passed and reference.hidden_passed
    assert reference.problems == [] and reference.exit_code == 0 and reference.infra_error is None
    assert wrong.shown_passed and not wrong.hidden_passed and not wrong.passed
    n_shown = len(task.shown_tests)
    assert [r.kind for r in reference.results] == ["shown"] * n_shown + ["hidden"] * len(task.hidden_tests)
    assert [r.index for r in reference.results] == list(range(len(task.all_tests)))
    assert all(r.outcome == "passed" for r in reference.results)


def tiny(task_id: str) -> Task:
    return next(task_from_dict(t) for t in TINY_TASKS if t["id"] == task_id)


def test_wrong_code_details_name_the_failed_hidden_test() -> None:
    task = tiny("t-clamp")
    wrong = verdicts.exec_verdict(task, task.wrong_code, timeout_s=5)
    failed = [(r.kind, r.outcome, r.detail) for r in wrong.results if r.outcome != "passed"]
    assert failed == [("hidden", "failed", "got 0, expected 2"), ("hidden", "failed", "got 0, expected 3")]


def test_exit_zero_at_import_is_never_a_pass() -> None:
    task = tiny("t-clamp")
    code = "import os\n" + task.reference_code + "os._exit(0)\n"
    verdict = verdicts.exec_verdict(task, code, timeout_s=5)
    assert verdict.exit_code == 0 and verdict.infra_error is None
    assert not verdict.passed and not verdict.shown_passed and not verdict.hidden_passed
    assert {r.outcome for r in verdict.results} == {"missing"}
    assert any("exit status 0" in p for p in verdict.problems)


def test_exit_zero_after_the_shown_test_does_not_pass_the_hidden_ones() -> None:
    task = tiny("t-clamp")
    code = task.reference_code.replace("    return max(lo, min(x, hi))\n", "    if x != 5:\n        import os\n        os._exit(0)\n    return max(lo, min(x, hi))\n")
    verdict = verdicts.exec_verdict(task, code, timeout_s=5)
    assert verdict.shown_passed and not verdict.hidden_passed and not verdict.passed
    assert [r.outcome for r in verdict.results] == ["passed", "missing", "missing", "missing"]


def test_a_run_with_exit_zero_and_no_records_fails_every_test(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SandboxResult(started=True, exit_code=0, timed_out=False, duration_s=0.01)
    monkeypatch.setattr(sandbox, "run_calls", lambda *a, **k: fake)
    task = tiny("t-parse-port")
    verdict = verdicts.exec_verdict(task, task.reference_code)
    assert not verdict.passed and verdict.infra_error is None
    assert {r.outcome for r in verdict.results} == {"missing"}


def test_zero_tests_never_pass() -> None:
    task = Task(id="t-empty", entry_point="f", buggy_code="def f(): return 0\n", shown_tests=(), hidden_tests=(),
                reference_code="def f(): return 1\n", wrong_code="def f(): return 2\n")
    verdict = verdicts.exec_verdict(task, task.reference_code)
    assert not verdict.passed and not verdict.shown_passed and not verdict.hidden_passed
    assert verdict.problems == ["the task has no tests"]


def test_a_task_without_hidden_tests_never_passes() -> None:
    task = Task(id="t-shown-only", entry_point="f", buggy_code="def f(): return 0\n", shown_tests=(Check("f()", expected="1"),),
                hidden_tests=(), reference_code="def f(): return 1\n", wrong_code="def f(): return 2\n")
    verdict = verdicts.exec_verdict(task, task.reference_code, timeout_s=5)
    assert verdict.shown_passed and not verdict.hidden_passed and not verdict.passed


def test_a_load_error_fails_every_test_even_an_expected_exception() -> None:
    task = tiny("t-parse-port")  # its shown test expects ValueError
    verdict = verdicts.exec_verdict(task, "raise ValueError('at import')\n", timeout_s=5)
    assert not verdict.shown_passed and not verdict.passed
    assert {r.outcome for r in verdict.results} == {"error"}
    assert verdict.results[0].detail == "the code did not load: ValueError: at import"


def test_a_forged_result_line_makes_its_test_fail() -> None:
    task = tiny("t-clamp")
    forge = 'import os, sys\nos.write(int(sys.argv[1]), b\'{"i": 0, "ok": true, "repr": "3"}\\n\')\n'
    verdict = verdicts.exec_verdict(task, forge + task.buggy_code, timeout_s=5)
    assert verdict.results[0].outcome == "missing" and "more than once" in verdict.results[0].detail
    assert not verdict.shown_passed and not verdict.passed
    assert any("more than once" in p for p in verdict.problems)


def test_a_hang_times_out_and_earlier_results_count() -> None:
    task = tiny("t-clamp")
    code = task.reference_code.replace("    return max", "    while x == 2:\n        pass\n    return max")
    verdict = verdicts.exec_verdict(task, code, timeout_s=1.0)
    assert verdict.timed_out and not verdict.passed
    assert [r.outcome for r in verdict.results] == ["passed", "passed", "timeout", "timeout"]


def test_system_exit_in_a_call_is_an_exception() -> None:
    task = Task(id="t-exit", entry_point="f", buggy_code="def f(x): return x\n",
                shown_tests=(Check("f(0)", expected="0"),), hidden_tests=(Check("f(1)", raises="SystemExit"), Check("f(2)", expected="2")),
                reference_code="def f(x): return x\n", wrong_code="def f(x): return 0\n")
    code = "import sys\ndef f(x):\n    sys.exit(x)\n"
    verdict = verdicts.exec_verdict(task, code, timeout_s=5)
    assert [r.outcome for r in verdict.results] == ["error", "passed", "error"]
    assert not verdict.passed


def test_infra_errors_are_passed_through_and_never_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = SandboxResult(started=False, exit_code=None, timed_out=False, duration_s=0.0, infra_error="could not start the interpreter")
    monkeypatch.setattr(sandbox, "run_calls", lambda *a, **k: broken)
    task = tiny("t-clamp")
    verdict = verdicts.exec_verdict(task, task.reference_code)
    assert verdict.infra_error == "could not start the interpreter"
    assert not verdict.passed and verdict.results == []


def test_a_real_start_failure_is_an_infra_error() -> None:
    task = tiny("t-clamp")
    verdict = verdicts.exec_verdict(task, task.reference_code, python="/nonexistent/python3")
    assert verdict.infra_error is not None and not verdict.passed and verdict.results == []


def test_a_platform_without_the_sandbox_is_an_infra_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_posix", lambda: False)
    task = tiny("t-clamp")
    verdict = verdicts.exec_verdict(task, task.reference_code)
    assert verdict.infra_error is not None and "POSIX" in verdict.infra_error and not verdict.passed


def test_exec_verdict_to_json() -> None:
    task = tiny("t-count-vowels")
    verdict = verdicts.exec_verdict(task, task.wrong_code, timeout_s=5)
    data = verdict.to_json()
    assert set(data) == {"rule", "passed", "shown_passed", "hidden_passed", "timed_out", "exit_code", "infra_error", "problems", "results"}
    assert data["rule"] == verdicts.EXEC_RULE == "values/1"
    assert data["results"][0] == {"index": 0, "kind": "shown", "outcome": "passed", "detail": ""}
    assert json.loads(util.canonical_json(data)) == data
    again = verdicts.exec_verdict(task, task.wrong_code, timeout_s=5).to_json()
    assert util.canonical_json(again) == util.canonical_json(data), "the same reply gives the same detail"
