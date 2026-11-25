"""tasks.py: loading a suite and every refusal, the dev / held-out split, the structure checks,
and twin detection. Nothing here executes task code."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from conftest import TINY_MANIFEST, TINY_TASKS, task_from_dict, write_suite
from llm_eval_harness import tasks
from llm_eval_harness.records import ALL, DEV, HELDOUT, Check, Suite, Task
from llm_eval_harness.util import sha256_json

# ---- helpers ------------------------------------------------------------------------------------


def tiny_json() -> list[dict[str, Any]]:
    """A deep copy of the fixture tasks as JSON-ready dicts, safe to edit."""
    return copy.deepcopy(TINY_TASKS)


def suite_of(task_list: list[Task], heldout_pct: int = 50, name: str = "t") -> Suite:
    """A Suite built from records directly (no loader): check_structure must cope with anything."""
    return Suite(name=name, version=1, heldout_pct=heldout_pct, tasks=tuple(task_list), sha256="0" * 64, source="t")


def write_lines(root: Path, lines: list[str], manifest: dict[str, Any] | None = None) -> Path:
    """A suite directory whose tasks.jsonl holds exactly these lines (for line-number tests)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "suite.json").write_text(json.dumps(manifest or TINY_MANIFEST) + "\n", encoding="utf-8")
    (root / "tasks.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return root


def ids_in(split: str, n: int, heldout_pct: int = 50) -> list[str]:
    """n ids that land in `split`, found by scanning candidate names: the split is a hash, so a
    test that needs a task on a given side must look for an id there rather than assume one."""
    held = split == HELDOUT
    found = [f"case-{i}" for i in range(2000) if (tasks.bucket(f"case-{i}") < heldout_pct) == held]
    assert len(found) >= n
    return found[:n]


def ids_on_both_sides() -> tuple[str, str]:
    """A held-out id and a dev id (heldout_pct 50)."""
    return ids_in(HELDOUT, 1)[0], ids_in(DEV, 1)[0]


CLAMP = task_from_dict(TINY_TASKS[0])
VOWELS = task_from_dict(TINY_TASKS[1])


def clamp_twin(task_id: str) -> Task:
    """The clamp task with the function, its parameters, the docstring and the constants changed:
    the same problem in other words."""
    code = 'def limit(value, low, high):\n    """Keep value between low and high."""\n    return max(low, min(value, low))\n'
    return dataclasses.replace(
        CLAMP,
        id=task_id,
        entry_point="limit",
        buggy_code=code,
        reference_code=code.replace("min(value, low)", "min(value, high)"),
        wrong_code=code.replace("return", "if value == 9:\n        return 4\n    return"),
        shown_tests=(Check("limit(9, 1, 4)", expected="4"),),
        hidden_tests=(Check("limit(0, 1, 4)", expected="1"), Check("limit(2, 1, 4)", expected="2")),
    )


# ---- loading ------------------------------------------------------------------------------------


def test_load_the_tiny_suite(tiny_suite_dir: Path, tiny_tasks: list[Task]) -> None:
    suite = tasks.load_suite(tiny_suite_dir)
    assert (suite.name, suite.version, suite.heldout_pct, suite.synthetic) == ("tiny", 1, 50, True)
    assert suite.description == TINY_MANIFEST["description"]
    assert suite.tasks == tuple(tiny_tasks)
    assert not suite.source.startswith("/") and suite.source.endswith("tiny-suite")
    assert len(suite.sha256) == 64 and int(suite.sha256, 16) >= 0


def test_suite_sha256_is_the_documented_hash(tiny_suite_dir: Path) -> None:
    suite = tasks.load_suite(tiny_suite_dir)
    manifest = json.loads((tiny_suite_dir / "suite.json").read_text(encoding="utf-8"))
    expected = sha256_json({"manifest": manifest, "tasks": [t.to_json() for t in sorted(suite.tasks, key=lambda t: t.id)]})
    assert suite.sha256 == expected


def test_suite_sha256_ignores_formatting_and_line_order(tmp_path: Path) -> None:
    plain = tasks.load_suite(write_suite(tmp_path / "plain", TINY_TASKS))
    other = tmp_path / "other"
    other.mkdir()
    reordered = dict(reversed(list(TINY_MANIFEST.items())))
    (other / "suite.json").write_text(json.dumps(reordered, separators=(",", ":")), encoding="utf-8")
    lines = []
    for record in reversed(TINY_TASKS):
        record = dict(reversed(list(record.items())))
        record.setdefault("group", None)  # an explicit null is the same as no group
        lines.append(json.dumps(record, indent=None, separators=(", ", " : ")))
    (other / "tasks.jsonl").write_text("\n\n".join(lines) + "\n\n", encoding="utf-8")
    assert tasks.load_suite(other).sha256 == plain.sha256

    changed = tiny_json()
    changed[2]["hidden_tests"][0]["expected"] = "[0]"
    assert tasks.load_suite(write_suite(tmp_path / "changed", changed)).sha256 != plain.sha256
    renamed = tasks.load_suite(write_suite(tmp_path / "renamed", TINY_TASKS, name="tiny-2"))
    assert renamed.sha256 != plain.sha256


def test_source_is_relative_inside_the_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_suite(tmp_path / "suites" / "tiny", TINY_TASKS)
    monkeypatch.chdir(tmp_path)
    assert tasks.load_suite("suites/tiny").source == "suites/tiny"


def test_parse_task_round_trips_to_json() -> None:
    for task in (CLAMP, VOWELS):
        assert tasks.parse_task(task.to_json(), "x") == task


def test_parse_check_types() -> None:
    assert tasks.parse_check({"call": "f(1)", "expected": "2"}, "x") == Check("f(1)", expected="2")
    assert tasks.parse_check({"call": "f('a')", "raises": "KeyError"}, "x") == Check("f('a')", raises="KeyError")
    check = tasks.parse_check({"call": "f(1)", "expected": "0.5", "approx": 1}, "x")
    assert check.approx == 1.0 and isinstance(check.approx, float)


# ---- refusals -----------------------------------------------------------------------------------


def _edit(index: int, **changes: Any) -> list[dict[str, Any]]:
    records = tiny_json()
    for key, value in changes.items():
        if value is _DELETE:
            del records[index][key]
        else:
            records[index][key] = value
    return records


_DELETE = object()

REFUSED_TASKS = [
    # (records, pattern the message must match)
    (lambda: [{**{k: v for k, v in r.items() if k != "hidden_tests"}, "hiden_tests": r["hidden_tests"]} for r in tiny_json()],
     r"tasks\.jsonl:1: unknown key 'hiden_tests' in a task \(did you mean 'hidden_tests'\?\)"),
    (lambda: _edit(1, wrong_code=_DELETE), r"tasks\.jsonl:2: missing key 'wrong_code'"),
    (lambda: _edit(0, shown_tests=[{"call": "clamp(5, 0, 3)", "expect": "3"}]), r"tasks\.jsonl:1: shown_tests\[0\]: unknown key 'expect'"),
    (lambda: _edit(0, shown_tests=[{"expected": "3"}]), r"shown_tests\[0\]: missing key 'call'"),
    (lambda: _edit(0, id=7), r"tasks\.jsonl:1: id must be a string, not a number"),
    (lambda: _edit(0, buggy_code=None), r"buggy_code must be a string, not null"),
    (lambda: _edit(0, shown_tests="clamp(5, 0, 3)"), r"shown_tests must be a list of test objects"),
    (lambda: _edit(0, hidden_tests=["clamp(2, 0, 3)"]), r"hidden_tests\[0\]: a test must be a JSON object, not a string"),
    (lambda: _edit(0, shown_tests=[{"call": "clamp(5, 0, 3)", "expected": 3}]), r"expected must be a string, not a number"),
    (lambda: _edit(0, shown_tests=[{"call": "clamp(5, 0, 3)", "expected": None}]), r'expected must be a string, not null \(write "None"'),
    (lambda: _edit(3, shown_tests=[{"call": "parse_port('0')", "raises": 1}]), r"raises must be a string"),
    (lambda: _edit(0, shown_tests=[{"call": 5, "expected": "3"}]), r"call must be a string"),
    (lambda: _edit(0, shown_tests=[{"call": "clamp(5, 0, 3)", "expected": "3", "approx": "1e-9"}]), r"approx must be a number, not a string"),
    (lambda: _edit(0, shown_tests=[{"call": "clamp(5, 0, 3)", "expected": "3", "approx": True}]), r"approx must be a number, not a boolean"),
    (lambda: _edit(0, tags="boundary"), r"tags must be a list of strings"),
    (lambda: _edit(0, tags=["ok", 3]), r"tags must be a list of strings"),
    (lambda: _edit(0, synthetic="yes"), r"synthetic must be true or false"),
    (lambda: _edit(0, group=5), r"group must be a string or null"),
    (lambda: [tiny_json()[0], tiny_json()[1], tiny_json()[0]], r"tasks\.jsonl:3: duplicate task id 't-clamp' \(first on line 1\)"),
]


@pytest.mark.parametrize("make, pattern", REFUSED_TASKS)
def test_refused_task_lines(tmp_path: Path, make: Any, pattern: str) -> None:
    root = write_suite(tmp_path / "suite", make())
    with pytest.raises(tasks.SuiteError, match=pattern):
        tasks.load_suite(root)


@pytest.mark.parametrize(
    "line3, pattern",
    [
        ('{"id": "x", ', r"tasks\.jsonl:3: not JSON"),
        ("[1, 2]", r"tasks\.jsonl:3: an array, not a JSON object"),
        ('"just a string"', r"tasks\.jsonl:3: a string, not a JSON object"),
        ('{"id": "a", "id": "b"}', r"tasks\.jsonl:3: duplicate key 'id' in one object"),
        ('{"id": NaN}', r"tasks\.jsonl:3: NaN is not JSON"),
    ],
)
def test_bad_json_names_the_file_and_line(tmp_path: Path, line3: str, pattern: str) -> None:
    good = [json.dumps(TINY_TASKS[0]), ""]  # line 2 is blank: blank lines are skipped, not renumbered
    with pytest.raises(tasks.SuiteError, match=pattern):
        tasks.load_suite(write_lines(tmp_path / "s", good + [line3]))


@pytest.mark.parametrize(
    "manifest, pattern",
    [
        ({**TINY_MANIFEST, "heldout": 50}, r"suite\.json: unknown key 'heldout' in a suite manifest \(did you mean 'heldout_pct'\?\)"),
        ({k: v for k, v in TINY_MANIFEST.items() if k != "name"}, r"suite\.json: missing key 'name'"),
        ({**TINY_MANIFEST, "schema": "llm_eval_harness.suite/2"}, r"schema is 'llm_eval_harness.suite/2'"),
        ({**TINY_MANIFEST, "heldout_pct": -1}, r"heldout_pct must be an integer from 0 to 100"),
        ({**TINY_MANIFEST, "heldout_pct": 101}, r"heldout_pct must be an integer from 0 to 100"),
        ({**TINY_MANIFEST, "heldout_pct": 50.0}, r"heldout_pct must be an integer"),
        ({**TINY_MANIFEST, "heldout_pct": True}, r"heldout_pct must be an integer"),
        ({**TINY_MANIFEST, "heldout_pct": "50"}, r"heldout_pct must be an integer"),
        ({**TINY_MANIFEST, "version": "1"}, r"version must be an integer"),
        ({**TINY_MANIFEST, "name": " "}, r"name must be a non-empty string"),
        ({**TINY_MANIFEST, "synthetic": "true"}, r"synthetic must be true or false"),
        ({**TINY_MANIFEST, "description": 3}, r"description must be a string"),
    ],
)
def test_refused_manifests(tmp_path: Path, manifest: dict[str, Any], pattern: str) -> None:
    root = tmp_path / "s"
    root.mkdir()
    (root / "suite.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "tasks.jsonl").write_text(json.dumps(TINY_TASKS[0]) + "\n", encoding="utf-8")
    with pytest.raises(tasks.SuiteError, match=pattern):
        tasks.load_suite(root)


@pytest.mark.parametrize("pct", [0, 100])
def test_heldout_pct_bounds_are_accepted(tmp_path: Path, pct: int) -> None:
    assert tasks.load_suite(write_suite(tmp_path / "s", TINY_TASKS, heldout_pct=pct)).heldout_pct == pct


def test_refused_files(tmp_path: Path) -> None:
    with pytest.raises(tasks.SuiteError, match=r"not a directory"):
        tasks.load_suite(tmp_path / "missing")
    root = tmp_path / "s"
    root.mkdir()
    with pytest.raises(tasks.SuiteError, match=r"suite\.json: no such file"):
        tasks.load_suite(root)
    (root / "suite.json").write_text('{"schema": "llm_eval_harness.suite/1",\n "name": }', encoding="utf-8")
    with pytest.raises(tasks.SuiteError, match=r"suite\.json:2: not JSON"):
        tasks.load_suite(root)
    (root / "suite.json").write_text("[]", encoding="utf-8")
    with pytest.raises(tasks.SuiteError, match=r"suite\.json: an array, not a JSON object"):
        tasks.load_suite(root)
    (root / "suite.json").write_text(json.dumps(TINY_MANIFEST), encoding="utf-8")
    with pytest.raises(tasks.SuiteError, match=r"tasks\.jsonl: no such file"):
        tasks.load_suite(root)
    (root / "tasks.jsonl").write_text("\n  \n", encoding="utf-8")
    with pytest.raises(tasks.SuiteError, match=r"tasks\.jsonl: holds no tasks"):
        tasks.load_suite(root)
    (root / "tasks.jsonl").write_bytes(b'{"id": "\xff"}\n')
    with pytest.raises(tasks.SuiteError, match=r"tasks\.jsonl: not UTF-8"):
        tasks.load_suite(root)


def test_a_line_separator_inside_a_string_is_not_a_line_break(tmp_path: Path) -> None:
    record = tiny_json()[0]
    record["buggy_code"] = record["buggy_code"].replace("Return x", "Return x")
    root = write_lines(tmp_path / "s", [json.dumps(record, ensure_ascii=False)])
    assert " " in tasks.load_suite(root).tasks[0].buggy_code


def test_suite_error_is_a_value_error() -> None:
    assert issubclass(tasks.SuiteError, ValueError)


# ---- the split ----------------------------------------------------------------------------------


def test_bucket_is_the_documented_hash() -> None:
    for key in ["t-clamp", "digit-sum", "Clamp Range", "ünïcode key"]:
        digest = hashlib.sha256(tasks.normalize_key(key).encode("utf-8")).hexdigest()
        assert tasks.bucket(key) == int(digest[:8], 16) % 100


def test_bucket_values_are_pinned() -> None:
    # A change here moves tasks across the split: every held-out result would become a dev one.
    assert {k: tasks.bucket(k) for k in ["t-clamp", "t-count-vowels", "t-parse-port", "digit-sum", "clamp range"]} == {
        "t-clamp": 43, "t-count-vowels": 95, "t-parse-port": 64, "digit-sum": 46, "clamp range": 25,
    }


def test_normalize_key() -> None:
    assert tasks.normalize_key("  Clamp \t Range\n") == "clamp range"
    assert tasks.normalize_key("clamp range") == "clamp range"
    assert tasks.normalize_key("A  B") == tasks.normalize_key(tasks.normalize_key("a b"))
    assert tasks.bucket("Clamp  Range ") == tasks.bucket("clamp range") == 25


def test_split_of_compares_the_bucket_with_heldout_pct() -> None:
    b = tasks.bucket(CLAMP.id)
    assert tasks.split_of(CLAMP, b) == DEV
    assert tasks.split_of(CLAMP, b + 1) == HELDOUT
    assert tasks.split_of(CLAMP, 0) == DEV
    assert tasks.split_of(CLAMP, 100) == HELDOUT


def test_a_group_decides_the_split_of_all_its_tasks() -> None:
    first = dataclasses.replace(CLAMP, id="first", group="Shared Group")
    second = dataclasses.replace(VOWELS, id="second", group="shared  group")
    alone = dataclasses.replace(CLAMP, id="shared group")
    for pct in range(101):
        assert tasks.split_of(first, pct) == tasks.split_of(second, pct) == tasks.split_of(alone, pct)
    held, dev = ids_on_both_sides()
    assert tasks.split_of(dataclasses.replace(CLAMP, id=dev, group=held), 50) == HELDOUT


def test_tasks_in_split(tiny_suite_dir: Path) -> None:
    suite = tasks.load_suite(tiny_suite_dir)
    dev, held, both = (tasks.tasks_in_split(suite, s) for s in (DEV, HELDOUT, ALL))
    assert [t.id for t in held] == ["t-clamp"]
    assert [t.id for t in dev] == ["t-count-vowels", "t-parse-port", "t-running-total"]
    assert [t.id for t in both] == sorted(t.id for t in suite.tasks)
    assert set(dev) | set(held) == set(both) and not set(dev) & set(held)
    assert tasks.split_counts(suite) == {"dev": 3, "heldout": 1, "all": 4}
    with pytest.raises(ValueError, match="unknown split"):
        tasks.tasks_in_split(suite, "train")


def test_split_sha256_is_the_documented_hash(tiny_suite_dir: Path) -> None:
    suite = tasks.load_suite(tiny_suite_dir)
    assert tasks.split_sha256(suite, DEV) == sha256_json({"suite": "tiny", "split": "dev", "tasks": ["t-count-vowels", "t-parse-port", "t-running-total"]})
    assert len({tasks.split_sha256(suite, s) for s in (DEV, HELDOUT, ALL)}) == 3
    with pytest.raises(ValueError):
        tasks.split_sha256(suite, "test")


def test_split_sha256_ignores_task_text_but_not_membership(tmp_path: Path) -> None:
    base = tasks.load_suite(write_suite(tmp_path / "base", TINY_TASKS))
    edited_records = tiny_json()
    edited_records[0]["buggy_code"] = edited_records[0]["buggy_code"].replace("closed interval", "closed range")
    edited_records[0]["hidden_tests"].append({"call": "clamp(1, 0, 3)", "expected": "1"})
    edited = tasks.load_suite(write_suite(tmp_path / "edited", edited_records))
    assert edited.sha256 != base.sha256
    for split in (DEV, HELDOUT, ALL):
        assert tasks.split_sha256(edited, split) == tasks.split_sha256(base, split)

    held, _ = ids_on_both_sides()
    grown = tasks.load_suite(write_suite(tmp_path / "grown", TINY_TASKS + [{**TINY_TASKS[1], "id": held}]))
    assert tasks.split_sha256(grown, HELDOUT) != tasks.split_sha256(base, HELDOUT)
    assert tasks.split_sha256(grown, DEV) == tasks.split_sha256(base, DEV)


# ---- check_structure ----------------------------------------------------------------------------


def test_the_tiny_suite_is_well_formed(tiny_tasks: list[Task]) -> None:
    assert tasks.check_structure(suite_of(tiny_tasks)) == []


def _clamp(**changes: Any) -> Task:
    return dataclasses.replace(CLAMP, **changes)


SHOWN = CLAMP.shown_tests
HIDDEN = CLAMP.hidden_tests
ONE_HIDDEN = (Check("clamp(-2, 0, 3)", expected="0"),)

STRUCTURE_RULES = [
    # (the broken task, a phrase of the problem it must cause)
    (_clamp(id="Clamp_Task"), "the id does not match"),
    (_clamp(id="-clamp"), "the id does not match"),
    (_clamp(group=""), "the group is empty"),
    (_clamp(group="   "), "the group is empty"),
    (_clamp(entry_point="clamp-it"), "is not a Python identifier"),
    (_clamp(entry_point="class"), "is not a Python identifier"),
    (_clamp(buggy_code="def clamp(x, lo, hi:\n    return x\n"), "buggy_code does not parse (line 1"),
    (_clamp(reference_code="def bound(x, lo, hi):\n    return x\n"), "reference_code defines no top-level function clamp()"),
    (_clamp(wrong_code="class Box:\n    def clamp(self, x, lo, hi):\n        return x\n"), "wrong_code defines no top-level function clamp()"),
    (_clamp(wrong_code="async def clamp(x, lo, hi):\n    return x\n"), "wrong_code defines no top-level function clamp()"),
    (_clamp(shown_tests=()), "there is no shown test"),
    (_clamp(hidden_tests=ONE_HIDDEN), "1 hidden test(s); at least 2 are needed"),
    (_clamp(hidden_tests=HIDDEN + (Check("clamp(5, 0", expected="3"),)), "hidden_tests[3]: the call 'clamp(5, 0' is not one Python expression"),
    (_clamp(hidden_tests=HIDDEN + (Check("x = clamp(1, 0, 3)", expected="1"),)), "is not one Python expression"),
    (_clamp(hidden_tests=HIDDEN + (Check(" clamp(1, 0, 3)", expected="1"),)), "is not one Python expression"),
    (_clamp(hidden_tests=HIDDEN + (Check("  ", expected="1"),)), "hidden_tests[3]: the call is empty"),
    (_clamp(hidden_tests=HIDDEN + (Check("clamp(1, 0, 3)", expected="1", raises="ValueError"),)), "give either expected or raises, not both"),
    (_clamp(hidden_tests=HIDDEN + (Check("clamp(1, 0, 3)"),)), "give expected or raises"),
    (_clamp(hidden_tests=HIDDEN + (Check("clamp(1, 0, 3)", expected="one()"),)), "expected 'one()' is not a Python literal"),
    (_clamp(hidden_tests=HIDDEN + (Check("clamp(1, 0, 3)", expected="frozenset({1})"),)), "is not a Python literal"),
    (_clamp(hidden_tests=HIDDEN + (Check("clamp(1, 0, 3)", raises="Value Error"),)), "raises 'Value Error' is not the name of an exception class"),
    (_clamp(hidden_tests=HIDDEN + (Check("clamp(1, 0, 3)", raises="ValueError()"),)), "is not the name of an exception class"),
    (_clamp(hidden_tests=HIDDEN + (Check("clamp(1, 0, 3)", expected="1", approx=0.0),)), "approx 0.0 is not a positive number"),
    (_clamp(hidden_tests=HIDDEN + (Check("clamp(1, 0, 3)", expected="1", approx=-1e-9),)), "is not a positive number"),
    (_clamp(hidden_tests=HIDDEN + (Check("clamp(1, 0, 3)", expected="1", approx=float("inf")),)), "is not a positive number"),
    (_clamp(hidden_tests=HIDDEN + (Check("clamp(1, 0, 3)", expected="'one'", approx=1e-9),)), "approx needs an expected number"),
    (_clamp(hidden_tests=HIDDEN + (Check("clamp(1, 0, 3)", expected="True", approx=1e-9),)), "approx needs an expected number"),
    (_clamp(hidden_tests=HIDDEN + (Check("clamp(1, 0, 3)", raises="ValueError", approx=1e-9),)), "approx needs an expected number"),
    (_clamp(hidden_tests=HIDDEN + (Check("clamp(5,0,3)", expected="3"),)), "hidden_tests[3] repeats the call of shown_tests[0]"),
    (_clamp(hidden_tests=HIDDEN + (Check("clamp(2, 0, 3)", expected="1"),)), "hidden_tests[3] repeats the call of hidden_tests[1]"),
    (_clamp(buggy_code=CLAMP.buggy_code.replace('[lo, hi]."""', '[lo, hi]: clamp(-2, 0, 3) is 0."""')),
     "hidden_tests[0]: its call 'clamp(-2, 0, 3)' appears in buggy_code"),
    (_clamp(buggy_code=CLAMP.buggy_code.replace('[lo, hi]."""', '[lo, hi], e.g. clamp(-2,0,3)."""')), "appears in buggy_code"),
    (_clamp(reference_code=CLAMP.buggy_code), "buggy_code and reference_code do not differ"),
    (_clamp(wrong_code=CLAMP.buggy_code.replace("\n    return", "\n    # the same code, with a comment\n    return")), "buggy_code and wrong_code do not differ"),
    (_clamp(wrong_code=CLAMP.reference_code), "reference_code and wrong_code do not differ"),
]


@pytest.mark.parametrize("task, phrase", STRUCTURE_RULES, ids=[phrase[:40] for _, phrase in STRUCTURE_RULES])
def test_each_structure_rule(task: Task, phrase: str) -> None:
    problems = tasks.check_structure(suite_of([task, task_from_dict(TINY_TASKS[1])]))
    assert any(phrase in p for p in problems), problems
    assert all(p.startswith(f"task {task.id!r}: ") for p in problems), problems


def test_set_and_none_literals_are_accepted() -> None:
    task = _clamp(hidden_tests=HIDDEN + (Check("clamp(1, 0, 3)", expected="set()"), Check("clamp(0, 0, 3)", expected="None")))
    assert tasks.check_structure(suite_of([task])) == []


def test_a_shown_call_in_the_docstring_is_fine() -> None:
    task = _clamp(buggy_code=CLAMP.buggy_code.replace('[lo, hi]."""', '[lo, hi]: clamp(5, 0, 3) is 3."""'))
    assert tasks.check_structure(suite_of([task])) == []


@pytest.mark.parametrize(
    "call, text, found",
    [
        ("clamp(-2, 0, 3)", "clamp(-2, 0, 3)", True),
        ("clamp(-2, 0, 3)", 'e.g. "clamp(-2,0,3) is 0"', True),
        ("clamp(1,0,3)", "clamp( 1 , 0 , 3 )", True),
        ("f(x=1)", "f(x = 1)", True),
        ("(lambda t: f(t))(1)", "(lambda  t : f( t ))( 1 )", True),
        ("clamp(1, 0, 3)", "unclamp(1, 0, 3)", True),  # verbatim: the plain substring rule
        ("clamp(1, 0, 3)", "unclamp(1,0,3)", False),  # respaced: only at a name boundary
        ("clamp(1, 0, 3)", "box.clamp(1,0,3)", False),
        ("(lambda t: f(t))(1)", "(lambdat: f(t))(1)", False),
        ("parse_int_list(' [ ] ')", "assert parse_int_list('[]') == []", False),  # spaces inside a literal count
        ('f("a b")', 'f("a  b")', False),
        ("f('a')", 'f("a")', False),
        ("f(f'{x:{w}}', 2)", "f( f'{x:{w}}' ,2)", True),
        ("f(", "f(", True),
        ("f(", "g(", False),
    ],
)
def test_call_appears_in(call: str, text: str, found: bool) -> None:
    assert tasks._call_appears_in(call, text) is found


def test_a_hidden_call_that_differs_inside_a_string_is_not_flagged() -> None:
    # The shown call parse_int_list('[]') in the docstring is not the hidden call
    # parse_int_list(' [ ] '): squeezing all whitespace would wrongly make them one.
    record = next(t for t in tasks.load_suite(Path(__file__).resolve().parents[1] / "tasks" / "demo").tasks if t.id == "parse-int-list")
    doc_example = record.buggy_code.replace('like "[4, -1, 7]"', "like parse_int_list('[]')")
    assert doc_example != record.buggy_code
    assert tasks.check_structure(suite_of([dataclasses.replace(record, buggy_code=doc_example)])) == []


def test_duplicate_ids_are_reported_once() -> None:
    problems = tasks.check_structure(suite_of([CLAMP, CLAMP, CLAMP, VOWELS]))
    assert problems == ["task 't-clamp': more than one task has this id"]


def test_check_structure_survives_any_values() -> None:
    junk = Task(id=None, entry_point=None, buggy_code=None, shown_tests=(Check(None, expected=3, approx="x"),),  # type: ignore[arg-type]
                hidden_tests=(), reference_code=b"", wrong_code=1)  # type: ignore[arg-type]
    problems = tasks.check_structure(suite_of([junk]))
    for phrase in ["id does not match", "not a Python identifier", "buggy_code is not a string", "call is not a string",
                   "expected is not a string", "approx 'x' is not a positive number", "0 hidden test(s)"]:
        assert any(phrase in p for p in problems), (phrase, problems)


# ---- normalize_code -----------------------------------------------------------------------------


def test_normalize_code_is_pinned() -> None:
    code = (
        "import math\n\n\n"
        "def area(w, h):\n"
        '    """The area of a right triangle."""\n'
        "    # half of the rectangle\n"
        "    return round(w * h * 0.5, 2) if w else 'none'\n"
    )
    assert tasks.normalize_code(code) == [
        "import", "ID",
        "def", "ID", "(", "ID", ",", "ID", ")", ":",
        "return", "round", "(", "ID", "*", "ID", "*", "NUM", ",", "NUM", ")", "if", "ID", "else", "STR",
    ]


def test_names_messages_and_constants_normalise_away() -> None:
    first = 'def clamp(x, lo, hi):\n    """Limit x."""\n    if x > 100:\n        raise ValueError("too big")\n    return max(lo, min(x, hi))\n'
    second = 'def bound(value, low, high):\n    """Keep value in range."""\n    if value > 7:\n        raise ValueError(f"{value} is large")\n    return max(low, min(value, high))\n'
    assert tasks.normalize_code(first) == tasks.normalize_code(second)
    assert "max" in tasks.normalize_code(first) and "ValueError" in tasks.normalize_code(first)


def test_layout_comments_and_string_statements_are_dropped() -> None:
    compact = "def f(x):\n  return [y for y in x if y]\n"
    spread = 'def f(x):\n\n    """Doc\n    on two lines."""\n    "a stray string"   # and a comment\n    return [y\n            for y in x\n            if y]\n'
    assert tasks.normalize_code(compact) == tasks.normalize_code(spread)
    assert tasks.normalize_code("x = 'kept'\n") == ["ID", "=", "STR"]


def test_f_strings_are_one_string_on_every_python() -> None:
    code = "def f(x):\n    return f\"value {x!r:>{4}} and {f'{x}'}\" + rb'raw'\n"
    assert tasks.normalize_code(code) == ["def", "ID", "(", "ID", ")", ":", "return", "STR", "+", "STR"]


def test_structure_still_counts() -> None:
    assert tasks.normalize_code("def f(x):\n    return x + 1\n") != tasks.normalize_code("def f(x):\n    return x - 1\n")
    assert tasks.normalize_code("def f(x):\n    return len(x)\n") != tasks.normalize_code("def f(x):\n    return sum(x)\n")


@pytest.mark.parametrize("text", ["def f(:\n", "def f():\n    return 'unterminated\n", "\x00", "", "))))"])
def test_normalize_code_never_raises(text: str) -> None:
    assert isinstance(tasks.normalize_code(text), list)


# ---- find_twins ---------------------------------------------------------------------------------


def test_a_renamed_twin_across_the_split_is_found() -> None:
    held, dev = ids_on_both_sides()
    original = _clamp(id=held)
    suite = suite_of([original, clamp_twin(dev), dataclasses.replace(VOWELS, id=f"{dev}-vowels")])
    assert tasks.split_of(original, 50) == HELDOUT and tasks.split_of(clamp_twin(dev), 50) == DEV
    twins = tasks.find_twins(suite)
    assert twins == [(dev, held, 1.0)]


def test_an_unrelated_task_is_not_a_twin() -> None:
    held, dev = ids_on_both_sides()
    suite = suite_of([_clamp(id=held), dataclasses.replace(VOWELS, id=dev)])
    assert tasks.find_twins(suite) == []
    everything = tasks.find_twins(suite, threshold=0.0)
    assert [(a, b) for a, b, _ in everything] == [(dev, held)]
    assert 0.0 <= everything[0][2] < tasks.TWIN_THRESHOLD


def test_twins_in_one_group_stay_together_and_are_not_reported() -> None:
    held, dev = ids_on_both_sides()
    grouped = [_clamp(id=held, group="clamp"), dataclasses.replace(clamp_twin(dev), group="clamp")]
    assert len({tasks.split_of(t, 50) for t in grouped}) == 1
    assert tasks.find_twins(suite_of(grouped), threshold=0.0) == []


def test_twins_are_sorted_by_similarity() -> None:
    (held,) = ids_in(HELDOUT, 1)
    exact_id, near_id = ids_in(DEV, 2)
    near = dataclasses.replace(clamp_twin(near_id), buggy_code="def limit(v, a, b):\n    return max(a, min(v, a)) + 0\n")
    found = tasks.find_twins(suite_of([_clamp(id=held), near, clamp_twin(exact_id)]), threshold=0.5)
    assert [(pair[0], pair[1]) for pair in found] == [(exact_id, held), (near_id, held)]
    assert found[0][2] == 1.0 > found[1][2] >= 0.5
    assert all(isinstance(pair[2], float) for pair in found)


def test_the_tiny_suite_has_no_twins(tiny_tasks: list[Task]) -> None:
    assert tasks.find_twins(suite_of(tiny_tasks)) == []


def test_gzipped_suite_round_trip_and_errors(tmp_path: Path) -> None:
    import gzip

    plain_dir = write_suite(tmp_path / "plain", TINY_TASKS)
    plain = tasks.load_suite(plain_dir)

    gz_dir = tmp_path / "gz"
    gz_dir.mkdir()
    (gz_dir / "suite.json").write_bytes((plain_dir / "suite.json").read_bytes())
    (gz_dir / "tasks.jsonl.gz").write_bytes(gzip.compress((plain_dir / "tasks.jsonl").read_bytes(), mtime=0))
    gz_suite = tasks.load_suite(gz_dir)
    assert gz_suite.sha256 == plain.sha256 and gz_suite.tasks == plain.tasks

    # Both tasks.jsonl and tasks.jsonl.gz together must be refused.
    (gz_dir / "tasks.jsonl").write_bytes((plain_dir / "tasks.jsonl").read_bytes())
    with pytest.raises(tasks.SuiteError, match="both"):
        tasks.load_suite(gz_dir)
    (gz_dir / "tasks.jsonl").unlink()

    (gz_dir / "tasks.jsonl.gz").write_bytes(b"corrupt")
    with pytest.raises(tasks.SuiteError, match="not valid gzip"):
        tasks.load_suite(gz_dir)

    (gz_dir / "tasks.jsonl.gz").write_bytes(gzip.compress(b"\xff\xfe"))
    with pytest.raises(tasks.SuiteError, match="not UTF-8"):
        tasks.load_suite(gz_dir)


def test_review_and_patch_suites_structure_and_twins(tmp_path: Path) -> None:
    held, dev = ids_on_both_sides()
    rev_dir = write_suite(
        tmp_path / "rev",
        [
            {"id": held, "user_prompt": "Review fix for auth token check.", "label": True, "patch_files": ["auth.py"]},
            {"id": dev, "user_prompt": "Review fix for auth token check!", "label": False, "patch_files": ["auth.py"]},
        ],
        kind="review",
    )
    rev = tasks.load_suite(rev_dir)
    assert rev.kind == "review" and tasks.check_structure(rev) == []
    assert rev.tasks[0].to_json()["label"] is True
    assert len(tasks.find_twins(rev, threshold=0.8)) == 1

    # Non-boolean label refused
    bad_rev = write_suite(tmp_path / "bad-rev", [{"id": "r1", "user_prompt": "hi", "label": 1}], kind="review")
    with pytest.raises(tasks.SuiteError, match="true or false"):
        tasks.load_suite(bad_rev)

    pat_dir = write_suite(
        tmp_path / "pat",
        [
            {
                "id": "pat-1",
                "user_prompt": "Fix return value in src/mod.py",
                "files": {"src/mod.py": "def f():\n    return 1\n", "drop.py": "old\n"},
                "gold_files": {"src/mod.py": "def f():\n    return 2\n", "drop.py": None},
            }
        ],
        kind="patch",
    )
    pat = tasks.load_suite(pat_dir)
    assert pat.kind == "patch" and tasks.check_structure(pat) == []
    assert set(pat.tasks) == {pat.tasks[0]}  # Task remains hashable with dict fields
    assert pat.tasks[0].to_json()["gold_files"]["drop.py"] is None

    noop_pat = write_suite(
        tmp_path / "noop-pat",
        [{"id": "p1", "user_prompt": "Noop", "files": {"a.py": "x\n"}, "gold_files": {"a.py": "x\n"}}],
        kind="patch",
    )
    assert any("does not differ" in p for p in tasks.check_structure(tasks.load_suite(noop_pat)))

    bad_path_pat = write_suite(
        tmp_path / "bad-path",
        [{"id": "p2", "user_prompt": "Traversal", "files": {"../a.py": "x\n"}, "gold_files": {"../a.py": "y\n"}}],
        kind="patch",
    )
    assert any("relative paths" in p for p in tasks.check_structure(tasks.load_suite(bad_path_pat)))
