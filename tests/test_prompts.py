"""prompts.py: the published templates (pinned: they never change), render_check, rendering, and
the leakage rule checked on every demo task under every template. Items are built directly with
records.Item, so these tests do not depend on features.py."""

from __future__ import annotations

import dataclasses
import json
import re
import string
from pathlib import Path

import pytest

from conftest import TINY_TASKS, task_from_dict
from llm_eval_harness import prompts, tasks
from llm_eval_harness.records import ChatRequest, Check, Item, Message, ModelSettings, Task

ROOT = Path(__file__).resolve().parents[1]

#: The sha256 of every published template. A published template never changes: cassettes hold
#: answers to requests made with it. If this test fails, restore the template and publish the
#: change as the next version (bugfix/v3, judge/v3) instead, then point prompts.CURRENT at it.
PINNED = {
    "bugfix/v1": "df15600dd36bd6c9c6f1275c5dfbf1e8b14b2ab5dc20cc49bb15df907871e032",
    "bugfix/v2": "fe6987aaee2f36c0e53a2fe9f0371c759f06ac53c61437fd5a9f70ce8b08c03a",
    "judge/v1": "6f9cafa596490c20b250f76260101ecc6b1434ca7760a170a9967f32ff304de2",
    "judge/v2": "f0991b81e96f73d62fb4330cfcd560644ca8063adb01e50743908ca72585e039",
    "review/v1": "e460731827ffd4910378cb3b044f1f0c86b784f6a8048b76e6021c53fe008076",
    "patch/v1": "98962798e5f54c42700ee2546582d54ea9f79da4b9ae6865d95db4da4a241ff9",
}
FIELDS = {
    "bugfix": {"entry_point", "buggy_code", "shown_tests"},
    "judge": {"entry_point", "buggy_code", "shown_tests", "patched_code"},
    "review": {"user_prompt"},
    "patch": {"user_prompt"},
}
SENTINEL = "@@PATCHED-CODE@@"

CLAMP = task_from_dict(TINY_TASKS[0])
PORT = task_from_dict(TINY_TASKS[3])


# ---- helpers ------------------------------------------------------------------------------------


def placeholders(text: str) -> set[str]:
    """The string.Template fields a text uses (Template.get_identifiers is Python 3.11+)."""
    found = set()
    for match in string.Template.pattern.finditer(text):
        assert match.group("invalid") is None, f"invalid placeholder at offset {match.start()}"
        name = match.group("named") or match.group("braced")
        if name:
            found.add(name)
    return found


def bugfix_item(task: Task, **changes: object) -> Item:
    item = Item(id=f"bugfix:{task.id}", feature="bugfix", task=task, split="heldout", cluster=task.group or task.id)
    return dataclasses.replace(item, **changes)


def judge_item(task: Task, variant: str, **changes: object) -> Item:
    code = {"reference": task.reference_code, "wrong": task.wrong_code}.get(variant, task.reference_code)
    item = Item(id=f"judge:{task.id}:{variant}", feature="judge", task=task, split="heldout", cluster=task.group or task.id,
                variant=variant, label=variant == "reference", patched_code=code)
    return dataclasses.replace(item, **changes)


def review_item(task: Task, variant: str = "reference", **changes: object) -> Item:
    item = Item(id=f"review:{task.id}:{variant}", feature="review", task=task, split="heldout", cluster=task.group or task.id,
                variant=variant, label=variant == "reference", user_prompt=f"Review `{task.entry_point}`")
    return dataclasses.replace(item, **changes)


def patch_item(task: Task, **changes: object) -> Item:
    item = Item(id=f"patch:{task.id}", feature="patch", task=task, split="heldout", cluster=task.group or task.id,
                user_prompt=f"Patch `{task.entry_point}`:\n```python\n{task.buggy_code}\n```")
    return dataclasses.replace(item, **changes)


def text_of(messages: tuple[Message, ...]) -> str:
    return "\n".join(m.content for m in messages)


DEMO = tasks.load_suite(ROOT / "tasks" / "demo")
BUGFIX_TEMPLATES = [t for t in prompts.TEMPLATES.values() if t.feature == "bugfix"]
JUDGE_TEMPLATES = [t for t in prompts.TEMPLATES.values() if t.feature == "judge"]


# ---- the published templates --------------------------------------------------------------------


def test_published_templates_are_pinned() -> None:
    assert {tid: t.sha256() for tid, t in prompts.TEMPLATES.items()} == PINNED, (
        "a published prompt template changed; restore it and add the next version instead")


def test_template_sha256_is_the_documented_hash() -> None:
    from llm_eval_harness.util import sha256_json

    for template in prompts.TEMPLATES.values():
        assert template.sha256() == sha256_json({"id": template.id, "system": template.system, "user": template.user})


@pytest.mark.parametrize("template", list(prompts.TEMPLATES.values()), ids=list(prompts.TEMPLATES))
def test_template_ids_fields_and_contract(template: prompts.PromptTemplate) -> None:
    assert re.fullmatch(r"(bugfix|judge|review|patch)/v[1-9][0-9]*", template.id)
    assert prompts.TEMPLATES[template.id] is template
    assert template.id.split("/")[0] == template.feature
    assert placeholders(template.user) == FIELDS[template.feature]
    assert "$" not in template.system, "the system message is not a template"
    assert "exactly one JSON object and nothing else" in template.system
    if template.feature in ("bugfix", "judge"):
        assert "<think>...</think>" in template.system
        max_words = 250
    else:
        assert "1,200 words" in template.system
        max_words = 450
    keys = {
        "bugfix": ("explanation", "patched_code"),
        "judge": ("verdict", "reason"),
        "review": ("passed", "defect_severity", "line_comments"),
        "patch": ("explanation", "unified_diff"),
    }[template.feature]
    assert all(f'"{key}"' in template.system for key in keys)
    assert template.notes
    assert len(template.system.split()) + len(template.user.split()) <= max_words


def test_v2_adds_the_rules() -> None:
    v1, v2 = prompts.TEMPLATES["bugfix/v1"], prompts.TEMPLATES["bugfix/v2"]
    for phrase in ["every input its docstring describes", "special-case", "eval", "exec", "subprocesses", "network"]:
        assert phrase in v2.system and phrase not in v1.system, phrase
    j1, j2 = prompts.TEMPLATES["judge/v1"], prompts.TEMPLATES["judge/v2"]
    for phrase in ["every valid input", "special-cases", "empty inputs and boundaries"]:
        assert phrase in j2.system and phrase not in j1.system, phrase


def test_templates_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        prompts.TEMPLATES["bugfix/v1"].user = "changed"  # type: ignore[misc]


# ---- lookup -------------------------------------------------------------------------------------


def test_current_templates() -> None:
    assert dict(prompts.CURRENT) == {"bugfix": "bugfix/v2", "judge": "judge/v2", "review": "review/v1", "patch": "patch/v1"}
    assert prompts.current_template("bugfix") is prompts.TEMPLATES["bugfix/v2"]
    assert prompts.current_template("judge") is prompts.TEMPLATES["judge/v2"]
    assert prompts.current_template("review") is prompts.TEMPLATES["review/v1"]
    assert prompts.current_template("patch") is prompts.TEMPLATES["patch/v1"]
    assert all(cid in prompts.TEMPLATES for cid in prompts.CURRENT.values())


def test_get_template() -> None:
    assert prompts.get_template("judge/v1") is prompts.TEMPLATES["judge/v1"]
    assert prompts.get_template("judge/v1", "judge") is prompts.TEMPLATES["judge/v1"]
    with pytest.raises(prompts.PromptError, match=r"no published prompt template 'bugfix/v9' \(published: bugfix/v1, bugfix/v2"):
        prompts.get_template("bugfix/v9")
    with pytest.raises(prompts.PromptError, match=r"belongs to the judge feature, not to bugfix"):
        prompts.get_template("judge/v2", "bugfix")
    with pytest.raises(prompts.PromptError, match=r"no current prompt template for feature 'summary'"):
        prompts.current_template("summary")


def test_prompt_error_is_a_key_error_with_a_plain_message() -> None:
    error = prompts.PromptError("no published prompt template 'x'")
    assert isinstance(error, KeyError)
    assert str(error) == "no published prompt template 'x'"


# ---- render_check -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "check, line",
    [
        (Check("clamp(5, 0, 3)", expected="3"), "assert clamp(5, 0, 3) == 3"),
        (Check("mean([1, 2])", expected="1.5", approx=1e-9), "assert math.isclose(mean([1, 2]), 1.5, rel_tol=1e-09, abs_tol=1e-09)"),
        (Check("parse_port('0')", raises="ValueError"), "assert_raises(ValueError, lambda: parse_port('0'))"),
        (Check("f(0.1)", expected="0.25", approx=0.001), "assert math.isclose(f(0.1), 0.25, rel_tol=0.001, abs_tol=0.001)"),
        (Check("first([])", expected="None"), "assert first([]) == None"),
        (Check("pair('a')", expected="('a', 1)"), "assert pair('a') == ('a', 1)"),
        (Check("strip('a\\tb')", expected="'a b'"), "assert strip('a\\tb') == 'a b'"),
    ],
)
def test_render_check_is_golden(check: Check, line: str) -> None:
    assert prompts.render_check(check) == line


@pytest.mark.parametrize("check", [Check("f(1)"), Check("f(1)", expected="1", raises="ValueError")])
def test_render_check_refuses_an_ill_formed_test(check: Check) -> None:
    with pytest.raises(ValueError):
        prompts.render_check(check)


# ---- rendering ----------------------------------------------------------------------------------

GOLDEN_BUGFIX_V1_USER = '''The function `clamp` has a bug:

```python
def clamp(x, lo, hi):
    """Return x limited to the closed interval [lo, hi]."""
    return max(lo, min(x, lo))
```

This test fails:

```python
assert clamp(5, 0, 3) == 3
```

Fix the function so that the test passes. Reply with the JSON object only.'''


def test_render_messages_bugfix_is_golden() -> None:
    system, user = prompts.render_messages(prompts.TEMPLATES["bugfix/v1"], bugfix_item(CLAMP))
    assert (system.role, user.role) == ("system", "user")
    assert system.content == prompts.TEMPLATES["bugfix/v1"].system
    assert user.content == GOLDEN_BUGFIX_V1_USER


def test_render_messages_judge_shows_the_patch() -> None:
    _, user = prompts.render_messages(prompts.TEMPLATES["judge/v1"], judge_item(PORT, "wrong"))
    assert "```python\n" + PORT.wrong_code.strip("\n") + "\n```" in user.content
    assert "```python\n" + PORT.buggy_code.strip("\n") + "\n```" in user.content
    assert "assert_raises(ValueError, lambda: parse_port('0'))" in user.content


def test_several_shown_tests_are_one_per_line() -> None:
    task = dataclasses.replace(CLAMP, shown_tests=(Check("clamp(5, 0, 3)", expected="3"), Check("clamp(9, 0, 3)", raises="OverflowError")))
    _, user = prompts.render_messages(prompts.TEMPLATES["bugfix/v2"], bugfix_item(task))
    assert "```python\nassert clamp(5, 0, 3) == 3\nassert_raises(OverflowError, lambda: clamp(9, 0, 3))\n```" in user.content


def test_code_blocks_drop_leading_and_trailing_newlines() -> None:
    task = dataclasses.replace(CLAMP, buggy_code="\n\n" + CLAMP.buggy_code + "\n\n")
    assert prompts.render_messages(prompts.TEMPLATES["bugfix/v1"], bugfix_item(task))[1].content == GOLDEN_BUGFIX_V1_USER


def test_rendering_refusals() -> None:
    with pytest.raises(prompts.PromptError, match=r"is for bugfix items, not judge items"):
        prompts.render_messages(prompts.TEMPLATES["bugfix/v1"], judge_item(CLAMP, "wrong"))
    with pytest.raises(prompts.PromptError, match=r"is for judge items, not bugfix items"):
        prompts.render_messages(prompts.TEMPLATES["judge/v1"], bugfix_item(CLAMP))
    with pytest.raises(ValueError, match="needs patched_code"):
        prompts.render_messages(prompts.TEMPLATES["judge/v2"], judge_item(CLAMP, "wrong", patched_code=None))
    with pytest.raises(ValueError, match="needs user_prompt"):
        prompts.render_messages(prompts.TEMPLATES["review/v1"], review_item(CLAMP, user_prompt=None))
    with pytest.raises(ValueError, match="needs user_prompt"):
        prompts.render_messages(prompts.TEMPLATES["patch/v1"], patch_item(CLAMP, user_prompt=None))
    sneaky = prompts.PromptTemplate("bugfix/v99", "bugfix", "s", "Fix $entry_point; the answer is $patched_code")
    with pytest.raises(prompts.PromptError, match=r"uses \$patched_code, which bugfix items do not provide"):
        prompts.render_messages(sneaky, bugfix_item(CLAMP, patched_code=CLAMP.reference_code))
    broken = prompts.PromptTemplate("bugfix/v98", "bugfix", "s", "costs $5 for $entry_point")
    with pytest.raises(prompts.PromptError, match="not a valid string.Template"):
        prompts.render_messages(broken, bugfix_item(CLAMP))
    unknown = prompts.PromptTemplate("summary/v1", "summary", "s", "$entry_point")
    with pytest.raises(prompts.PromptError, match="no prompt fields are defined for feature 'summary'"):
        prompts.render_messages(unknown, dataclasses.replace(bugfix_item(CLAMP), feature="summary"))


def test_a_dollar_sign_in_the_code_is_kept_as_is() -> None:
    task = dataclasses.replace(CLAMP, buggy_code=CLAMP.buggy_code.replace("[lo, hi].", "[lo, hi], costs $hi or ${x}."))
    _, user = prompts.render_messages(prompts.TEMPLATES["bugfix/v1"], bugfix_item(task))
    assert "costs $hi or ${x}." in user.content


def test_nothing_but_the_task_reaches_a_bugfix_prompt() -> None:
    plain = text_of(prompts.render_messages(prompts.TEMPLATES["bugfix/v2"], bugfix_item(CLAMP)))
    loaded = bugfix_item(CLAMP, id="bugfix:zq-item-zq", split="dev", cluster="zq-cluster-zq", variant="zq-variant-zq",
                         label=True, patched_code="zq-patch-zq")
    assert text_of(prompts.render_messages(prompts.TEMPLATES["bugfix/v2"], loaded)) == plain
    assert "zq" not in plain


def test_nothing_but_the_patch_reaches_a_judge_prompt() -> None:
    template = prompts.TEMPLATES["judge/v2"]
    base = text_of(prompts.render_messages(template, judge_item(CLAMP, "reference", patched_code=SENTINEL)))
    odd = judge_item(CLAMP, "zq-variant-zq", id="judge:t-clamp:zq-variant-zq", split="dev", cluster="zq-cluster-zq",
                     label=False, patched_code=SENTINEL)
    assert text_of(prompts.render_messages(template, odd)) == base
    assert "zq" not in base


# ---- the leakage rule on every demo task --------------------------------------------------------


def assert_no_hidden_call(text: str, task: Task) -> None:
    """No hidden call verbatim, nor with other spacing between its tokens."""
    for check in task.hidden_tests:
        assert check.call not in text, (task.id, check.call)
        assert not tasks._call_appears_in(check.call, text), (task.id, check.call)


def distinguishing_lines(code: str, buggy: str) -> set[str]:
    """The lines of a fix that the buggy code does not have: what would give the fix away."""
    return {line.strip() for line in code.splitlines()} - {line.strip() for line in buggy.splitlines()} - {""}


@pytest.mark.parametrize("template", BUGFIX_TEMPLATES, ids=lambda t: t.id)
def test_bugfix_prompts_of_the_demo_suite_leak_nothing(template: prompts.PromptTemplate) -> None:
    for task in DEMO.tasks:
        item = bugfix_item(task)
        text = text_of(prompts.render_messages(template, item))
        assert_no_hidden_call(text, task)
        prompt_lines = {line.strip() for line in text.splitlines()}
        for code in (task.reference_code, task.wrong_code):
            assert code.strip("\n") not in text, task.id
            leaked = distinguishing_lines(code, task.buggy_code) & prompt_lines
            assert not leaked, (task.id, leaked)
        assert item.id not in text
        assert task.buggy_code.strip("\n") in text and prompts.render_check(task.shown_tests[0]) in text


@pytest.mark.parametrize("template", JUDGE_TEMPLATES, ids=lambda t: t.id)
def test_judge_prompts_of_the_demo_suite_differ_only_in_the_patch(template: prompts.PromptTemplate) -> None:
    for task in DEMO.tasks:
        reference, wrong = judge_item(task, "reference"), judge_item(task, "wrong")
        ref_text = text_of(prompts.render_messages(template, reference))
        wrong_text = text_of(prompts.render_messages(template, wrong))
        frame = text_of(prompts.render_messages(template, judge_item(task, "reference", patched_code=SENTINEL)))
        assert frame.count(SENTINEL) == 1
        assert ref_text == frame.replace(SENTINEL, task.reference_code.strip("\n")), task.id
        assert wrong_text == frame.replace(SENTINEL, task.wrong_code.strip("\n")), task.id
        assert ref_text != wrong_text
        for item, text in ((reference, ref_text), (wrong, wrong_text)):
            assert_no_hidden_call(text, task)
            assert item.id not in text
        assert "reference" not in frame and "wrong" not in frame, task.id


def test_every_demo_task_renders_under_every_template() -> None:
    for template in prompts.TEMPLATES.values():
        for task in DEMO.tasks:
            if template.feature == "bugfix":
                item = bugfix_item(task)
            elif template.feature == "judge":
                item = judge_item(task, "wrong")
            elif template.feature == "review":
                item = review_item(task)
            else:
                item = patch_item(task)
            system, user = prompts.render_messages(template, item)
            # every placeholder was filled (the demo code itself holds no "$")
            assert f"`{task.entry_point}`" in user.content and "$" not in user.content
            assert system.content == template.system


# ---- build_request ------------------------------------------------------------------------------


def test_build_request_is_the_rendered_request() -> None:
    template, item = prompts.TEMPLATES["bugfix/v1"], bugfix_item(CLAMP)
    settings = ModelSettings(temperature=0.0, max_tokens=4096, seed=0)
    request = prompts.build_request(template, item, "demo-model", settings)
    assert request == ChatRequest(item_id="bugfix:t-clamp", model="demo-model", messages=prompts.render_messages(template, item), settings=settings)
    assert "bugfix:t-clamp" not in json.dumps(request.payload())


def test_request_hashes_are_pinned() -> None:
    # The whole chain a cassette depends on: template text, render_check, the code blocks and
    # the request body. If this changes, every recorded cassette stops replaying.
    bugfix = prompts.build_request(prompts.TEMPLATES["bugfix/v1"], bugfix_item(CLAMP), "demo-model", ModelSettings())
    assert bugfix.sha256() == "8acfcaf669574cc118a254d5662e3309e0271cd127d0a491949b1e4a9ceee733"
    judge = prompts.build_request(prompts.TEMPLATES["judge/v2"], judge_item(CLAMP, "wrong"), "demo-model",
                                  ModelSettings(temperature=0.0, max_tokens=2048, seed=0))
    assert judge.sha256() == "c25c79fa114f308d08cd4811bdfdb8483647f218879454a063cb6a909b44b94a"


def test_request_hash_depends_on_the_question_not_the_item_id() -> None:
    template, settings = prompts.TEMPLATES["bugfix/v2"], ModelSettings()
    first = prompts.build_request(template, bugfix_item(CLAMP), "m", settings)
    renamed = prompts.build_request(template, bugfix_item(CLAMP, id="bugfix:other"), "m", settings)
    assert first.sha256() == renamed.sha256()
    assert prompts.build_request(prompts.TEMPLATES["bugfix/v1"], bugfix_item(CLAMP), "m", settings).sha256() != first.sha256()
    assert prompts.build_request(template, bugfix_item(CLAMP), "other-model", settings).sha256() != first.sha256()
    assert prompts.build_request(template, bugfix_item(CLAMP), "m", ModelSettings(temperature=0.7)).sha256() != first.sha256()
    edited = dataclasses.replace(CLAMP, buggy_code=CLAMP.buggy_code.replace("closed", "closed, inclusive"))
    assert prompts.build_request(template, bugfix_item(edited), "m", settings).sha256() != first.sha256()
    hidden_changed = dataclasses.replace(CLAMP, hidden_tests=CLAMP.hidden_tests[:2])
    assert prompts.build_request(template, bugfix_item(hidden_changed), "m", settings).sha256() == first.sha256()
