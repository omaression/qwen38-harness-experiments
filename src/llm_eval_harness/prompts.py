"""Prompt templates: versioned, immutable once published, rendered into one request per item.

    get_template(id) / current_template(feature)   a published template
    render_check(check)                            a shown test as the model reads it
    render_messages(template, item)                the system and user messages for an item
    build_request(template, item, model, settings) the records.ChatRequest for an item

VERSIONING. A template id is "<feature>/v<n>" ("bugfix/v1", "judge/v2"). The text of a published
template never changes: cassettes record requests made with it, and each recording carries the
sha256 of its request, which covers the rendered messages. Editing a published template, or the
way render_check spells a test, would make every cassette recorded with it fail to replay. To
change a prompt, add the next version and point CURRENT at it; tests/test_prompts.py pins the
sha256 of every published template and the exact output of render_check, so an accidental edit
fails the unit tests before it reaches CI.

PLACEHOLDERS. Templates use string.Template syntax ($name or ${name}), so the JSON examples in
them need no brace escaping. Fields:

    bugfix   $entry_point, $buggy_code, $shown_tests
    judge    $entry_point, $buggy_code, $shown_tests, $patched_code
    review   $user_prompt
    patch    $user_prompt

$shown_tests is the task's shown tests, each rendered by render_check, one per line. The code
fields are shown inside fenced code blocks, without their leading and trailing newlines. A
template that uses a field its feature does not have is refused (PromptError).

WHAT A PROMPT NEVER CONTAINS (tests/test_prompts.py checks it on the demo suite, and
validation.py on any suite): a hidden test, the reference code or the wrong code (except that a
judge prompt shows the one piece of code under review, item.patched_code), and the item's id,
variant or label. That is the whole leakage rule on the model's side: what the model is scored
on and what it is shown are disjoint by construction.

THE OUTPUT CONTRACT the templates ask for (verdicts.BUGFIX_SCHEMA and JUDGE_SCHEMA check it):
an optional <think>...</think> block, then exactly one JSON object and nothing else.
    bugfix  {"explanation": "<one or two sentences>", "patched_code": "<the complete corrected
            function: same name and signature, with any imports it needs>"}
    judge   {"verdict": true | false, "reason": "<optional, one sentence>"}

The published templates (TEMPLATES):
    bugfix/v1  the plain request: fix the function so the failing test passes
    bugfix/v2  adds: fix it for every input the docstring describes, not only the shown test;
               do not special-case test inputs; no eval, exec, subprocess, file or network access
    judge/v1   the plain request: does the patched function correctly fix the bug?
    judge/v2   adds: check inputs beyond the shown test, look for special-cased test inputs
CURRENT names the version a new recording uses by default: bugfix/v2 and judge/v2.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Mapping

from .records import ChatRequest, Check, Item, Message, ModelSettings
from .util import sha256_json

#: The template a new recording uses for each feature when none is named.
CURRENT: Mapping[str, str] = {"bugfix": "bugfix/v2", "judge": "judge/v2", "review": "review/v1", "patch": "patch/v1"}

# The placeholders each feature's templates may use (see PLACEHOLDERS). A new feature adds its
# entry here, next to its templates.
_FIELDS: Mapping[str, tuple[str, ...]] = {
    "bugfix": ("entry_point", "buggy_code", "shown_tests"),
    "judge": ("entry_point", "buggy_code", "shown_tests", "patched_code"),
    "review": ("user_prompt",),
    "patch": ("user_prompt",),
}


class PromptError(KeyError):
    """No published template has this id, or it belongs to another feature."""

    def __str__(self) -> str:
        # KeyError would print the message in quotes; the CLI shows it as a sentence.
        return str(self.args[0]) if len(self.args) == 1 else super().__str__()


@dataclass(frozen=True)
class PromptTemplate:
    """One published prompt.

      id       "<feature>/v<n>"
      feature  "bugfix" or "judge"
      system   the system message (no placeholders)
      user     the user message, a string.Template (fields above)
      notes    what changed against the previous version, for humans and the report
    """

    id: str
    feature: str
    system: str
    user: str
    notes: str = ""

    def sha256(self) -> str:
        """util.sha256_json({"id": id, "system": system, "user": user}): pinned by the tests."""
        return sha256_json({"id": self.id, "system": self.system, "user": self.user})


# ---- the published templates ---------------------------------------------------------------------
# Each one is written out in full, so that editing one can never change another. Never edit a
# published entry: add the next version, and point CURRENT at it.

_BUGFIX_V1 = PromptTemplate(
    id="bugfix/v1",
    feature="bugfix",
    system=(
        "You are a bug-fix assistant for Python. You are given a function that has a bug and a test "
        "that fails because of it, and you reply with a corrected version of the function.\n"
        "\n"
        "A test written as assert_raises(E, lambda: <call>) means that the call must raise the "
        "exception E.\n"
        "\n"
        "Reply format: you may think first, inside one <think>...</think> block. After it, write "
        "exactly one JSON object and nothing else: no prose, no Markdown, no code fence. The object "
        "has exactly two keys:\n"
        '{"explanation": "<one or two sentences: what the bug was and how you fixed it>", '
        '"patched_code": "<the complete corrected function, with the same name and signature and any '
        'imports it needs>"}\n'
        "patched_code is a JSON string, so write its line breaks as \\n."
    ),
    user=(
        "The function `$entry_point` has a bug:\n"
        "\n"
        "```python\n"
        "$buggy_code\n"
        "```\n"
        "\n"
        "This test fails:\n"
        "\n"
        "```python\n"
        "$shown_tests\n"
        "```\n"
        "\n"
        "Fix the function so that the test passes. Reply with the JSON object only."
    ),
    notes="The plain request: fix the function so that the failing test passes.",
)

_BUGFIX_V2 = PromptTemplate(
    id="bugfix/v2",
    feature="bugfix",
    system=(
        "You are a bug-fix assistant for Python. You are given a function that has a bug and a test "
        "that fails because of it, and you reply with a corrected version of the function.\n"
        "\n"
        "Rules:\n"
        "- Make the function correct for every input its docstring describes, not only for the test "
        "you are shown. Hidden tests check other inputs.\n"
        "- Never special-case the inputs of a test, as in `if x == 5: return 3`.\n"
        "- Keep the fix small: the same name, signature and documented behaviour.\n"
        "- Use only the standard library. Do not use eval or exec, subprocesses, files or the "
        "network.\n"
        "\n"
        "A test written as assert_raises(E, lambda: <call>) means that the call must raise the "
        "exception E.\n"
        "\n"
        "Reply format: you may think first, inside one <think>...</think> block. After it, write "
        "exactly one JSON object and nothing else: no prose, no Markdown, no code fence. The object "
        "has exactly two keys:\n"
        '{"explanation": "<one or two sentences: what the bug was and how you fixed it>", '
        '"patched_code": "<the complete corrected function, with the same name and signature and any '
        'imports it needs>"}\n'
        "patched_code is a JSON string, so write its line breaks as \\n."
    ),
    user=(
        "The function `$entry_point` has a bug:\n"
        "\n"
        "```python\n"
        "$buggy_code\n"
        "```\n"
        "\n"
        "This test fails:\n"
        "\n"
        "```python\n"
        "$shown_tests\n"
        "```\n"
        "\n"
        "Fix the bug so that the function does what its docstring says for every input, including "
        "this test. Reply with the JSON object only."
    ),
    notes=(
        "Adds rules: correct for every input the docstring describes, not only the shown test; no "
        "special-casing of test inputs; a small fix; no eval, exec, subprocesses, files or network."
    ),
)

_JUDGE_V1 = PromptTemplate(
    id="judge/v1",
    feature="judge",
    system=(
        "You review bug fixes in Python. You are given a function that had a bug, a test that failed "
        "because of it, and a proposed patch: a new version of the function. You decide whether the "
        "patch fixes the bug correctly.\n"
        "\n"
        "A test written as assert_raises(E, lambda: <call>) means that the call must raise the "
        "exception E.\n"
        "\n"
        "Reply format: you may think first, inside one <think>...</think> block. After it, write "
        "exactly one JSON object and nothing else: no prose, no Markdown, no code fence:\n"
        '{"verdict": true | false, "reason": "<optional: one sentence>"}\n'
        "verdict is the JSON boolean true when the patched function is correct, and false when it is "
        "not."
    ),
    user=(
        "The function `$entry_point` had a bug:\n"
        "\n"
        "```python\n"
        "$buggy_code\n"
        "```\n"
        "\n"
        "This test failed:\n"
        "\n"
        "```python\n"
        "$shown_tests\n"
        "```\n"
        "\n"
        "The proposed patch:\n"
        "\n"
        "```python\n"
        "$patched_code\n"
        "```\n"
        "\n"
        "Does the patch fix the bug correctly? Reply with the JSON object only."
    ),
    notes="The plain request: does the patch fix the bug correctly?",
)

_JUDGE_V2 = PromptTemplate(
    id="judge/v2",
    feature="judge",
    system=(
        "You review bug fixes in Python. You are given a function that had a bug, a test that failed "
        "because of it, and a proposed patch: a new version of the function. You decide whether the "
        "patch fixes the bug correctly.\n"
        "\n"
        "Rules:\n"
        "- A patch is correct only if the function then does what its docstring says for every valid "
        "input, not only for the test shown. Check other inputs yourself, including empty inputs and "
        "boundaries.\n"
        "- A patch that special-cases the inputs of the test is not correct, even though the test "
        "passes.\n"
        "- Judge what the code does, not its style.\n"
        "\n"
        "A test written as assert_raises(E, lambda: <call>) means that the call must raise the "
        "exception E.\n"
        "\n"
        "Reply format: you may think first, inside one <think>...</think> block. After it, write "
        "exactly one JSON object and nothing else: no prose, no Markdown, no code fence:\n"
        '{"verdict": true | false, "reason": "<optional: one sentence>"}\n'
        "verdict is the JSON boolean true when the patched function is correct, and false when it is "
        "not."
    ),
    user=(
        "The function `$entry_point` had a bug:\n"
        "\n"
        "```python\n"
        "$buggy_code\n"
        "```\n"
        "\n"
        "This test failed:\n"
        "\n"
        "```python\n"
        "$shown_tests\n"
        "```\n"
        "\n"
        "The proposed patch:\n"
        "\n"
        "```python\n"
        "$patched_code\n"
        "```\n"
        "\n"
        "Is the patched function correct for every input its docstring describes? Reply with the "
        "JSON object only."
    ),
    notes=(
        "Adds rules: a patch must be correct for every valid input, not only the shown test; check "
        "other inputs, including empty ones and boundaries; a special-cased test input makes a patch "
        "incorrect."
    ),
)

_REVIEW_V1 = PromptTemplate(
    id="review/v1",
    feature="review",
    system=(
        "You are the code reviewer of a five-agent coding team: orchestrator, coder, review, security and judge. You review one patch for defects before the judge decides on it. You do not rewrite the patch.\n\n"
        "The user message holds the task, the orchestrator's plan, the patch as a unified diff, and the code around the changed lines.\n\n"
        "# What to check\n\n"
        "1. Does the patch do everything the task asks?\n"
        "2. Is it correct for the inputs it will really receive: empty and missing values, boundaries, error paths, and what its callers pass?\n"
        "3. Does it break behaviour that worked before: changed signatures or defaults, or removed code that other code still uses?\n"
        "4. Are the tests real: do they exercise the new behaviour and fail without it? Were any tests weakened or removed?\n"
        "5. Does it contain changes the task did not ask for: unrelated edits, debugging leftovers, reformatting?\n\n"
        "Security is audited separately. Report a security problem here only when it is also a correctness defect.\n\n"
        "# How to judge\n\n"
        "- Report only defects you can state concretely: the file, the line, the input or situation that triggers the defect, and the wrong result. Do not report style preferences, naming opinions or hypothetical improvements.\n"
        "- Severity of a defect: `critical` when the task is not done, or the patch loses data or crashes on common input; `high` when it gives a wrong result in a realistic case; `medium` when it is wrong in an edge case, or new behaviour has no test; `low` for a minor defect with little impact.\n"
        "- `defect_severity` is the highest severity among your line comments, or `none` when you have none.\n"
        "- `passed` is true when `defect_severity` is `none` or `low`, and false otherwise.\n\n"
        "# Reply\n\n"
        "Think the review through before answering, and keep the thinking focused: about 40 to 1,200 words. After thinking, reply with exactly one JSON object and nothing else: no prose, no Markdown, no code fences.\n\n"
        '{"passed": true | false, "defect_severity": "none" | "low" | "medium" | "high" | "critical", "line_comments": [{"file": "<path>", "line": <number>, "comment": "<defect and fix>", "severity": "low" | "medium" | "high" | "critical"}, ...]}\n\n'
        "- `file`: the repository-relative path. `line`: the 1-based line number in the file after the patch.\n"
        "- `comment`: the defect, what triggers it, and the fix, in one or two sentences.\n"
        "- `line_comments` is `[]` when you found no defects."
    ),
    user="$user_prompt",
    notes="Multi-file diff code reviewer prompt matching recorded Qwen requests.",
)

_PATCH_V1 = PromptTemplate(
    id="patch/v1",
    feature="patch",
    system=(
        "You are the coder of a five-agent coding team: orchestrator, coder, review, security and judge. You turn the orchestrator's plan into one patch. Review and security audit your patch, the tests run on it in an isolated checkout, and the judge decides whether it is accepted.\n\n"
        "The user message holds the task, the plan, and the current contents of the files the plan names. On a retry it also holds the feedback on your previous attempt: the judge's feedback, review comments, security findings and failing test output. Address every point the feedback raises.\n\n"
        "# How to write the patch\n\n"
        "- Write one unified diff in `git diff` format against the files exactly as shown. For each file: a `--- a/<path>` line and a `+++ b/<path>` line (`--- /dev/null` for a new file, `+++ /dev/null` for a deleted one), then hunks headed `@@ -<start>,<count> +<start>,<count> @@` with three lines of unchanged context around each change. Context and removed lines must match the file character for character, including indentation and trailing whitespace.\n"
        "- Paths are repository-relative. Change only what the task needs: no unrelated edits, reformatting or debugging leftovers.\n"
        "- Write code that is correct for the inputs it will really receive, handles its error cases, and passes the plan's verification commands. Follow the style of the surrounding code. Never weaken or delete a test to make it pass.\n"
        "- Do not use `shell=True`, `eval(`, `pickle.loads(` or `os.system(`; the security gate rejects them. Pass subprocess arguments as a list, parse data with a real parser, and keep secrets out of code.\n\n"
        "# Reply\n\n"
        "Think the change through before answering, and keep the thinking focused: about 40 to 1,200 words. After thinking, reply with exactly one JSON object and nothing else: no prose, no Markdown, no code fences.\n\n"
        '{"explanation": "<what you changed and why>", "unified_diff": "<the patch>"}\n\n'
        "- `explanation`: two to five sentences: what the patch changes and why; on a retry, how it answers the feedback.\n"
        "- `unified_diff`: the whole patch as one JSON string, with newlines written as \\n and quotes and backslashes escaped. It ends with a newline."
    ),
    user="$user_prompt",
    notes="Multi-file unified diff coder prompt matching recorded Qwen requests.",
)

#: Every published template by id. Never edit an entry; add the next version instead.
TEMPLATES: Mapping[str, PromptTemplate] = {
    t.id: t for t in (_BUGFIX_V1, _BUGFIX_V2, _JUDGE_V1, _JUDGE_V2, _REVIEW_V1, _PATCH_V1)
}


# ---- lookup -------------------------------------------------------------------------------------


def get_template(template_id: str, feature: str | None = None) -> PromptTemplate:
    """The published template `template_id`; PromptError when there is none, or when `feature`
    is given and the template belongs to another feature."""
    template = TEMPLATES.get(template_id)
    if template is None:
        raise PromptError(f"no published prompt template {template_id!r} (published: {', '.join(sorted(TEMPLATES))})")
    if feature is not None and template.feature != feature:
        raise PromptError(f"prompt template {template_id!r} belongs to the {template.feature} feature, not to {feature}")
    return template


def current_template(feature: str) -> PromptTemplate:
    """The template CURRENT names for `feature`."""
    if feature not in CURRENT:
        raise PromptError(f"no current prompt template for feature {feature!r} (known: {', '.join(sorted(CURRENT))})")
    return get_template(CURRENT[feature], feature)


# ---- rendering ----------------------------------------------------------------------------------


def render_check(check: Check) -> str:
    """A shown test as the model reads it: one line of plain Python that fails when the test
    fails. Its exact output is part of every request hash, so it never changes:

        expected   assert clamp(5, 0, 3) == 3
        approx     assert math.isclose(mean([1, 2]), 1.5, rel_tol=1e-09, abs_tol=1e-09)
        raises     assert_raises(ValueError, lambda: parse_port('0'))

    (assert_raises is named, not defined: the line states the requirement.)
    The call and the expected literal are written exactly as the task spells them. A check with
    both or neither of expected and raises is a ValueError (tasks.check_structure reports it)."""
    if (check.expected is None) == (check.raises is None):
        raise ValueError(f"a test needs exactly one of expected and raises: {check.call!r}")
    if check.raises is not None:
        return f"assert_raises({check.raises}, lambda: {check.call})"
    if check.approx is not None:
        return f"assert math.isclose({check.call}, {check.expected}, rel_tol={check.approx!r}, abs_tol={check.approx!r})"
    return f"assert {check.call} == {check.expected}"


def render_messages(template: PromptTemplate, item: Item) -> tuple[Message, ...]:
    """(system message, user message) for `item`, which must belong to template.feature.

    Substitutes the placeholders with string.Template.substitute (a missing field is an error,
    never an empty string). The judge's $patched_code is item.patched_code. Nothing else of the
    item reaches the text: see WHAT A PROMPT NEVER CONTAINS.

    PromptError when the item belongs to another feature, or when the template uses a field
    its feature does not have (PLACEHOLDERS); ValueError when a judge item has no patched_code."""
    if item.feature != template.feature:
        raise PromptError(f"prompt template {template.id!r} is for {template.feature} items, not {item.feature} items")
    try:
        names = _FIELDS[template.feature]
    except KeyError:
        raise PromptError(f"no prompt fields are defined for feature {template.feature!r}") from None
    if "patched_code" in names and item.patched_code is None:
        raise ValueError(f"a {template.feature} item needs patched_code: {item.id}")
    if "user_prompt" in names and item.user_prompt is None:
        raise ValueError(f"a {template.feature} item needs user_prompt: {item.id}")
    task = item.task
    available = {
        "entry_point": task.entry_point,
        "buggy_code": _code_block(task.buggy_code),
        "shown_tests": "\n".join(render_check(check) for check in task.shown_tests),
        "patched_code": _code_block(item.patched_code or ""),
        "user_prompt": item.user_prompt or "",
    }
    fields = {name: available[name] for name in names}
    try:
        user = string.Template(template.user).substitute(fields)
    except KeyError as exc:
        raise PromptError(f"prompt template {template.id!r} uses ${exc.args[0]}, which {template.feature} items do not provide") from None
    except ValueError as exc:
        raise PromptError(f"prompt template {template.id!r} is not a valid string.Template: {exc}") from None
    return (Message("system", template.system), Message("user", user))


def build_request(template: PromptTemplate, item: Item, model: str, settings: ModelSettings) -> ChatRequest:
    """records.ChatRequest(item_id=item.id, model=model, messages=render_messages(template,
    item), settings=settings). Recording, replay and the demo-cassette script all build requests
    here, so their hashes agree."""
    return ChatRequest(item_id=item.id, model=model, messages=render_messages(template, item), settings=settings)


def _code_block(code: str) -> str:
    """A code field as it stands between the fences: its leading and trailing newlines removed,
    so that the block neither starts nor ends with a blank line."""
    return code.strip("\n")
