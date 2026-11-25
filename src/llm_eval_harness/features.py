"""The features under evaluation: what their items are and how one reply is scored.

    get_feature(name) -> Feature        "bugfix", "judge", "review", or "patch"
    Feature.items(suite, split)         the items of a split, sorted by id
    Feature.score(item, response)       one reply -> records.ItemResult

To evaluate another LLM feature with this harness, implement the Feature protocol (items, a
schema, a score function and a primary metric in metrics.py) and register it in FEATURES; the
runner, the gate, the report and CI need no change. docs/DESIGN.md, "Adding a feature".

BUGFIX. One item per task: id "bugfix:<task id>", cluster the task's split key. Scoring:
  1. verdicts.check_think(content, reasoning, BUGFIX_THINK) and split_reply for the answer;
  2. verdicts.check_schema(answer, BUGFIX_SCHEMA, finish_reason=...);
  3. the code to scan: patched_code of the strict value, else of the lenient value, else none.
     sast.added_findings(task.buggy_code, code): a reply is accountable for the code it wrote
     even when its JSON was malformed;
  4. when the schema is valid and no block finding was added: verdicts.exec_verdict(task,
     patched_code). An infra_error there raises ScoringError: no verdict, and the evaluation
     stops (exit 1); it never scores the item as failed;
  5. correct = schema valid and no block finding and exec passed. ItemResult.detail holds the
     task's tags (metrics counts them per tag), the think and schema checks, the code's source
     ("strict", "lenient" or None), the findings (to_json) and the ExecVerdict (to_json).
JUDGE. Two items per task: "judge:<task id>:reference" (label True, the reference code) and
"judge:<task id>:wrong" (label False, the wrong code); cluster the task's split key. Scoring:
check_think, check_schema(answer, JUDGE_SCHEMA); predicted = value["verdict"] when valid, else
None; correct = predicted is the label. No code runs and no SAST applies.
REVIEW. One item per task on "review" suites ("review:<task id>"), or two items per task on
"function" suites ("review:<task id>:reference" / ":wrong"). Scoring checks think and
REVIEW_SCHEMA (including defect_severity enum and line_comments dicts); predicted = passed.
PATCH. One item per task on "patch" or "function" suites ("patch:<task id>"). Scoring checks
think, PATCH_SCHEMA, pure-Python unified diff application via diff_apply, non-noop edits,
Python syntax validity, and zero added block-class SAST findings.
All features: ItemResult.problems lists short reasons; latency, tokens and finish_reason are
copied from the response.
"""

from __future__ import annotations

import ast
import difflib
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from . import diff_apply, prompts, sast, tasks, verdicts
from .records import ChatResponse, Item, ItemResult, ModelSettings, Suite, Task
from .sandbox import SandboxUnavailable
from .verdicts import BUGFIX_SCHEMA, JUDGE_SCHEMA, PATCH_SCHEMA, REVIEW_SCHEMA, JsonSchema, SchemaCheck, ThinkCheck, ThinkPolicy

#: Thinking is optional for all features; when present it must be closed and at most 1,200 words.
BUGFIX_THINK = ThinkPolicy(required=False, min_words=1, max_words=1200)
JUDGE_THINK = ThinkPolicy(required=False, min_words=1, max_words=1200)
REVIEW_THINK = ThinkPolicy(required=False, min_words=1, max_words=1200)
PATCH_THINK = ThinkPolicy(required=False, min_words=1, max_words=1200)
REVIEW_SEVERITIES = frozenset({"none", "low", "medium", "high", "critical"})
#: At most this many failing tests and findings are spelt out in ItemResult.problems.
MAX_LISTED = 3


class ScoringError(RuntimeError):
    """A reply could not be scored for a reason that is not the model's (the sandbox did not
    start). The evaluation stops; nothing is counted."""


@runtime_checkable
class Feature(Protocol):
    """What the runner needs from a feature."""

    name: str
    primary_metric: str
    writes_code: bool
    schema: JsonSchema
    think_policy: ThinkPolicy
    default_settings: ModelSettings

    def items(self, suite: Suite, split: str) -> list[Item]:
        ...

    def score(self, item: Item, response: ChatResponse, *, timeout_s: float) -> ItemResult:
        ...


@dataclass(frozen=True)
class BugfixFeature:
    """The bug-fix assistant (see BUGFIX)."""

    name: str = "bugfix"
    primary_metric: str = "resolved_rate"
    writes_code: bool = True
    schema: JsonSchema = BUGFIX_SCHEMA
    think_policy: ThinkPolicy = BUGFIX_THINK
    default_settings: ModelSettings = ModelSettings(temperature=0.0, max_tokens=4096, seed=0)

    def items(self, suite: Suite, split: str) -> list[Item]:
        if suite.kind != "function":
            return []
        out = [
            Item(id=f"bugfix:{task.id}", feature=self.name, task=task, split=tasks.split_of(task, suite.heldout_pct), cluster=cluster_of(task))
            for task in tasks.tasks_in_split(suite, split)
        ]
        return sorted(out, key=lambda item: item.id)

    def score(self, item: Item, response: ChatResponse, *, timeout_s: float) -> ItemResult:
        task = item.task
        think, schema = _read_reply(response, self.schema, self.think_policy)

        # The code the reply wrote: from the strict object, else from the lenient parse. A reply
        # is accountable for the code it wrote even when its JSON was malformed.
        code, code_source = None, None
        if schema.valid and schema.value is not None:
            code, code_source = schema.value.get("patched_code"), "strict"
        elif isinstance(schema.lenient_value, dict) and isinstance(schema.lenient_value.get("patched_code"), str):
            code, code_source = schema.lenient_value["patched_code"], "lenient"
        findings = sast.added_findings(task.buggy_code, code) if isinstance(code, str) else []
        block = sast.blocking(findings)
        block_rules = tuple(sorted({f.rule for f in block}))
        flag_rules = tuple(sorted({f.rule for f in findings if f.severity == sast.FLAG}))

        verdict = None
        if schema.valid and isinstance(code, str) and not block:
            try:
                verdict = verdicts.exec_verdict(task, code, timeout_s=timeout_s)
            except SandboxUnavailable as exc:
                raise ScoringError(f"{item.id}: the sandbox is not available here: {exc}") from None
            if verdict.infra_error:
                raise ScoringError(f"{item.id}: the sandbox did not run: {verdict.infra_error}")
        correct = bool(schema.valid and not block and verdict is not None and verdict.passed)

        problems = _reply_problems(think, schema, self.think_policy)
        listed = findings[:MAX_LISTED]
        problems += [f"sast {f.severity}: {f.rule}: {f.snippet}" for f in listed]
        if len(findings) > len(listed):
            problems.append(f"sast: {len(findings) - len(listed)} more finding(s)")
        if verdict is None:
            if not schema.valid:
                problems.append("exec: not run (the answer is not schema-valid)")
            elif block:
                problems.append(f"exec: not run (block-class finding: {', '.join(block_rules)})")
        else:
            problems += _exec_problems(verdict)

        detail: dict[str, Any] = {
            "tags": list(task.tags),
            "think": _think_json(think),
            "schema": _schema_json(schema),
            "code_source": code_source,
            "sast": [f.to_json() for f in findings],
            "exec": None if verdict is None else verdict.to_json(),
        }
        return ItemResult(
            item_id=item.id,
            feature=self.name,
            task_id=task.id,
            cluster=item.cluster,
            split=item.split,
            correct=correct,
            schema_valid=schema.valid,
            lenient_valid=bool(schema.lenient_valid),
            think_present=think.present,
            think_valid=think.valid,
            think_words=think.words,
            finish_reason=response.finish_reason,
            latency_s=response.latency_s,
            prompt_tokens=_usage(response, "prompt_tokens"),
            completion_tokens=_usage(response, "completion_tokens"),
            shown_passed=None if verdict is None else verdict.shown_passed,
            hidden_passed=None if verdict is None else verdict.hidden_passed,
            sast_block=block_rules,
            sast_flag=flag_rules,
            problems=tuple(problems),
            detail=detail,
        )


@dataclass(frozen=True)
class JudgeFeature:
    """The patch-correctness judge (see JUDGE)."""

    name: str = "judge"
    primary_metric: str = "balanced_accuracy"
    writes_code: bool = False
    schema: JsonSchema = JUDGE_SCHEMA
    think_policy: ThinkPolicy = JUDGE_THINK
    default_settings: ModelSettings = ModelSettings(temperature=0.0, max_tokens=2048, seed=0)

    def items(self, suite: Suite, split: str) -> list[Item]:
        if suite.kind != "function":
            return []
        out: list[Item] = []
        for task in tasks.tasks_in_split(suite, split):
            common = {"feature": self.name, "task": task, "split": tasks.split_of(task, suite.heldout_pct), "cluster": cluster_of(task)}
            out.append(Item(id=f"judge:{task.id}:reference", variant="reference", label=True, patched_code=task.reference_code, **common))
            out.append(Item(id=f"judge:{task.id}:wrong", variant="wrong", label=False, patched_code=task.wrong_code, **common))
        return sorted(out, key=lambda item: item.id)

    def score(self, item: Item, response: ChatResponse, *, timeout_s: float) -> ItemResult:
        think, schema = _read_reply(response, self.schema, self.think_policy)
        predicted = None
        if schema.valid and schema.value is not None and type(schema.value.get("verdict")) is bool:
            predicted = schema.value["verdict"]
        correct = predicted is not None and predicted is item.label

        problems = _reply_problems(think, schema, self.think_policy)
        if predicted is not None and not correct:
            problems.append(f"verdict: {str(predicted).lower()} for the {item.variant} patch (label {str(item.label).lower()})")
        detail: dict[str, Any] = {
            "tags": list(item.task.tags),
            "think": _think_json(think),
            "schema": _schema_json(schema),
            "variant": item.variant,
        }
        return ItemResult(
            item_id=item.id,
            feature=self.name,
            task_id=item.task.id,
            cluster=item.cluster,
            split=item.split,
            correct=correct,
            schema_valid=schema.valid,
            lenient_valid=bool(schema.lenient_valid),
            think_present=think.present,
            think_valid=think.valid,
            think_words=think.words,
            finish_reason=response.finish_reason,
            latency_s=response.latency_s,
            prompt_tokens=_usage(response, "prompt_tokens"),
            completion_tokens=_usage(response, "completion_tokens"),
            label=item.label,
            predicted=predicted,
            problems=tuple(problems),
            detail=detail,
        )


def _function_review_prompt(task: Task, patched_code: str) -> str:
    """Render a function-suite task and proposed fix into a diff + context prompt for review/v1."""
    shown = "\n".join(prompts.render_check(check) for check in task.shown_tests)
    filename = f"{task.entry_point}.py"
    old_lines = task.buggy_code.strip("\n").splitlines(keepends=True)
    new_lines = patched_code.strip("\n").splitlines(keepends=True)
    if old_lines and not old_lines[-1].endswith("\n"):
        old_lines[-1] += "\n"
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"
    diff = "".join(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{filename}", tofile=f"b/{filename}"))
    numbered = "\n".join(f"{idx}: {line}" for idx, line in enumerate(patched_code.strip("\n").splitlines(), start=1))
    return (
        f"# Task\n\nFix `{task.entry_point}` so all inputs described by its contract pass.\n\n"
        f"Shown test(s):\n```python\n{shown}\n```\n\n"
        f"# Patch\n\n```diff\n{diff.rstrip()}\n```\n\n"
        f"# Code around the changed lines (`{filename}`)\n\n```python\n{numbered}\n```"
    )


def _review_field_problems(obj: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    severity = obj.get("defect_severity")
    if severity not in REVIEW_SEVERITIES:
        problems.append(f"'defect_severity' must be one of {', '.join(sorted(REVIEW_SEVERITIES))}, got {severity!r}")
    comments = obj.get("line_comments")
    if isinstance(comments, list) and not all(isinstance(comment, dict) for comment in comments):
        problems.append("'line_comments' entries must be JSON objects")
    return problems


@dataclass(frozen=True)
class ReviewFeature:
    """The diff code reviewer (see REVIEW)."""

    name: str = "review"
    primary_metric: str = "balanced_accuracy"
    writes_code: bool = False
    schema: JsonSchema = REVIEW_SCHEMA
    think_policy: ThinkPolicy = REVIEW_THINK
    default_settings: ModelSettings = ModelSettings(temperature=0.0, max_tokens=16384, seed=0)

    def items(self, suite: Suite, split: str) -> list[Item]:
        if suite.kind == "review":
            out = [
                Item(
                    id=f"review:{task.id}",
                    feature=self.name,
                    task=task,
                    split=tasks.split_of(task, suite.heldout_pct),
                    cluster=cluster_of(task),
                    label=task.label,
                    user_prompt=task.user_prompt,
                )
                for task in tasks.tasks_in_split(suite, split)
            ]
            return sorted(out, key=lambda item: item.id)
        if suite.kind == "function":
            out = []
            for task in tasks.tasks_in_split(suite, split):
                common = {"feature": self.name, "task": task, "split": tasks.split_of(task, suite.heldout_pct), "cluster": cluster_of(task)}
                out.append(
                    Item(
                        id=f"review:{task.id}:reference",
                        variant="reference",
                        label=True,
                        user_prompt=_function_review_prompt(task, task.reference_code),
                        **common,
                    )
                )
                out.append(
                    Item(
                        id=f"review:{task.id}:wrong",
                        variant="wrong",
                        label=False,
                        user_prompt=_function_review_prompt(task, task.wrong_code),
                        **common,
                    )
                )
            return sorted(out, key=lambda item: item.id)
        return []

    def score(self, item: Item, response: ChatResponse, *, timeout_s: float) -> ItemResult:
        think, schema = _read_reply(response, self.schema, self.think_policy)
        if schema.valid and schema.value is not None:
            extra = _review_field_problems(schema.value)
            if extra:
                schema.valid = False
                schema.value = None
                schema.problems.extend(extra)
        if schema.lenient_valid and schema.lenient_value is not None and _review_field_problems(schema.lenient_value):
            schema.lenient_valid = False

        predicted = None
        if schema.valid and schema.value is not None and type(schema.value.get("passed")) is bool:
            predicted = schema.value["passed"]
        correct = predicted is not None and predicted is item.label

        problems = _reply_problems(think, schema, self.think_policy)
        if predicted is not None and not correct:
            problems.append(f"verdict: {str(predicted).lower()} for review item (label {str(item.label).lower()})")
        detail: dict[str, Any] = {
            "tags": list(item.task.tags),
            "think": _think_json(think),
            "schema": _schema_json(schema),
            "defect_severity": schema.value.get("defect_severity") if schema.valid and schema.value is not None else None,
            "line_comments": len(schema.value.get("line_comments", [])) if schema.valid and schema.value is not None else None,
        }
        if item.variant is not None:
            detail["variant"] = item.variant
        return ItemResult(
            item_id=item.id,
            feature=self.name,
            task_id=item.task.id,
            cluster=item.cluster,
            split=item.split,
            correct=correct,
            schema_valid=schema.valid,
            lenient_valid=bool(schema.lenient_valid),
            think_present=think.present,
            think_valid=think.valid,
            think_words=think.words,
            finish_reason=response.finish_reason,
            latency_s=response.latency_s,
            prompt_tokens=_usage(response, "prompt_tokens"),
            completion_tokens=_usage(response, "completion_tokens"),
            label=item.label,
            predicted=predicted,
            problems=tuple(problems),
            detail=detail,
        )


def _function_patch_prompt(task: Task) -> tuple[str, dict[str, str], dict[str, str | None]]:
    filename = f"{task.entry_point}.py"
    shown = "\n".join(prompts.render_check(check) for check in task.shown_tests)
    buggy = task.buggy_code.strip("\n")
    user_prompt = (
        f"# Task\n\nFix the bug in `{filename}` so that `{task.entry_point}` matches its specification, "
        f"including:\n```python\n{shown}\n```\n\n"
        f"# Files\n\n## `{filename}`\n```python\n{buggy}\n```"
    )
    return user_prompt, {filename: task.buggy_code}, {filename: task.reference_code}


def _can_parse_py(code: str) -> tuple[bool, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(code)
        return True, ""
    except (SyntaxError, ValueError, RecursionError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _ast_equal(first: str, second: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.dump(ast.parse(first)) == ast.dump(ast.parse(second))
    except (SyntaxError, ValueError, RecursionError):
        return False


def _matches_gold(
    base_files: Mapping[str, str],
    gold_files: Mapping[str, str | None],
    updated: Mapping[str, str | None] | None,
) -> bool:
    if updated is None or not gold_files:
        return False
    expected = {**base_files, **gold_files}
    for path in set(expected) | set(updated):
        exp = expected.get(path)
        got = updated.get(path)
        if got == exp:
            continue
        if isinstance(exp, str) and isinstance(got, str) and path.endswith((".py", ".pyi")) and _ast_equal(exp, got):
            continue
        return False
    return True


@dataclass(frozen=True)
class PatchFeature:
    """The multi-file unified diff coder (see PATCH)."""

    name: str = "patch"
    primary_metric: str = "usable_rate"
    writes_code: bool = True
    schema: JsonSchema = PATCH_SCHEMA
    think_policy: ThinkPolicy = PATCH_THINK
    default_settings: ModelSettings = ModelSettings(temperature=0.1, max_tokens=32768, seed=0)

    def items(self, suite: Suite, split: str) -> list[Item]:
        if suite.kind == "patch":
            out = [
                Item(
                    id=f"patch:{task.id}",
                    feature=self.name,
                    task=task,
                    split=tasks.split_of(task, suite.heldout_pct),
                    cluster=cluster_of(task),
                    user_prompt=task.user_prompt,
                    files=task.files,
                    gold_files=task.gold_files,
                )
                for task in tasks.tasks_in_split(suite, split)
            ]
            return sorted(out, key=lambda item: item.id)
        if suite.kind == "function":
            out = []
            for task in tasks.tasks_in_split(suite, split):
                user_prompt, files, gold_files = _function_patch_prompt(task)
                out.append(
                    Item(
                        id=f"patch:{task.id}",
                        feature=self.name,
                        task=task,
                        split=tasks.split_of(task, suite.heldout_pct),
                        cluster=cluster_of(task),
                        user_prompt=user_prompt,
                        files=files,
                        gold_files=gold_files,
                    )
                )
            return sorted(out, key=lambda item: item.id)
        return []

    def score(self, item: Item, response: ChatResponse, *, timeout_s: float) -> ItemResult:
        think, schema = _read_reply(response, self.schema, self.think_policy)

        diff_text: str | None = None
        diff_source: str | None = None
        if schema.valid and schema.value is not None:
            diff_text, diff_source = schema.value.get("unified_diff"), "strict"
        elif isinstance(schema.lenient_value, dict) and isinstance(schema.lenient_value.get("unified_diff"), str):
            diff_text, diff_source = schema.lenient_value["unified_diff"], "lenient"

        base_files = dict(item.files if item.files is not None else item.task.files)
        gold_files = dict(item.gold_files if item.gold_files is not None else item.task.gold_files)
        updated: dict[str, str | None] | None = None
        apply_status = "no-diff"
        if isinstance(diff_text, str):
            updated, apply_status = diff_apply.apply_unified_diff(base_files, diff_text)
        applied = updated is not None

        changed_files: list[str] = []
        if updated is not None:
            for path in sorted(set(base_files) | set(updated)):
                if path not in base_files:
                    if updated.get(path) is not None:
                        changed_files.append(path)
                elif updated.get(path) != base_files[path]:
                    changed_files.append(path)
        noop = applied and not changed_files

        py_parses = True
        py_syntax_problems: list[str] = []
        findings: list[sast.SastFinding] = []
        if updated is not None:
            for path in changed_files:
                new_text = updated.get(path)
                if new_text is None or not path.endswith((".py", ".pyi")):
                    continue
                orig_text = base_files.get(path)
                if orig_text is None or _can_parse_py(orig_text)[0]:
                    ok, err = _can_parse_py(new_text)
                    if not ok:
                        py_parses = False
                        py_syntax_problems.append(f"syntax: {path}: {err}")
                findings.extend(sast.added_findings(orig_text or "", new_text))

        block = sast.blocking(findings)
        block_rules = tuple(sorted({f.rule for f in block}))
        flag_rules = tuple(sorted({f.rule for f in findings if f.severity == sast.FLAG}))
        gold_match = _matches_gold(base_files, gold_files, updated) if applied else False
        correct = bool(schema.valid and applied and not noop and py_parses and not block)

        problems = _reply_problems(think, schema, self.think_policy)
        if isinstance(diff_text, str):
            if not applied:
                problems.append(f"diff: failed to apply ({apply_status})")
            elif noop:
                problems.append("diff: no files changed (no-op patch)")
        problems += py_syntax_problems
        listed = findings[:MAX_LISTED]
        problems += [f"sast {f.severity}: {f.rule}: {f.snippet}" for f in listed]
        if len(findings) > len(listed):
            problems.append(f"sast: {len(findings) - len(listed)} more finding(s)")

        detail: dict[str, Any] = {
            "tags": list(item.task.tags),
            "think": _think_json(think),
            "schema": _schema_json(schema),
            "diff_source": diff_source,
            "applied": applied,
            "apply_status": apply_status,
            "noop": noop,
            "py_parses": py_parses,
            "changed_files": changed_files,
            "gold_match": gold_match,
            "sast": [f.to_json() for f in findings],
        }
        return ItemResult(
            item_id=item.id,
            feature=self.name,
            task_id=item.task.id,
            cluster=item.cluster,
            split=item.split,
            correct=correct,
            schema_valid=schema.valid,
            lenient_valid=bool(schema.lenient_valid),
            think_present=think.present,
            think_valid=think.valid,
            think_words=think.words,
            finish_reason=response.finish_reason,
            latency_s=response.latency_s,
            prompt_tokens=_usage(response, "prompt_tokens"),
            completion_tokens=_usage(response, "completion_tokens"),
            sast_block=block_rules,
            sast_flag=flag_rules,
            problems=tuple(problems),
            detail=detail,
        )


#: The registered features by name.
FEATURES: Mapping[str, Feature] = {
    "bugfix": BugfixFeature(),
    "judge": JudgeFeature(),
    "review": ReviewFeature(),
    "patch": PatchFeature(),
}


def get_feature(name: str) -> Feature:
    """The registered feature `name`; KeyError naming the known ones otherwise."""
    try:
        return FEATURES[name]
    except KeyError:
        raise KeyError(f"unknown feature {name!r} (known: {', '.join(sorted(FEATURES))})") from None


def cluster_of(task: Task) -> str:
    """The bootstrap's resampling unit for a task's items: its normalised split key."""
    return tasks.normalize_key(task.group or task.id)


# ---- helpers ------------------------------------------------------------------------------------


def _read_reply(response: ChatResponse, schema: JsonSchema, policy: ThinkPolicy) -> tuple[ThinkCheck, SchemaCheck]:
    """The think check of the whole reply and the strict schema check of its answer part."""
    think = verdicts.check_think(response.content, response.reasoning, policy)
    _thinking, answer, _problems = verdicts.split_reply(response.content, response.reasoning)
    return think, verdicts.check_schema(answer, schema, finish_reason=response.finish_reason)


def _reply_problems(think: ThinkCheck, schema: SchemaCheck, policy: ThinkPolicy) -> list[str]:
    problems: list[str] = []
    if not think.valid:
        if think.present and not think.closed:
            problems.append("think: not closed (no answer follows the thinking)")
        elif think.present and not think.in_range:
            problems.append(f"think: {think.words} words, outside {policy.min_words}..{policy.max_words}")
        elif not think.present and policy.required:
            problems.append("think: missing")
    problems += [f"think: {p}" for p in think.problems]
    problems += [f"schema: {p}" for p in schema.problems]
    if not schema.valid and schema.lenient_valid:
        problems.append("schema: valid only under the lenient parse (a format problem)")
    return problems


def _exec_problems(verdict: verdicts.ExecVerdict) -> list[str]:
    failing = [r for r in verdict.results if r.outcome != "passed"]
    problems = [f"exec: {r.kind} test {r.index} {r.outcome}" + (f": {r.detail}" if r.detail else "") for r in failing[:MAX_LISTED]]
    if len(failing) > MAX_LISTED:
        problems.append(f"exec: {len(failing) - MAX_LISTED} more test(s) not passed")
    problems += [f"exec: {p}" for p in verdict.problems[:MAX_LISTED]]
    return problems


def _think_json(think: ThinkCheck) -> dict[str, Any]:
    return {
        "present": think.present,
        "closed": think.closed,
        "words": think.words,
        "in_range": think.in_range,
        "valid": think.valid,
        "source": think.source,
        "problems": list(think.problems),
    }


def _schema_json(schema: SchemaCheck) -> dict[str, Any]:
    return {"valid": schema.valid, "lenient_valid": bool(schema.lenient_valid), "problems": list(schema.problems)}


def _usage(response: ChatResponse, key: str) -> int | None:
    value = response.usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None
