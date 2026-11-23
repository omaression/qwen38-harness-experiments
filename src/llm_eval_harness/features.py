"""The features under evaluation: what their items are and how one reply is scored.

    get_feature(name) -> Feature        "bugfix" or "judge"
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
Both: ItemResult.problems lists short reasons; latency, tokens and finish_reason are copied
from the response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from . import sast, tasks, verdicts
from .records import ChatResponse, Item, ItemResult, ModelSettings, Suite, Task
from .sandbox import SandboxUnavailable
from .verdicts import BUGFIX_SCHEMA, JUDGE_SCHEMA, JsonSchema, SchemaCheck, ThinkCheck, ThinkPolicy

#: Thinking is optional for both features; when present it must be closed and at most 1,200 words.
BUGFIX_THINK = ThinkPolicy(required=False, min_words=1, max_words=1200)
JUDGE_THINK = ThinkPolicy(required=False, min_words=1, max_words=1200)
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


#: The registered features by name.
FEATURES: Mapping[str, Feature] = {"bugfix": BugfixFeature(), "judge": JudgeFeature()}


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
