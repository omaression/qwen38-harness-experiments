"""The shared data records: the contract between the modules (docs/DESIGN.md, "Data model").

    Check, Task, Suite              the task suite (tasks.py loads it)
    Item                            one unit a feature is evaluated on (features.py makes them)
    Message, ModelSettings,
    ChatRequest, ChatResponse       what goes to a model and what comes back (prompts.py builds
                                    requests; clients.py and cassettes.py carry them)
    ItemResult, ArmRun              what scoring produces (features.py, runner.py) and what
                                    metrics.py and gate.py read

Results that belong to one module stay in it: ThinkCheck, SchemaCheck and ExecVerdict
(verdicts), SastFinding (sast), SandboxResult (sandbox), Comparison (metrics), GateConfig and
Decision (gate). ItemResult copies the primitive facts it needs from them, so metrics and gate
depend on nothing but this module.

THE REQUEST HASH. ChatRequest.sha256() is a request's identity in a cassette: the sha256 of the
canonical JSON (util.canonical_json) of exactly the body the live client POSTs
(ChatRequest.payload()). The item id is not part of it: it is the lookup key and is never sent.
Adding a key to payload() changes every recorded hash, so it needs a new cassette schema.

This module is complete. Change it only together with every module that reads the changed type.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

from .util import canonical_json, sha256_text

#: The features this harness evaluates (features.FEATURES maps each name to its implementation).
FEATURES = ("bugfix", "judge")

#: The two halves of a suite, and "all" for both together. The gate scores HELDOUT; DEV is for
#: iterating on prompts as often as one likes (docs/DESIGN.md, "Splits").
DEV, HELDOUT, ALL = "dev", "heldout", "all"
SPLITS = (DEV, HELDOUT)
SPLIT_CHOICES = (DEV, HELDOUT, ALL)


# ---- the task suite ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """One test of a task, as data: a call expression and what it must give.

    Exactly one of `expected` and `raises` is set (tasks.check_structure refuses anything else).

      call      a Python expression that calls the task's entry point, e.g. "clamp(5, 0, 3)".
                It is evaluated inside the sandbox, in the namespace of the code under test.
      expected  the source of a Python literal (ast.literal_eval), e.g. "3", "[1, 3]", "None",
                "'abc'". The parent process compares it with the returned value using ==, or
                math.isclose with `approx`. It never enters the sandbox.
      raises    the name of the exception class the call must raise, e.g. "ValueError". A
                subclass passes too: the sandbox reports the raised class's MRO names.
      approx    a tolerance for numbers: math.isclose(got, want, rel_tol=approx, abs_tol=approx).
    """

    call: str
    expected: str | None = None
    raises: str | None = None
    approx: float | None = None

    def to_json(self) -> dict[str, Any]:
        """The record as tasks.jsonl spells it (unset fields left out)."""
        out: dict[str, Any] = {"call": self.call}
        if self.expected is not None:
            out["expected"] = self.expected
        if self.raises is not None:
            out["raises"] = self.raises
        if self.approx is not None:
            out["approx"] = self.approx
        return out


@dataclass(frozen=True)
class Task:
    """One bug-fix task (a line of tasks.jsonl; docs/DESIGN.md, "Task suite").

      id              unique, stable, [a-z0-9][a-z0-9-]*; the item ids are built from it
      entry_point     the function the tests call; buggy, reference and wrong code all define it
      buggy_code      the function with its docstring and one bug: what the model is shown
      shown_tests     the failing test(s) the model is shown (the demo shows exactly one)
      hidden_tests    the tests the model never sees; resolved means shown AND hidden all pass
      reference_code  a correct fix. Never shown to the bug-fix model; it is the judge's
                      positive item and proves the task solvable
      wrong_code      a plausible fix that passes the shown tests and fails a hidden one: the
                      judge's negative item, and the shape of an overfitted patch
      group           the split key when several tasks must stay on one side (twins); default id
      tags            free-form labels (bug class), reported per tag
      synthetic       true for tasks written for this repository rather than taken from real code
    """

    id: str
    entry_point: str
    buggy_code: str
    shown_tests: tuple[Check, ...]
    hidden_tests: tuple[Check, ...]
    reference_code: str
    wrong_code: str
    group: str | None = None
    tags: tuple[str, ...] = ()
    synthetic: bool = False

    @property
    def all_tests(self) -> tuple[Check, ...]:
        """Shown tests first, then hidden ones: the order the sandbox runs them in."""
        return self.shown_tests + self.hidden_tests

    def to_json(self) -> dict[str, Any]:
        """The record as tasks.jsonl spells it, every field present (the suite hash covers it)."""
        return {
            "id": self.id,
            "entry_point": self.entry_point,
            "buggy_code": self.buggy_code,
            "shown_tests": [c.to_json() for c in self.shown_tests],
            "hidden_tests": [c.to_json() for c in self.hidden_tests],
            "reference_code": self.reference_code,
            "wrong_code": self.wrong_code,
            "group": self.group,
            "tags": list(self.tags),
            "synthetic": self.synthetic,
        }


@dataclass(frozen=True)
class Suite:
    """A loaded task suite (tasks.load_suite).

      name, version   from suite.json
      heldout_pct     buckets 0 .. heldout_pct-1 of 100 are held out (tasks.bucket)
      tasks           in file order
      sha256          util.sha256_json({"manifest": <suite.json object>, "tasks": [to_json() of
                      every task, sorted by id]}): the suite's identity, independent of formatting
      source          the directory as util.display_path shows it (never an absolute path)
      synthetic       the manifest's flag: the tasks were written for this repository
    """

    name: str
    version: int
    heldout_pct: int
    tasks: tuple[Task, ...]
    sha256: str
    source: str
    synthetic: bool = False
    description: str = ""

    def task(self, task_id: str) -> Task:
        """The task with this id (KeyError if there is none)."""
        for t in self.tasks:
            if t.id == task_id:
                return t
        raise KeyError(task_id)


@dataclass(frozen=True)
class Item:
    """One unit a feature is evaluated on (features.Feature.items).

      id            "bugfix:<task id>" or "judge:<task id>:<variant>"; the cassette lookup key.
                    Never rendered into a prompt.
      feature       "bugfix" or "judge"
      task          the task it comes from
      split         DEV or HELDOUT, the task's split
      cluster       the bootstrap's resampling unit: the task's split key (group or id), so a
                    judge task's two items, and twins, are drawn together
      variant       judge only: "reference" or "wrong"
      label         judge only: the verdict a correct judge gives (True for "reference")
      patched_code  judge only: the code under review (the task's reference or wrong code)
    """

    id: str
    feature: str
    task: Task
    split: str
    cluster: str
    variant: str | None = None
    label: bool | None = None
    patched_code: str | None = None


# ---- requests and responses -------------------------------------------------------------------


@dataclass(frozen=True)
class Message:
    """One chat message: role "system", "user" or "assistant", and its text."""

    role: str
    content: str


@dataclass(frozen=True)
class ModelSettings:
    """The sampling settings of a recording. Part of every request hash.

      extra   further body keys sent as given, e.g. {"chat_template_kwargs": {"enable_thinking":
              true}} for servers that take it. It may not override a key payload() sets.
    """

    temperature: float = 0.0
    max_tokens: int = 4096
    seed: int | None = 0
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "max_tokens": self.max_tokens, "seed": self.seed, "extra": dict(self.extra)}

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "ModelSettings":
        return cls(
            temperature=float(obj.get("temperature", 0.0)),
            max_tokens=int(obj.get("max_tokens", 4096)),
            seed=None if obj.get("seed") is None else int(obj["seed"]),
            extra=dict(obj.get("extra") or {}),
        )


@dataclass(frozen=True)
class ChatRequest:
    """A chat-completions request for one item (prompts.build_request makes them)."""

    item_id: str
    model: str
    messages: tuple[Message, ...]
    settings: ModelSettings = field(default_factory=ModelSettings)

    def payload(self) -> dict[str, Any]:
        """The JSON body POSTed to <base_url>/chat/completions. The item id is not in it."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        if self.settings.seed is not None:
            body["seed"] = self.settings.seed
        for key in sorted(self.settings.extra):
            if key in body:
                raise ValueError(f"ModelSettings.extra may not override {key!r}")
            body[key] = self.settings.extra[key]
        return body

    def sha256(self) -> str:
        """sha256 of canonical_json(payload()): the request's identity in a cassette."""
        return sha256_text(canonical_json(self.payload()))


@dataclass(frozen=True)
class ChatResponse:
    """What a model returned for one request.

      content        the assistant message's text ("" when the server sent null)
      reasoning      thinking the server returned separately (vLLM's `reasoning_content`, or
                     `reasoning`); None when the server sent none. Thinking inside `content` as a
                     <think>...</think> block stays in `content`
      finish_reason  "stop", "length", ... as the server reported it; None when unknown
      usage          token counts as reported: prompt_tokens, completion_tokens, total_tokens,
                     reasoning_tokens (flattened from completion_tokens_details) when present
      latency_s      wall time of the call, as measured when it was recorded
    """

    content: str
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    latency_s: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "finish_reason": self.finish_reason,
            "usage": dict(self.usage),
            "latency_s": self.latency_s,
        }

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "ChatResponse":
        latency = obj.get("latency_s")
        return cls(
            content=str(obj.get("content") or ""),
            reasoning=None if obj.get("reasoning") is None else str(obj["reasoning"]),
            finish_reason=None if obj.get("finish_reason") is None else str(obj["finish_reason"]),
            usage={str(k): int(v) for k, v in (obj.get("usage") or {}).items() if isinstance(v, int) and not isinstance(v, bool)},
            latency_s=None if latency is None else float(latency),
        )


# ---- results ----------------------------------------------------------------------------------


@dataclass
class ItemResult:
    """The scored outcome of one item for one arm (features.Feature.score). One line of an
    items file (to_json); metrics and the gate read nothing else.

      correct        the per-item outcome the primary metric counts. bugfix: resolved, i.e. the
                     reply is schema-valid, its code adds no block-class SAST finding, and every
                     shown and hidden test passes under the strict verdict. judge: the verdict is
                     valid and equals the label. A reply that cannot be scored is not correct.
      schema_valid   the answer is exactly the feature's JSON object (verdicts.check_schema)
      lenient_valid  a JSON object with the right keys could be dug out of the answer (diagnostic)
      think_*        verdicts.check_think: present, valid, and the thinking's word count
      label, predicted          judge only (predicted None: no valid verdict)
      shown_passed, hidden_passed  bugfix only (None when the code was not run)
      sast_block, sast_flag     bugfix only: rule ids of the findings the reply's code added,
                                block and flag class (sast.added_findings), sorted and unique
      problems       short reasons for the item log ("schema: missing key 'patched_code'", ...)
      detail         free-form, JSON-serialisable: per-test outcomes, findings, parse notes
    """

    item_id: str
    feature: str
    task_id: str
    cluster: str
    split: str
    correct: bool
    schema_valid: bool
    lenient_valid: bool
    think_present: bool
    think_valid: bool
    think_words: int
    finish_reason: str | None = None
    latency_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    label: bool | None = None
    predicted: bool | None = None
    shown_passed: bool | None = None
    hidden_passed: bool | None = None
    sast_block: tuple[str, ...] = ()
    sast_flag: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """A JSON-ready dict (tuples become lists)."""
        out = asdict(self)
        for key in ("sast_block", "sast_flag", "problems"):
            out[key] = list(out[key])
        return out

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "ItemResult":
        """The inverse of to_json; unknown keys are ignored."""
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in obj.items() if k in known}
        for key in ("sast_block", "sast_flag", "problems"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        if "detail" in kwargs:
            kwargs["detail"] = dict(kwargs["detail"] or {})
        return cls(**kwargs)


@dataclass
class ArmRun:
    """One arm (a cassette set) evaluated on one feature and split (runner.evaluate_arm).

      name           the arm name from the cassette manifest ("baseline", "candidate", ...)
      source         the cassette directory as util.display_path shows it
      fingerprint    cassettes.fingerprint of that directory: what the ledger records
      model, prompt  the model name and prompt template id the responses were recorded with
      prompt_sha256  prompts.PromptTemplate.sha256 of that template
      synthetic      the manifest says the responses were made up for the demo, not recorded
      suite_sha256   the suite this run scored; recorded_suite_sha256 is the one the cassettes
                     were recorded on (they differ after a task edit that left the prompts alone)
      results        one ItemResult per item, sorted by item id
      metrics        metrics.arm_metrics(feature, results)
      notes          arm-level warnings for the report (suite drift, items skipped, ...)
    """

    name: str
    feature: str
    split: str
    source: str
    fingerprint: str
    model: str
    prompt: str
    prompt_sha256: str
    synthetic: bool
    suite_sha256: str
    recorded_suite_sha256: str | None
    results: list[ItemResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """The arm as the report shows it: everything but the per-item results."""
        return {
            "name": self.name,
            "source": self.source,
            "fingerprint": self.fingerprint,
            "model": self.model,
            "prompt": self.prompt,
            "prompt_sha256": self.prompt_sha256,
            "synthetic": self.synthetic,
            "recorded_suite_sha256": self.recorded_suite_sha256,
            "metrics": self.metrics,
            "notes": list(self.notes),
        }
