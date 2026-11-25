"""Evaluating an arm, recording one, and the whole gate run.

    evaluate_arm(feature, suite, split, cassette_dir)   replay one cassette set and score it
    record_arm(feature, suite, split, client, out_dir)  ask a live model and write a cassette set
    gate_run(feature, suite, split, candidate, baseline, config)   both arms, the decision, the report

EVALUATE. Loads the cassette set (cassettes.load_cassettes) and its tape for the feature, the
template the tape names (prompts.get_template), the feature's items for the split, and the
recordings. For each item it rebuilds the request (prompts.build_request with the tape's model
and settings), answers it with a clients.ReplayClient, and scores the reply
(Feature.score). Items are scored on a thread pool of `workers` (each bugfix item starts one
sandbox process); results are sorted by item id, so the output does not depend on scheduling.
  - an item with no recording, or a recording made for another request, refuses the whole
    evaluation (RefusedError, exit 2): the comparison would not be paired. allow_missing=True
    drops such items instead and names them in ArmRun.notes;
  - a tape recorded on another suite version (suite_sha256 differs) is evaluated, with a note:
    the answers fit the prompts (their hashes matched), the scoring uses today's tests;
  - a ScoringError (the sandbox did not start) stops everything (RunError, exit 1).
RECORD. The same loop with a clients.RecordingClient around a live client, writing
<out_dir>/<feature>.jsonl and manifest.json (cassettes.CassetteWriter). Items already recorded
with the same request hash are skipped, so an interrupted recording resumes. A request that
still fails after the client's retries is counted in `failed` and left unrecorded (its message
names the endpoint "<endpoint>", never by its URL). Recording scores nothing.
GATE. evaluate_arm for the candidate and (unless the baseline is None) the baseline, then
gate.check_comparable, metrics.compare, gate.decide, gate.report_warnings (with the held-out
openings from the ledger, when one is given) and gate.build_report. With items_out, writes
<items_out>/<feature>.candidate.items.jsonl and <feature>.baseline.items.jsonl
(ItemResult.to_json per line). With record_evaluation, appends an evaluation row to the ledger.
With allow_missing, an item only one arm could score is dropped from both, with a note, so the
comparison stays paired.
Logs progress to stderr only.
PROGRESS. Each function takes progress=, a callable that receives the progress lines (default
util.log, i.e. stderr); `llm-eval --quiet` passes one that drops them. Warnings (items dropped,
suite drift, failed recordings) always go to util.log.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from . import __version__, cassettes, metrics, prompts, sast, tasks, verdicts
from . import gate as gate_mod
from .clients import CassetteMismatch, ClientError, MissingRecording, ModelClient, RecordingClient, ReplayClient
from .features import Feature, ScoringError, get_feature
from .gate import GateConfig, GateRefused
from .ledger import Ledger
from .prompts import PromptError, PromptTemplate
from .records import HELDOUT, SPLIT_CHOICES, SPLITS, ArmRun, ChatResponse, Item, ItemResult, ModelSettings, Suite
from .report import dumps_report
from .util import display_path, log, sha256_text, write_jsonl

DEFAULT_WORKERS = 4
#: Item ids spelt out in a refusal or a note before "and N more".
MAX_NAMED = 10

#: A sink for progress lines (default util.log, i.e. stderr). Warnings always go to util.log.
Progress = Callable[[str], None]


class RunError(RuntimeError):
    """The evaluation could not run or finish (I/O, the sandbox, the endpoint). CLI exit 1."""


class RefusedError(RuntimeError):
    """The inputs do not allow a fair evaluation (missing or mismatched recordings, arms that
    are not comparable, a bad argument). CLI exit 2."""


@dataclass
class RecordSummary:
    """record_arm's result, printed as JSON by `llm-eval record`."""

    arm: str
    feature: str
    split: str
    out: str
    prompt: str
    model: str
    recorded: int = 0
    skipped_existing: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "feature": self.feature,
            "split": self.split,
            "out": self.out,
            "prompt": self.prompt,
            "model": self.model,
            "recorded": self.recorded,
            "skipped_existing": self.skipped_existing,
            "failed": self.failed,
            "failures": list(self.failures),
        }


def evaluate_arm(
    feature: str,
    suite: Suite,
    split: str,
    cassette_dir: str | os.PathLike[str],
    *,
    workers: int = DEFAULT_WORKERS,
    timeout_s: float = 10.0,
    allow_missing: bool = False,
    progress: Progress | None = None,
) -> ArmRun:
    """See EVALUATE."""
    say = progress or log
    feat = _feature(feature)
    _check_split(split, SPLIT_CHOICES)
    try:
        cset = cassettes.load_cassettes(cassette_dir)
        tape = cset.features.get(feature)
        if tape is None:
            listed = ", ".join(sorted(cset.features)) or "none"
            raise RefusedError(f"{cset.source} holds no {feature} recordings (its manifest has: {listed})")
        recordings = cassettes.read_recordings(cset, feature)
    except cassettes.CassetteError as exc:
        raise RefusedError(_text(exc)) from None
    try:
        template = prompts.get_template(tape.prompt, feature)
    except PromptError as exc:
        raise RefusedError(f"{cset.source}: {_text(exc)}") from None

    replay = ReplayClient(recordings)
    answered: list[tuple[Item, ChatResponse]] = []
    missing: list[str] = []
    mismatched: list[str] = []
    for item in feat.items(suite, split):
        try:
            request = prompts.build_request(template, item, tape.model, tape.settings)
        except ValueError as exc:
            raise RefusedError(f"{cset.source}: the {feature} request cannot be rebuilt: {exc}") from None
        try:
            answered.append((item, replay.complete(request)))
        except MissingRecording:
            missing.append(item.id)
        except CassetteMismatch:
            mismatched.append(item.id)

    notes: list[str] = []
    if missing or mismatched:
        reasons = _unfit(missing, mismatched)
        if not allow_missing:
            raise RefusedError(
                f"{cset.source}: {feature} on {split}: {reasons}. An unpaired comparison would compare different "
                "questions: record again, or pass --allow-missing to drop these items"
            )
        notes.append(f"{len(missing) + len(mismatched)} {feature} item(s) dropped (allow-missing): {reasons}")
        log(f"warning: {cset.arm}: {notes[-1]}")
    if tape.suite_sha256 and tape.suite_sha256 != suite.sha256:
        notes.append(
            f"recorded on another version of the suite ({tape.suite or 'unnamed'} {tape.suite_sha256[:12]}; scored on "
            f"{suite.name} {suite.sha256[:12]}): every request matched its recording, so the prompts are unchanged, "
            "and the scoring uses today's tests"
        )
        log(f"note: {cset.arm}: {notes[-1]}")

    say(f"{cset.arm} ({cset.source}): scoring {len(answered)} {feature} item(s) on {split} with {workers} worker(s)")
    try:
        results = _score_all(feat, answered, workers=workers, timeout_s=timeout_s)
    except ScoringError as exc:
        raise RunError(f"{cset.source}: {exc}; the evaluation stopped and nothing was scored") from None
    arm_metrics = metrics.arm_metrics(feature, results)
    say(f"{cset.arm}: {feat.primary_metric} = {_fmt(arm_metrics.get(feat.primary_metric))} on {len(results)} item(s)")
    return ArmRun(
        name=cset.arm,
        feature=feature,
        split=split,
        source=cset.source,
        fingerprint=cset.fingerprint,
        model=tape.model,
        prompt=tape.prompt,
        prompt_sha256=template.sha256(),
        synthetic=cset.synthetic,
        suite_sha256=suite.sha256,
        recorded_suite_sha256=tape.suite_sha256,
        results=results,
        metrics=arm_metrics,
        notes=notes,
    )


def record_arm(
    feature: str,
    suite: Suite,
    split: str,
    client: ModelClient,
    out_dir: str | os.PathLike[str],
    *,
    arm: str,
    model: str,
    template: PromptTemplate,
    settings: ModelSettings,
    workers: int = DEFAULT_WORKERS,
    description: str = "",
    progress: Progress | None = None,
) -> RecordSummary:
    """See RECORD."""
    say = progress or log
    feat = _feature(feature)
    _check_split(split, SPLIT_CHOICES)
    if template.feature != feature:
        raise RefusedError(f"template {template.id} belongs to {template.feature}, not {feature}")
    root = Path(out_dir)
    summary = RecordSummary(arm=arm, feature=feature, split=split, out=display_path(root), prompt=template.id, model=model)
    try:
        requests = [prompts.build_request(template, item, model, settings) for item in feat.items(suite, split)]
    except ValueError as exc:
        raise RefusedError(f"the {feature} requests cannot be built: {exc}") from None
    try:
        writer = cassettes.CassetteWriter(
            root,
            arm=arm,
            feature=feature,
            prompt=template.id,
            model=model,
            settings=settings,
            suite_name=suite.name,
            suite_sha256=suite.sha256,
            synthetic=False,
            description=description,
        )
        try:
            pending = []
            for request in requests:
                if writer.has(request):
                    summary.skipped_existing += 1
                else:
                    pending.append(request)
            say(f"recording {len(pending)} {feature} item(s) on {split} from {model} into {summary.out} "
                f"({summary.skipped_existing} already recorded)")
            recorder = RecordingClient(client, writer)
            failures: list[tuple[str, str]] = []

            # The endpoint's URL is often a secret (the CI live job keeps it in one): keep it
            # out of the summary, which is printed and may be uploaded.
            endpoint = getattr(client, "base_url", None)

            def ask(request: Any) -> None:
                try:
                    recorder.complete(request)
                except ClientError as exc:
                    message = _text(exc)
                    if isinstance(endpoint, str) and endpoint:
                        message = message.replace(endpoint, "<endpoint>")
                    failures.append((request.item_id, message))

            if pending:
                with ThreadPoolExecutor(max_workers=max(1, min(workers, len(pending)))) as pool:
                    for future in [pool.submit(ask, request) for request in pending]:
                        future.result()
            failures.sort()
            summary.failed = len(failures)
            summary.failures = [f"{item_id}: {message}" for item_id, message in failures]
            summary.recorded = len(pending) - len(failures)
        finally:
            writer.close()
    except cassettes.CassetteError as exc:
        raise RefusedError(_text(exc)) from None
    say(f"recorded {summary.recorded}, skipped {summary.skipped_existing} already recorded, failed {summary.failed}")
    for failure in summary.failures[:MAX_NAMED]:
        log(f"warning: not recorded: {failure}")
    return summary


def gate_run(
    feature: str,
    suite: Suite,
    split: str,
    candidate_dir: str | os.PathLike[str],
    baseline_dir: str | os.PathLike[str] | None,
    config: GateConfig,
    *,
    resamples: int = 2000,
    seed: int = 0,
    workers: int = DEFAULT_WORKERS,
    timeout_s: float = 10.0,
    allow_missing: bool = False,
    items_out: str | os.PathLike[str] | None = None,
    ledger: Ledger | None = None,
    record_evaluation: bool = False,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """See GATE. Returns the report dict (gate.build_report)."""
    say = progress or log
    _feature(feature)
    _check_split(split, SPLITS)
    try:
        config.validate()
    except ValueError as exc:
        raise RefusedError(f"gate configuration: {exc}") from None
    if record_evaluation and ledger is None:
        raise RefusedError("recording an evaluation needs a ledger")

    options = {"workers": workers, "timeout_s": timeout_s, "allow_missing": allow_missing, "progress": progress}
    candidate = evaluate_arm(feature, suite, split, candidate_dir, **options)
    baseline = None if baseline_dir is None else evaluate_arm(feature, suite, split, baseline_dir, **options)
    if allow_missing and baseline is not None:
        common = {r.item_id for r in candidate.results} & {r.item_id for r in baseline.results}
        candidate = _restrict(feature, candidate, common, "the baseline")
        baseline = _restrict(feature, baseline, common, "the candidate")
    try:
        gate_mod.check_comparable(candidate, baseline)
    except GateRefused as exc:
        raise RefusedError(f"the arms cannot be compared: {_text(exc)}") from None

    comparison = None
    if baseline is not None:
        comparison = metrics.compare(feature, candidate.results, baseline.results, split=split, resamples=resamples, seed=seed, alpha=config.alpha)
    decision = gate_mod.decide(candidate, baseline, comparison, config)
    split_sha256 = tasks.split_sha256(suite, split)
    openings = None
    if split == HELDOUT and ledger is not None:
        has_ledger = Path(ledger.path).is_file()
        if record_evaluation:
            prior = ledger.heldout_openings(split_sha256) if has_ledger else 0
            already = has_ledger and ledger.has_candidate_opening(split_sha256, candidate.fingerprint)
            openings = prior + (0 if already else 1)
        elif has_ledger:
            openings = ledger.heldout_openings(split_sha256)
    warnings = gate_mod.report_warnings(
        candidate, baseline, comparison, config, current_prompt=prompts.CURRENT.get(feature), heldout_openings=openings
    )
    report = gate_mod.build_report(
        feature=feature,
        split=split,
        suite=suite_block(suite, split, feature),
        candidate=candidate,
        baseline=baseline,
        comparison=comparison,
        decision=decision,
        config=config,
        warnings=warnings,
        provenance=provenance(feature),
    )
    if items_out is not None:
        out = Path(items_out)
        write_jsonl(_items_path(out, feature, "candidate"), [r.to_json() for r in candidate.results])
        if baseline is not None:
            write_jsonl(_items_path(out, feature, "baseline"), [r.to_json() for r in baseline.results])
        say(f"wrote the scored items to {display_path(out)}")
    if record_evaluation and ledger is not None:
        ledger.record_evaluation(report, report_sha256=sha256_text(dumps_report(report)))
        say(f"recorded the evaluation in {display_path(ledger.path)}")
    failed = [g["gate"] for g in report.get("gates", []) if g.get("enforced", True) and not g.get("passed")]
    say(f"gate {feature} on {split}: promote={str(report.get('promote')).lower()}" + (f" (failed: {', '.join(failed)})" if failed else ""))
    return report


def suite_block(suite: Suite, split: str, feature: str) -> dict[str, Any]:
    """The report's `suite` block: what was evaluated, by identity, never by path."""
    return {
        "name": suite.name,
        "sha256": suite.sha256,
        "split_sha256": tasks.split_sha256(suite, split),
        "tasks": len(tasks.tasks_in_split(suite, split)),
        "items": len(get_feature(feature).items(suite, split)),
        "heldout_pct": suite.heldout_pct,
        "synthetic": suite.synthetic,
    }


def provenance(feature: str) -> dict[str, Any]:
    """The report's `provenance` block: which harness, rules and prompt produced it."""
    return {
        "harness_version": __version__,
        "sast": sast.describe(),
        "exec_rule": verdicts.EXEC_RULE,
        "prompt_current": prompts.CURRENT.get(feature),
    }


# ---- helpers ------------------------------------------------------------------------------------


def _feature(name: str) -> Feature:
    try:
        return get_feature(name)
    except KeyError as exc:
        raise RefusedError(_text(exc)) from None


def _check_split(split: str, allowed: Sequence[str]) -> None:
    if split not in allowed:
        raise RefusedError(f"unknown split {split!r} here (use one of: {', '.join(allowed)})")


def _score_one(feat: Feature, item: Item, response: ChatResponse, timeout_s: float) -> ItemResult:
    return feat.score(item, response, timeout_s=timeout_s)


def _score_all(feat: Feature, answered: list[tuple[Item, ChatResponse]], *, workers: int, timeout_s: float) -> list[ItemResult]:
    """Score every (item, response); the first exception cancels what has not started."""
    if workers <= 1 or len(answered) <= 1:
        results = [_score_one(feat, item, response, timeout_s) for item, response in answered]
    else:
        pool = ThreadPoolExecutor(max_workers=min(workers, len(answered)))
        try:
            futures = [pool.submit(_score_one, feat, item, response, timeout_s) for item, response in answered]
            for future in as_completed(futures):
                future.result()
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        results = [future.result() for future in futures]
    return sorted(results, key=lambda r: r.item_id)


def _restrict(feature: str, arm: ArmRun, keep: set[str], other: str) -> ArmRun:
    """The arm on the items in `keep` only (allow-missing on a gate: both arms keep the items
    both could score, so the comparison stays paired)."""
    dropped = sorted(r.item_id for r in arm.results if r.item_id not in keep)
    if not dropped:
        return arm
    results = [r for r in arm.results if r.item_id in keep]
    note = f"{len(dropped)} item(s) dropped because {other} has no fitting recording for them (allow-missing): {_names(dropped)}"
    log(f"warning: {arm.name}: {note}")
    return replace(arm, results=results, metrics=metrics.arm_metrics(feature, results), notes=[*arm.notes, note])


def _unfit(missing: list[str], mismatched: list[str]) -> str:
    parts = []
    if missing:
        parts.append(f"no recording for {len(missing)} item(s) ({_names(missing)})")
    if mismatched:
        parts.append(
            f"{len(mismatched)} recording(s) answered a different request ({_names(mismatched)}): the task, the "
            "template, the model or the settings changed since recording"
        )
    return "; ".join(parts)


def _names(ids: Sequence[str]) -> str:
    shown = ", ".join(ids[:MAX_NAMED])
    return shown + (f" and {len(ids) - MAX_NAMED} more" if len(ids) > MAX_NAMED else "")


def _text(exc: BaseException) -> str:
    """An exception's message on one line (KeyError's str() would add quotes)."""
    text = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
    return " ".join(text.split()) or type(exc).__name__


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _items_path(items_out: Path, feature: str, role: str) -> Path:
    return items_out / f"{feature}.{role}.items.jsonl"
