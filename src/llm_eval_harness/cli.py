"""The `llm-eval` command (also `python -m llm_eval_harness`).

    llm-eval tasks validate [--suite DIR] [--no-exec] [--timeout S]
    llm-eval tasks split    [--suite DIR]
    llm-eval run     --feature F --cassettes DIR [--suite DIR] [--split dev|heldout|all] [--items-out FILE]
    llm-eval record  --feature F --model NAME --out DIR [--prompt ID | --prompt-from DIR]
                     [--base-url URL] [--api-key-env VAR] [--split all] [--arm NAME] ...
    llm-eval gate    --feature F --candidate DIR --baseline DIR|@current|none [--split heldout]
                     [--margin X] [--tie pass|fail] [--alpha A] [--require-gain] [--items-out DIR] ...
    llm-eval summary REPORT [REPORT ...] [--title TEXT]
    llm-eval ledger  verify | status | current --feature F | promote ... | rollback ...

build_parser() below is the command-line contract (CI, the README and the tests call these
exact flags); tests/test_contract.py parses the documented command lines against it.

OUTPUT. Every command prints exactly one document on stdout: JSON (tasks, run, record, gate,
ledger verify / status / promote / rollback), markdown (summary), or one line of text (ledger
current: the baseline directory, for $(...) in a shell). Logs and progress go to stderr. While a
command works, file descriptor 1 is pointed at stderr (os.dup / os.dup2), so nothing a library or
a child process writes can reach stdout; the document is written to the saved descriptor at the
end. `llm-eval gate ... > report.json` therefore always holds exactly one parseable report.
--quiet drops the progress lines; warnings, problems and errors are always logged. A command
that fails prints no document, and one line "llm-eval: refused: ..." or "llm-eval: error: ..."
on stderr instead of a traceback.

EXIT STATUS.
  0  the command did its job. For gate: the evaluation ran and the report was printed, whatever
     the verdict. CI reads the verdict from the report (jq -e '.promote == true'), so "the gate
     said no" and "the gate could not run" stay distinguishable.
  1  it could not run or finish (runner.RunError, an endpoint or sandbox failure, I/O); for
     `tasks validate` an invalid suite; for `ledger verify` a damaged ledger; for `record` an
     item that could not be recorded.
  2  a usage error or a refusal: bad arguments (argparse's own errors exit 2 too), arms that
     cannot be compared, missing or mismatched recordings, a refused promotion or rollback.

SECRETS. `record` reads the API key from the environment variable --api-key-env names (default
LLM_EVAL_API_KEY), never from a flag value, so keys stay out of shell history and process lists.
The base URL comes from --base-url or $LLM_EVAL_BASE_URL.

THE BASELINE. `gate --baseline` takes a cassette directory, "@current" (the directory the ledger
names as the feature's current baseline: ledger.Ledger.current) or "none" (no baseline: the
primary metric is compared with the floor).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from . import __version__, cassettes, prompts, runner, tasks, util, validation
from . import report as report_mod
from .cassettes import CassetteError
from .clients import API_KEY_ENV, BASE_URL_ENV, CassetteMismatch, ClientError, MissingRecording, client_from_env
from .features import ScoringError, get_feature
from .gate import GateConfig, GateRefused
from .ledger import DEFAULT_LEDGER, Ledger, LedgerError, LedgerRefused
from .metrics import DEFAULT_RESAMPLES, DEFAULT_SEED
from .prompts import PromptError
from .records import ALL, DEV, FEATURES, HELDOUT, SPLIT_CHOICES, SPLITS, ModelSettings
from .report import DEFAULT_TITLE, ReportError
from .runner import DEFAULT_WORKERS, RefusedError, RunError
from .sandbox import DEFAULT_TIMEOUT_S, SandboxUnavailable
from .tasks import DEFAULT_SUITE, SuiteError

EXIT_OK, EXIT_ERROR, EXIT_USAGE = 0, 1, 2
#: --baseline values that are not directories.
BASELINE_CURRENT, BASELINE_NONE = "@current", "none"
_DEFAULT_GATE = GateConfig()


# ---- argument types ---------------------------------------------------------------------------


def _fraction(text: str) -> float:
    """A float in [0, 1]."""
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"{text!r} is not between 0 and 1")
    return value


def _open_fraction(text: str) -> float:
    """A float strictly between 0 and 1."""
    value = float(text)
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError(f"{text!r} is not strictly between 0 and 1")
    return value


def _nonnegative_float(text: str) -> float:
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} is negative")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not positive")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} is negative")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not positive")
    return value


def _seed(text: str) -> int | None:
    """An integer seed, or "none" to send no seed."""
    return None if text.lower() == "none" else int(text)


def _json_object(text: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"not JSON: {exc.msg}") from None
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("not a JSON object")
    return value


# ---- the parser ------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The whole command-line interface. Options are taken by their full names only."""
    parser = argparse.ArgumentParser(
        prog="llm-eval",
        description="Evaluate an LLM feature on a task suite and gate a candidate against a baseline.",
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-q", "--quiet", action="store_true", help="log only warnings and errors to stderr")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def sub(group: argparse._SubParsersAction, name: str, help_text: str) -> argparse.ArgumentParser:
        return group.add_parser(name, help=help_text, description=help_text, allow_abbrev=False)

    def suite_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--suite", default=DEFAULT_SUITE, metavar="DIR", help=f"the task suite directory (default: {DEFAULT_SUITE})")

    def ledger_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--ledger", default=DEFAULT_LEDGER, metavar="FILE", help=f"the ledger file (default: {DEFAULT_LEDGER})")

    def scoring_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS, help=f"items scored at once (default: {DEFAULT_WORKERS})")
        p.add_argument("--timeout", type=_positive_float, default=DEFAULT_TIMEOUT_S, metavar="SECONDS",
                       help=f"sandbox deadline per item (default: {DEFAULT_TIMEOUT_S:g})")
        p.add_argument("--allow-missing", action="store_true", help="drop items without a fitting recording instead of refusing")

    # tasks
    tasks = sub(commands, "tasks", "check and inspect a task suite")
    tasks_actions = tasks.add_subparsers(dest="action", required=True, metavar="ACTION")
    validate = sub(tasks_actions, "validate", "check a suite's structure, run its code in the sandbox, look for leakage")
    suite_arg(validate)
    validate.add_argument("--no-exec", action="store_true", help="structure and leakage checks only; run no code")
    validate.add_argument("--timeout", type=_positive_float, default=DEFAULT_TIMEOUT_S, metavar="SECONDS", help="sandbox deadline per run")
    split = sub(tasks_actions, "split", "show which tasks are dev and which are held out")
    suite_arg(split)

    # run
    run = sub(commands, "run", "replay one cassette set on a split and print its metrics")
    run.add_argument("--feature", required=True, choices=FEATURES)
    run.add_argument("--cassettes", required=True, metavar="DIR", help="the cassette set to replay")
    suite_arg(run)
    run.add_argument("--split", choices=SPLIT_CHOICES, default=DEV, help=f"default: {DEV} (the held-out split is the gate's)")
    run.add_argument("--items-out", metavar="FILE", help="write one scored item per line here")
    scoring_args(run)

    # record
    record = sub(commands, "record", "ask a live OpenAI-compatible endpoint and write a cassette set")
    record.add_argument("--feature", required=True, choices=FEATURES)
    record.add_argument("--model", required=True, help="the model name the endpoint serves")
    record.add_argument("--out", required=True, metavar="DIR", help="the cassette directory to write (resumed if it exists)")
    record.add_argument("--arm", help="the arm name in the manifest (default: the directory's name)")
    suite_arg(record)
    record.add_argument("--split", choices=SPLIT_CHOICES, default=ALL, help=f"default: {ALL}")
    prompt = record.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", metavar="ID", help="the template id, e.g. bugfix/v2 (default: the feature's current one)")
    prompt.add_argument("--prompt-from", metavar="DIR", help="use the template and settings another cassette set used for this feature")
    record.add_argument("--base-url", help="e.g. http://localhost:8000/v1 (default: $LLM_EVAL_BASE_URL)")
    record.add_argument("--api-key-env", default=API_KEY_ENV, metavar="VAR", help=f"the environment variable holding the key (default: {API_KEY_ENV})")
    record.add_argument("--temperature", type=_nonnegative_float, help="default: the feature's (0.0)")
    record.add_argument("--max-tokens", type=_positive_int, help="default: the feature's")
    record.add_argument("--seed", type=_seed, help='an integer, or "none" to send no seed (default: 0)')
    record.add_argument("--extra", type=_json_object, metavar="JSON", help="further request body keys, as a JSON object")
    record.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS, help=f"requests in flight (default: {DEFAULT_WORKERS})")
    record.add_argument("--request-timeout", type=_positive_float, default=120.0, metavar="SECONDS", help="per request (default: 120)")
    record.add_argument("--max-retries", type=_nonnegative_int, default=3, help="per request (default: 3)")
    record.add_argument("--description", default="", help="free text stored in the manifest")

    # gate
    gate = sub(commands, "gate", "evaluate a candidate against a baseline and print the JSON report")
    gate.add_argument("--feature", required=True, choices=FEATURES)
    gate.add_argument("--candidate", required=True, metavar="DIR", help="the candidate's cassette set")
    gate.add_argument("--baseline", required=True, metavar="DIR",
                      help=f'the baseline\'s cassette set, "{BASELINE_CURRENT}" (the ledger\'s current baseline) or "{BASELINE_NONE}"')
    suite_arg(gate)
    gate.add_argument("--split", choices=SPLITS, default=HELDOUT, help=f"default: {HELDOUT}")
    gate.add_argument("--margin", type=_nonnegative_float, default=_DEFAULT_GATE.margin, help="non-inferiority margin on the primary metric (default: 0)")
    gate.add_argument("--tie", choices=("pass", "fail"), default=_DEFAULT_GATE.tie, help="whether a candidate exactly at the bound passes (default: pass)")
    gate.add_argument("--alpha", type=_open_fraction, default=_DEFAULT_GATE.alpha, help="level of the exact McNemar tests (default: 0.05)")
    gate.add_argument("--schema-min", type=_fraction, default=_DEFAULT_GATE.schema_min, help="default: 0.95")
    gate.add_argument("--think-min", type=_fraction, default=_DEFAULT_GATE.think_min, help="default: 0.90")
    gate.add_argument("--max-new-sast", type=_nonnegative_int, default=_DEFAULT_GATE.max_new_sast, help="default: 0")
    gate.add_argument("--require-gain", action="store_true", help="also require a significant gain (superiority)")
    gate.add_argument("--floor", type=_fraction, help="the primary metric's bound without a baseline (default: the feature's)")
    gate.add_argument("--resamples", type=_positive_int, default=DEFAULT_RESAMPLES, help=f"bootstrap draws (default: {DEFAULT_RESAMPLES})")
    gate.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"bootstrap seed (default: {DEFAULT_SEED})")
    gate.add_argument("--items-out", metavar="DIR", help="write both arms' scored items here")
    ledger_arg(gate)
    gate.add_argument("--record-evaluation", action="store_true", help="append an evaluation row to the ledger (never in CI)")
    scoring_args(gate)

    # summary
    summary = sub(commands, "summary", "render gate reports as markdown (for $GITHUB_STEP_SUMMARY)")
    summary.add_argument("reports", nargs="+", metavar="REPORT", help="report files written by llm-eval gate")
    summary.add_argument("--title", default=DEFAULT_TITLE)

    # ledger
    ledger = sub(commands, "ledger", "the record of baselines, promotions and held-out evaluations")
    ledger_actions = ledger.add_subparsers(dest="action", required=True, metavar="ACTION")
    verify = sub(ledger_actions, "verify", "check the hash chain and the current baselines' fingerprints")
    ledger_arg(verify)
    status = sub(ledger_actions, "status", "current baselines, held-out openings, problems")
    ledger_arg(status)
    current = sub(ledger_actions, "current", "print the current baseline directory of a feature")
    current.add_argument("--feature", required=True, choices=FEATURES)
    ledger_arg(current)
    promote = sub(ledger_actions, "promote", "make a gated candidate the feature's baseline")
    promote.add_argument("--feature", required=True, choices=FEATURES)
    promote.add_argument("--to", required=True, metavar="DIR", help="the candidate's cassette set")
    promote.add_argument("--report", metavar="FILE", help="the gate report that passed it (required unless --initial)")
    promote.add_argument("--reason", default="", help="why (required with --override and --initial)")
    promote.add_argument("--override", action="store_true", help="promote although the report does not allow it (recorded as such)")
    promote.add_argument("--initial", action="store_true", help="the feature's first baseline: no report to check")
    ledger_arg(promote)
    rollback = sub(ledger_actions, "rollback", "undo the promotion in force")
    rollback.add_argument("--feature", required=True, choices=FEATURES)
    rollback.add_argument("--reason", required=True)
    ledger_arg(rollback)

    return parser


# ---- the commands ----------------------------------------------------------------------------
# Each handler returns (exit status, the document to print on stdout). main() maps exceptions to
# exit statuses and does the stdout discipline described in OUTPUT.

Handler = Callable[[argparse.Namespace], "tuple[int, str]"]

#: Exceptions that mean "refused" (exit 2) and "could not run" (exit 1). Subclasses come first
#: where the two overlap: LedgerRefused is a LedgerError, the replay errors are ClientErrors.
REFUSALS: tuple[type[BaseException], ...] = (
    RefusedError, GateRefused, LedgerRefused, CassetteMismatch, MissingRecording, SuiteError, PromptError, CassetteError,
    ReportError, util.JsonlError,
)
FAILURES: tuple[type[BaseException], ...] = (RunError, ClientError, ScoringError, LedgerError, SandboxUnavailable, OSError)


def json_document(obj: Any) -> str:
    """The JSON documents the commands print: indented, sorted keys, one final newline."""
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _progress(args: argparse.Namespace) -> Callable[[str], None]:
    """Where progress lines go: stderr, or nowhere under --quiet (warnings and errors always go)."""
    return (lambda message: None) if getattr(args, "quiet", False) else util.log


def cmd_tasks_validate(args: argparse.Namespace) -> tuple[int, str]:
    """validation.validate_suite; JSON report; exit 1 when the suite is not valid."""
    suite = tasks.load_suite(args.suite)
    say = _progress(args)
    say(f"validating {suite.source} ({len(suite.tasks)} tasks{'' if not args.no_exec else ', no execution'})")
    report = validation.validate_suite(suite, execute=not args.no_exec, timeout_s=args.timeout)
    for warning in report.warnings:
        util.log(f"warning: {warning}")
    for problem in report.problems:
        util.log(f"problem: {problem}")
    say(f"{suite.name}: {'valid' if report.valid else 'NOT valid'} ({len(report.problems)} problem(s), {len(report.warnings)} warning(s))")
    return (EXIT_OK if report.valid else EXIT_ERROR), json_document(report.to_json())


def cmd_tasks_split(args: argparse.Namespace) -> tuple[int, str]:
    """{"suite", "sha256", "heldout_pct", "dev": [ids], "heldout": [ids], "buckets": {id: bucket}}."""
    suite = tasks.load_suite(args.suite)
    document = {
        "suite": suite.name,
        "sha256": suite.sha256,
        "heldout_pct": suite.heldout_pct,
        DEV: [t.id for t in tasks.tasks_in_split(suite, DEV)],
        HELDOUT: [t.id for t in tasks.tasks_in_split(suite, HELDOUT)],
        "buckets": {t.id: tasks.bucket(t.group or t.id) for t in sorted(suite.tasks, key=lambda t: t.id)},
    }
    return EXIT_OK, json_document(document)


def cmd_run(args: argparse.Namespace) -> tuple[int, str]:
    """runner.evaluate_arm; {"arm": ArmRun.summary(), "feature", "split", "suite"} as JSON.
    Logs a reminder to stderr when the split is heldout: every look at it spends it."""
    suite = tasks.load_suite(args.suite)
    if args.split == HELDOUT:
        util.log("reminder: the held-out split is the gate's, and every look at it spends it; iterate on dev")
    arm = runner.evaluate_arm(
        args.feature, suite, args.split, args.cassettes,
        workers=args.workers, timeout_s=args.timeout, allow_missing=args.allow_missing, progress=_progress(args),
    )
    if args.items_out:
        util.write_jsonl(args.items_out, [r.to_json() for r in arm.results])
        _progress(args)(f"wrote {len(arm.results)} scored item(s) to {util.display_path(args.items_out)}")
    document = {"arm": arm.summary(), "feature": args.feature, "split": args.split, "suite": runner.suite_block(suite, args.split, args.feature)}
    return EXIT_OK, json_document(document)


def cmd_record(args: argparse.Namespace) -> tuple[int, str]:
    """clients.client_from_env + runner.record_arm; RecordSummary as JSON; exit 1 if any failed."""
    suite = tasks.load_suite(args.suite)
    feature = get_feature(args.feature)
    if args.prompt_from:
        other = cassettes.load_cassettes(args.prompt_from)
        tape = other.features.get(args.feature)
        if tape is None:
            raise RefusedError(f"{other.source} holds no {args.feature} recordings to take the prompt from")
        template = prompts.get_template(tape.prompt, args.feature)
        base = tape.settings
    else:
        template = prompts.get_template(args.prompt, args.feature) if args.prompt else prompts.current_template(args.feature)
        base = feature.default_settings
    settings = ModelSettings(
        temperature=base.temperature if args.temperature is None else args.temperature,
        max_tokens=base.max_tokens if args.max_tokens is None else args.max_tokens,
        seed=args.seed if getattr(args, "seed_given", False) else base.seed,
        extra=dict(base.extra) if args.extra is None else args.extra,
    )
    if not (args.base_url or os.environ.get(BASE_URL_ENV)):
        raise RefusedError(f"no endpoint: pass --base-url or set ${BASE_URL_ENV}")
    try:
        client = client_from_env(args.base_url, api_key_env=args.api_key_env, timeout_s=args.request_timeout, max_retries=args.max_retries)
    except ValueError as exc:
        raise RefusedError(f"--base-url: {exc}") from None
    arm = args.arm or Path(args.out).resolve().name or "arm"
    summary = runner.record_arm(
        args.feature, suite, args.split, client, args.out,
        arm=arm, model=args.model, template=template, settings=settings,
        workers=args.workers, description=args.description, progress=_progress(args),
    )
    return (EXIT_ERROR if summary.failed else EXIT_OK), json_document(summary.to_json())


def resolve_baseline(value: str, feature: str, ledger: Ledger | None) -> str | None:
    """--baseline: a directory, "@current" (the ledger's current baseline) or "none"."""
    if value == BASELINE_NONE:
        return None
    if value == BASELINE_CURRENT:
        current = ledger.current(feature) if ledger is not None else None
        if current is None:
            raise RefusedError(f"the ledger names no current {feature} baseline: pass --baseline DIR or --baseline none")
        return current
    return value


def cmd_gate(args: argparse.Namespace) -> tuple[int, str]:
    """GateConfig from the flags, the baseline resolved (@current / none), runner.gate_run;
    report.dumps_report(report)."""
    suite = tasks.load_suite(args.suite)
    config = GateConfig(
        schema_min=args.schema_min,
        think_min=args.think_min,
        max_new_sast=args.max_new_sast,
        margin=args.margin,
        tie=args.tie,
        alpha=args.alpha,
        require_gain=args.require_gain,
        floor=args.floor,
    )
    needs_ledger = args.baseline == BASELINE_CURRENT or args.split == HELDOUT or args.record_evaluation
    ledger = Ledger.open(args.ledger) if needs_ledger else None
    baseline = resolve_baseline(args.baseline, args.feature, ledger)
    report = runner.gate_run(
        args.feature, suite, args.split, args.candidate, baseline, config,
        resamples=args.resamples, seed=args.seed, workers=args.workers, timeout_s=args.timeout,
        allow_missing=args.allow_missing, items_out=args.items_out, ledger=ledger,
        record_evaluation=args.record_evaluation, progress=_progress(args),
    )
    return EXIT_OK, report_mod.dumps_report(report)


def cmd_summary(args: argparse.Namespace) -> tuple[int, str]:
    """report.markdown_summary of the loaded reports."""
    reports = [report_mod.load_report(path) for path in args.reports]
    return EXIT_OK, report_mod.markdown_summary(reports, title=args.title)


def cmd_ledger(args: argparse.Namespace) -> tuple[int, str]:
    """verify / status / current / promote / rollback (see ledger.py)."""
    ledger = Ledger.open(args.ledger)
    say = _progress(args)
    if args.action == "verify":
        problems = ledger.verify()
        for problem in problems:
            util.log(f"problem: {problem}")
        say(f"{util.display_path(args.ledger)}: {len(ledger.rows)} row(s), {'verified' if not problems else 'NOT verified'}")
        document = {"ledger": util.display_path(args.ledger), "rows": len(ledger.rows), "problems": problems, "valid": not problems}
        return (EXIT_ERROR if problems else EXIT_OK), json_document(document)
    if args.action == "status":
        return EXIT_OK, json_document(ledger.status())
    if args.action == "current":
        current = ledger.current(args.feature)
        if current is None:
            raise RefusedError(f"{util.display_path(args.ledger)} names no current {args.feature} baseline")
        return EXIT_OK, f"{current}\n"
    if args.action == "promote":
        if args.initial and args.report:
            raise RefusedError("--initial starts a feature without a report: drop --report, or drop --initial")
        if not args.initial and not args.report:
            raise RefusedError("--report is required (the gate report that passed the candidate), unless --initial")
        if (args.override or args.initial) and not args.reason.strip():
            raise RefusedError(f"{'--override' if args.override else '--initial'} needs a --reason")
        report = report_mod.load_report(args.report) if args.report else None
        report_sha256 = util.sha256_file(args.report) if args.report else None
        row = ledger.promote(
            args.feature, args.to, report,
            reason=args.reason, report_sha256=report_sha256, override=args.override, initial=args.initial,
        )
        say(f"{args.feature}: the baseline is now {util.display_path(args.to)}")
        return EXIT_OK, json_document(row)
    if args.action == "rollback":
        row = ledger.rollback(args.feature, reason=args.reason)
        say(f"{args.feature}: rolled back")
        return EXIT_OK, json_document(row)
    raise RefusedError(f"unknown ledger action {args.action!r}")


HANDLERS: dict[str, Handler] = {
    "tasks validate": cmd_tasks_validate,
    "tasks split": cmd_tasks_split,
    "run": cmd_run,
    "record": cmd_record,
    "gate": cmd_gate,
    "summary": cmd_summary,
    "ledger": cmd_ledger,
}


def _handler_key(args: argparse.Namespace) -> str:
    return f"tasks {args.action}" if args.command == "tasks" else args.command


def _message(exc: BaseException) -> str:
    """An exception as one line (KeyError's str() would add quotes)."""
    text = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
    return " ".join(text.split()) or type(exc).__name__


@contextlib.contextmanager
def stdout_to_stderr() -> Iterator[None]:
    """While the block runs, file descriptor 1 and sys.stdout point at stderr, so nothing a
    library or a child process writes can reach stdout. Restored afterwards, whatever happened."""
    stdout = sys.stdout
    with contextlib.suppress(Exception):
        stdout.flush()
    saved: int | None = None
    try:
        saved = os.dup(1)
        os.dup2(2, 1)
    except OSError:
        if saved is not None:
            os.close(saved)
        saved = None
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            sys.stderr.flush()
        sys.stdout = stdout
        if saved is not None:
            os.dup2(saved, 1)
            os.close(saved)


def _emit(document: str) -> None:
    """Write the one document to the real stdout, as UTF-8 whatever the locale says."""
    stream = sys.stdout
    if stream is None:  # started with no stdout at all: there is nowhere to print
        return
    try:
        buffer = getattr(stream, "buffer", None)
        if buffer is not None:
            stream.flush()
            buffer.write(document.encode("utf-8"))
            buffer.flush()
        else:
            stream.write(document)
            stream.flush()
    except BrokenPipeError:
        # The reader went away (`llm-eval ... | head`): drop the rest quietly.
        with contextlib.suppress(OSError):
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.close(devnull)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse `argv` (default sys.argv[1:]), run one handler under the OUTPUT discipline, print
    its document on the real stdout, and return the exit status (EXIT STATUS above): RefusedError,
    GateRefused, LedgerRefused, CassetteMismatch, MissingRecording, SuiteError and
    PromptError -> 2; RunError, ClientError, ScoringError, LedgerError and OSError -> 1, each
    with a one-line message on stderr and no traceback."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(arguments)
    except SystemExit as exc:  # argparse: --help / --version (0), a usage error (2)
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else EXIT_USAGE)
    # --seed none and no --seed both parse to None; record must tell them apart.
    args.seed_given = any(a == "--seed" or a.startswith("--seed=") for a in arguments)
    handler = HANDLERS[_handler_key(args)]
    status, document = EXIT_ERROR, None
    with stdout_to_stderr():
        try:
            status, document = handler(args)
        except REFUSALS as exc:
            status = EXIT_USAGE
            util.log(f"refused: {_message(exc)}")
        except FAILURES as exc:
            status = EXIT_ERROR
            util.log(f"error: {_message(exc)}")
        except KeyboardInterrupt:
            status = 130
            util.log("interrupted")
    if document is not None:
        _emit(document)
    return status
