"""The `llm-eval` command: one document on stdout, logs on stderr, and the exit statuses.

Most tests run the command as CI does, in a child process (python -m llm_eval_harness with
PYTHONPATH=src), in a temporary directory holding the tiny suite and three cassette sets
written here: a baseline (prompts v1, overfitted fixes, a judge that approves everything), a
candidate (current prompts, the reference fixes, which also print when they are imported) and
a regressed candidate (text instead of JSON).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import pytest

from conftest import TINY_TASKS, write_suite
from llm_eval_harness import cli, prompts, tasks
from llm_eval_harness.cassettes import CassetteWriter
from llm_eval_harness.clients import CassetteMismatch, ClientError, MissingRecording
from llm_eval_harness.features import FEATURES, ScoringError
from llm_eval_harness.gate import GateRefused
from llm_eval_harness.ledger import LedgerError, LedgerRefused
from llm_eval_harness.prompts import PromptError
from llm_eval_harness.records import ALL, ChatResponse, Item
from llm_eval_harness.report import dumps_report
from llm_eval_harness.runner import RefusedError, RunError
from llm_eval_harness.tasks import SuiteError

SRC = Path(__file__).resolve().parents[1] / "src"
PROXY_VARIABLES = {"http_proxy", "https_proxy", "all_proxy"}
NOISE = "import sys\nprint('noise from the patch on stdout')\nprint('noise from the patch on stderr', file=sys.stderr)\n\n\n"


def llm_eval(*args: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    environment = {k: v for k, v in os.environ.items() if not k.startswith("LLM_EVAL_") and k.lower() not in PROXY_VARIABLES}
    environment.update({"PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1", **(env or {})})
    return subprocess.run([sys.executable, "-m", "llm_eval_harness", *args], cwd=cwd, env=environment,
                          capture_output=True, text=True, timeout=300)


def answer_json(content: dict[str, Any]) -> ChatResponse:
    return ChatResponse(content=json.dumps(content), finish_reason="stop", usage={"completion_tokens": 20}, latency_s=0.2)


def baseline_answer(item: Item) -> ChatResponse:
    return answer_json({"explanation": "x", "patched_code": item.task.wrong_code} if item.feature == "bugfix" else {"verdict": True})


def candidate_answer(item: Item) -> ChatResponse:
    if item.feature == "bugfix":
        return answer_json({"explanation": "x", "patched_code": NOISE + item.task.reference_code})
    return answer_json({"verdict": bool(item.label)})


def regressed_answer(item: Item) -> ChatResponse:
    return ChatResponse(content="The patch looks fine to me.", finish_reason="stop")


def write_arm(root: Path, suite: Any, answer: Callable[[Item], ChatResponse], *, prompt_version: str | None = None,
              skip: frozenset[str] = frozenset()) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, feature in FEATURES.items():
        template = prompts.get_template(f"{name}/{prompt_version}") if prompt_version else prompts.current_template(name)
        settings = feature.default_settings
        writer = CassetteWriter(root, arm=root.name, feature=name, prompt=template.id, model="tiny-model", settings=settings,
                                suite_name=suite.name, suite_sha256=suite.sha256, synthetic=True)
        for item in feature.items(suite, ALL):
            if item.id not in skip:
                writer.write(prompts.build_request(template, item, "tiny-model", settings), answer(item))
        writer.close()


@pytest.fixture
def ws(tmp_path: Path, tiny_suite_dir: Path) -> Path:
    """A working directory: tiny-suite/ and cassettes/{baseline,candidate,regressed,partial}."""
    suite = tasks.load_suite(tiny_suite_dir)
    write_arm(tmp_path / "cassettes" / "baseline", suite, baseline_answer, prompt_version="v1")
    write_arm(tmp_path / "cassettes" / "candidate", suite, candidate_answer)
    write_arm(tmp_path / "cassettes" / "regressed", suite, regressed_answer)
    write_arm(tmp_path / "cassettes" / "partial", suite, candidate_answer, skip=frozenset({"bugfix:t-parse-port"}))
    return tmp_path


def gate(ws: Path, *extra: str, feature: str = "bugfix", candidate: str = "candidate", baseline: str = "cassettes/baseline",
         split: str = "dev") -> subprocess.CompletedProcess[str]:
    return llm_eval("gate", "--feature", feature, "--suite", "tiny-suite", "--split", split, "--candidate", f"cassettes/{candidate}",
                    "--baseline", baseline, "--resamples", "200", "--ledger", "ledger/ledger.jsonl", *extra, cwd=ws)


def only_log_lines(stderr: str) -> bool:
    return all(line.startswith("llm-eval: ") for line in stderr.splitlines())


# ---- stdout discipline ------------------------------------------------------------------------------


def test_gate_prints_exactly_one_report_even_when_the_patched_code_prints(ws: Path) -> None:
    result = gate(ws, "--items-out", "items")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)  # one document: json.loads refuses anything after it
    assert result.stdout == dumps_report(report)
    assert report["promote"] is True and report["candidate"]["source"] == "cassettes/candidate"
    assert "noise" not in result.stdout + result.stderr
    assert "llm-eval: " in result.stderr and only_log_lines(result.stderr), result.stderr
    assert (ws / "items" / "bugfix.candidate.items.jsonl").is_file() and (ws / "items" / "bugfix.baseline.items.jsonl").is_file()


def test_quiet_drops_the_progress_lines(ws: Path) -> None:
    loud = gate(ws, feature="judge")
    quiet = llm_eval("--quiet", "gate", "--feature", "judge", "--suite", "tiny-suite", "--split", "dev", "--candidate", "cassettes/candidate",
                     "--baseline", "cassettes/baseline", "--resamples", "200", cwd=ws)
    assert loud.returncode == quiet.returncode == 0
    assert loud.stdout == quiet.stdout
    assert "scoring" in loud.stderr and "scoring" not in quiet.stderr


def test_main_sends_everything_but_the_document_to_stderr(capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    def noisy(args: Any) -> tuple[int, str]:
        print("print noise")
        os.write(1, b"fd noise\n")
        subprocess.run([sys.executable, "-c", "print('child noise')"], check=True)
        return 0, "THE DOCUMENT\n"

    monkeypatch.setitem(cli.HANDLERS, "summary", noisy)
    assert cli.main(["summary", "report.json"]) == 0
    out, err = capfd.readouterr()
    assert out == "THE DOCUMENT\n"
    assert "print noise" in err and "fd noise" in err and "child noise" in err
    print("stdout works again")
    assert capfd.readouterr().out == "stdout works again\n"


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (RefusedError("refused"), 2), (GateRefused("not comparable"), 2), (LedgerRefused("stale report"), 2),
        (MissingRecording("no recording"), 2), (CassetteMismatch("other request"), 2), (SuiteError("bad line"), 2),
        (PromptError("bugfix/v9"), 2), (RunError("stopped"), 1), (ClientError("endpoint down"), 1),
        (ScoringError("no sandbox"), 1), (LedgerError("broken chain"), 1), (OSError("disk full"), 1),
    ],
)
def test_errors_become_exit_statuses_with_one_line_and_no_traceback(
    error: BaseException, status: int, capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(args: Any) -> tuple[int, str]:
        raise error

    monkeypatch.setitem(cli.HANDLERS, "summary", failing)
    assert cli.main(["summary", "report.json"]) == status
    out, err = capfd.readouterr()
    assert out == "" and len(err.splitlines()) == 1 and "Traceback" not in err
    if isinstance(error, PromptError):
        assert err.strip().endswith("bugfix/v9")


@pytest.mark.parametrize(
    "command",
    ["tasks validate", "tasks split", "run --feature bugfix --cassettes c", "record --feature judge --model m --out o",
     "gate --feature judge --candidate c --baseline none", "summary r.json", "ledger verify", "ledger status",
     "ledger current --feature bugfix", "ledger rollback --feature bugfix --reason r"],
)
def test_every_command_has_a_handler(command: str) -> None:
    args = cli.build_parser().parse_args(command.split())
    assert cli._handler_key(args) in cli.HANDLERS


def test_usage_errors_and_version(capfd: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["gate", "--feature", "bugfix"]) == 2
    assert cli.main(["--version"]) == 0
    out, _err = capfd.readouterr()
    assert out.startswith("llm-eval ")


# ---- exit statuses -----------------------------------------------------------------------------------


def test_exit_statuses(ws: Path) -> None:
    valid = llm_eval("tasks", "validate", "--suite", "tiny-suite", cwd=ws)
    assert valid.returncode == 0 and json.loads(valid.stdout)["valid"] is True, valid.stderr
    unexecuted = llm_eval("tasks", "validate", "--suite", "tiny-suite", "--no-exec", cwd=ws)
    assert unexecuted.returncode == 0 and json.loads(unexecuted.stdout)["executed"] is False

    correct_wrong_code = 'def clamp(x, lo, hi):\n    """Return x limited to the closed interval [lo, hi]."""\n    return min(max(x, lo), hi)\n'
    wrong_passes = [dict(t, wrong_code=correct_wrong_code) if t["id"] == "t-clamp" else t for t in TINY_TASKS]
    write_suite(ws / "broken-suite", wrong_passes)
    invalid = llm_eval("tasks", "validate", "--suite", "broken-suite", cwd=ws)
    assert invalid.returncode == 1 and json.loads(invalid.stdout)["problems"] == ["t-clamp: the wrong code passes every hidden test, so it is not wrong"]
    assert "problem: t-clamp" in invalid.stderr

    regressed = gate(ws, candidate="regressed")
    assert regressed.returncode == 0 and json.loads(regressed.stdout)["promote"] is False

    for refused in (
        llm_eval("tasks", "validate", "--suite", "no-such-suite", cwd=ws),
        gate(ws, baseline="@current"),  # the ledger names no baseline
        gate(ws, candidate="partial"),  # an item without a recording
        llm_eval("gate", "--feature", "bugfix", "--candidate", "cassettes/candidate", cwd=ws),  # no --baseline
    ):
        assert refused.returncode == 2 and refused.stdout == "", refused.stderr
        assert "Traceback" not in refused.stderr
    allowed = gate(ws, "--allow-missing", candidate="partial")
    assert allowed.returncode == 0 and json.loads(allowed.stdout)["comparison"]["n"] == 2


# ---- the ledger --------------------------------------------------------------------------------------


def test_promotion_through_the_ledger(ws: Path) -> None:
    ledger = ("--ledger", "ledger/ledger.jsonl")
    assert llm_eval("ledger", "current", "--feature", "bugfix", *ledger, cwd=ws).returncode == 2

    first = llm_eval("ledger", "promote", "--feature", "bugfix", "--to", "cassettes/baseline", "--initial", "--reason", "first", *ledger, cwd=ws)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["kind"] == "promote"
    current = llm_eval("ledger", "current", "--feature", "bugfix", *ledger, cwd=ws)
    assert (current.returncode, current.stdout) == (0, "cassettes/baseline\n")

    held_out = gate(ws, baseline="@current", split="heldout")
    assert held_out.returncode == 0, held_out.stderr
    report = json.loads(held_out.stdout)
    assert report["promote"] is True and report["baseline"]["source"] == "cassettes/baseline"
    (ws / "report.json").write_text(held_out.stdout, encoding="utf-8")
    promoted = llm_eval("ledger", "promote", "--feature", "bugfix", "--to", "cassettes/candidate", "--report", "report.json",
                        "--reason", "better", *ledger, cwd=ws)
    assert promoted.returncode == 0, promoted.stderr
    assert llm_eval("ledger", "current", "--feature", "bugfix", *ledger, cwd=ws).stdout == "cassettes/candidate\n"

    verified = llm_eval("ledger", "verify", *ledger, cwd=ws)
    assert verified.returncode == 0 and json.loads(verified.stdout)["valid"] is True, verified.stdout
    status = llm_eval("ledger", "status", *ledger, cwd=ws)
    assert status.returncode == 0 and isinstance(json.loads(status.stdout), dict)
    refused = llm_eval("ledger", "promote", "--feature", "bugfix", "--to", "cassettes/regressed", *ledger, cwd=ws)
    assert refused.returncode == 2  # no report

    rolled = llm_eval("ledger", "rollback", "--feature", "bugfix", "--reason", "a regression in production", *ledger, cwd=ws)
    assert rolled.returncode == 0 and json.loads(rolled.stdout)["kind"] == "rollback", rolled.stderr
    assert llm_eval("ledger", "current", "--feature", "bugfix", *ledger, cwd=ws).stdout == "cassettes/baseline\n"
    assert llm_eval("ledger", "verify", *ledger, cwd=ws).returncode == 0


def test_gate_can_record_an_evaluation(ws: Path) -> None:
    result = gate(ws, "--record-evaluation", split="heldout")
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in (ws / "ledger" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["kind"] == "evaluation" and rows[-1]["split_sha256"] == json.loads(result.stdout)["suite"]["split_sha256"]


# ---- summary, run, split -----------------------------------------------------------------------------


def test_summary_renders_the_reports_as_markdown(ws: Path) -> None:
    (ws / "good.json").write_text(gate(ws).stdout, encoding="utf-8")
    (ws / "bad.json").write_text(gate(ws, candidate="regressed").stdout, encoding="utf-8")
    result = llm_eval("summary", "good.json", "bad.json", "--title", "Tiny gate", cwd=ws)
    assert result.returncode == 0, result.stderr
    assert "Tiny gate" in result.stdout.splitlines()[0]
    assert "PROMOTE" in result.stdout and "REJECT" in result.stdout and "| " in result.stdout
    (ws / "not-a-report.json").write_text('{"hello": 1}\n', encoding="utf-8")
    assert llm_eval("summary", "not-a-report.json", cwd=ws).returncode == 2


def test_run_prints_one_arm_and_reminds_about_the_held_out_split(ws: Path) -> None:
    result = llm_eval("run", "--feature", "judge", "--cassettes", "cassettes/candidate", "--suite", "tiny-suite", "--split", "heldout",
                      "--items-out", "run.items.jsonl", cwd=ws)
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert set(document) == {"arm", "feature", "split", "suite"} and document["arm"]["name"] == "candidate"
    assert document["suite"]["items"] == 2 and "reminder" in result.stderr
    assert len((ws / "run.items.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    dev = llm_eval("run", "--feature", "bugfix", "--cassettes", "cassettes/candidate", "--suite", "tiny-suite", cwd=ws)
    assert dev.returncode == 0 and json.loads(dev.stdout)["split"] == "dev" and "reminder" not in dev.stderr


def test_tasks_split_lists_both_sides(ws: Path) -> None:
    result = llm_eval("tasks", "split", "--suite", "tiny-suite", cwd=ws)
    document = json.loads(result.stdout)
    assert document["heldout"] == ["t-clamp"] and document["dev"] == ["t-count-vowels", "t-parse-port", "t-running-total"]
    assert set(document["buckets"]) == {t["id"] for t in TINY_TASKS} and document["buckets"]["t-clamp"] < 50


# ---- record --------------------------------------------------------------------------------------------


@pytest.fixture
def stub() -> Iterator[SimpleNamespace]:
    """An OpenAI-compatible endpoint on 127.0.0.1: a "true" verdict for everything, HTTP 400 for parse_port."""
    seen: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"authorization": self.headers.get("Authorization"), "body": body})
            refused = any("def parse_port(" in m["content"] for m in body["messages"])
            payload = ({"error": {"message": "refused"}} if refused else
                       {"choices": [{"message": {"role": "assistant", "content": '{"verdict": true}'}, "finish_reason": "stop"}]})
            data = json.dumps(payload).encode("utf-8")
            self.send_response(400 if refused else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_address[1]}/v1", seen=seen)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_record_reads_the_key_from_the_named_variable_and_takes_a_prompt_from_another_set(ws: Path, stub: SimpleNamespace) -> None:
    common = ("--feature", "judge", "--model", "stub-model", "--suite", "tiny-suite", "--split", "dev", "--base-url", stub.url,
              "--api-key-env", "MY_TEST_KEY", "--max-retries", "0")
    first = llm_eval("record", *common, "--out", "cassettes/live", "--prompt", "judge/v1", "--seed", "none", cwd=ws, env={"MY_TEST_KEY": "key-abc"})
    assert first.returncode == 1, first.stderr  # the two parse_port items failed
    summary = json.loads(first.stdout)
    assert (summary["arm"], summary["recorded"], summary["failed"], summary["prompt"]) == ("live", 4, 2, "judge/v1")
    assert {s["authorization"] for s in stub.seen} == {"Bearer key-abc"}
    assert all("seed" not in s["body"] for s in stub.seen)
    assert "key-abc" not in first.stdout + first.stderr

    second = llm_eval("record", *common, "--out", "cassettes/again", "--prompt-from", "cassettes/live", "--arm", "again-arm", cwd=ws,
                      env={"MY_TEST_KEY": "key-abc"})
    manifest = json.loads((ws / "cassettes" / "again" / "manifest.json").read_text(encoding="utf-8"))
    live = json.loads((ws / "cassettes" / "live" / "manifest.json").read_text(encoding="utf-8"))
    assert second.returncode == 1 and manifest["arm"] == "again-arm"
    assert manifest["features"]["judge"]["prompt"] == "judge/v1"
    assert manifest["features"]["judge"]["settings"] == live["features"]["judge"]["settings"]

    no_endpoint = llm_eval("record", "--feature", "judge", "--model", "m", "--out", "cassettes/none", "--suite", "tiny-suite", cwd=ws)
    assert no_endpoint.returncode == 2 and no_endpoint.stdout == ""
