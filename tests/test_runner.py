"""runner.py: replaying and scoring an arm, recording one from a live endpoint, the gate run.

The cassette sets are written here, with cassettes.CassetteWriter and prompts.build_request over
the tiny suite, so their request hashes are exactly the ones the runner rebuilds. Answers come
from the tasks' own code: the reference fix, the overfitted wrong fix, a patch that uses eval(),
text that is not JSON, and a reply cut off at max_tokens.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import pytest

from conftest import TINY_TASKS, write_suite
from llm_eval_harness import __version__, cassettes, prompts, runner, sandbox, sast, tasks, verdicts
from llm_eval_harness.cassettes import CassetteWriter
from llm_eval_harness.clients import ClientError, OpenAICompatClient
from llm_eval_harness.features import FEATURES
from llm_eval_harness.gate import GateConfig, GateRefused
from llm_eval_harness.ledger import Ledger
from llm_eval_harness.records import ALL, DEV, HELDOUT, ChatRequest, ChatResponse, Item, ModelSettings, Suite
from llm_eval_harness.report import dumps_report
from llm_eval_harness.runner import RecordSummary, RefusedError, RunError

MODEL = "tiny-model"
Answer = Callable[[Item], ChatResponse]
RUNNING_TOTAL_EVAL = (
    "def running_total(xs):\n"
    '    """Return the prefix sums of xs: element i is xs[0] + ... + xs[i]."""\n'
    '    return eval("[sum(xs[:i + 1]) for i in range(len(xs))]")\n'
)


# ---- answers and cassette sets ---------------------------------------------------------------------


def fix(code: str, *, think: str | None = None, finish: str = "stop") -> ChatResponse:
    content = json.dumps({"explanation": "Fixed.", "patched_code": code})
    if think:
        content = f"<think>{think}</think>{content}"
    return ChatResponse(content=content, finish_reason=finish, usage={"prompt_tokens": 100, "completion_tokens": 30}, latency_s=0.5)


def verdict(value: bool) -> ChatResponse:
    return ChatResponse(content=json.dumps({"verdict": value}), finish_reason="stop", usage={"completion_tokens": 5}, latency_s=0.1)


def good(item: Item) -> ChatResponse:
    """The reference fix; the right verdict."""
    return fix(item.task.reference_code, think="The bug is in the last line.") if item.feature == "bugfix" else verdict(bool(item.label))


def overfit(item: Item) -> ChatResponse:
    """The wrong fix (passes the shown test only); a judge that approves everything."""
    return fix(item.task.wrong_code) if item.feature == "bugfix" else verdict(True)


def broken(item: Item) -> ChatResponse:
    """Text that is not the JSON object the feature asks for."""
    return ChatResponse(content="I would change the comparison on the last line.", finish_reason="stop")


def mixed(item: Item) -> ChatResponse:
    """One of everything: resolved, overfit, blocked, not JSON, truncated."""
    task = item.task
    if item.feature == "judge":
        return verdict(bool(item.label)) if task.id != "t-parse-port" else broken(item)
    return {
        "t-clamp": lambda: fix(task.reference_code, think="Use hi as the upper bound."),
        "t-count-vowels": lambda: fix(task.wrong_code),
        "t-running-total": lambda: fix(RUNNING_TOTAL_EVAL),
        "t-parse-port": lambda: fix(task.reference_code, finish="length"),
    }[task.id]()


def write_arm(
    root: str | Path,
    suite: Suite,
    feature: str,
    answer: Answer,
    *,
    arm: str | None = None,
    prompt: str | None = None,
    model: str = MODEL,
    settings: ModelSettings | None = None,
    synthetic: bool = True,
    skip: frozenset[str] = frozenset(),
) -> Path:
    """A cassette set holding `answer(item)` for every item of `feature` (except `skip`)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    template = prompts.get_template(prompt, feature) if prompt else prompts.current_template(feature)
    settings = settings or FEATURES[feature].default_settings
    writer = CassetteWriter(root, arm=arm or root.name, feature=feature, prompt=template.id, model=model, settings=settings,
                            suite_name=suite.name, suite_sha256=suite.sha256, synthetic=synthetic)
    for item in FEATURES[feature].items(suite, ALL):
        if item.id not in skip:
            writer.write(prompts.build_request(template, item, model, settings), answer(item))
    writer.close()
    return root


def stable(arm: Any) -> list[tuple[Any, ...]]:
    """What scoring decided per item, without timings."""
    return [(r.item_id, r.correct, r.schema_valid, r.lenient_valid, r.think_valid, r.shown_passed, r.hidden_passed,
             r.sast_block, r.sast_flag, r.predicted, r.problems) for r in arm.results]


@pytest.fixture(autouse=True)
def in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run in the test's directory, so display paths are relative ones."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def suite(tiny_suite_dir: Path) -> Suite:
    return tasks.load_suite(tiny_suite_dir)


def edited_suite(tmp_path: Path, task_id: str, **changes: Any) -> Suite:
    edited = [dict(t, **changes) if t["id"] == task_id else t for t in TINY_TASKS]
    return tasks.load_suite(write_suite(tmp_path / f"edited-{len(list(tmp_path.iterdir()))}", edited))


# ---- evaluate_arm --------------------------------------------------------------------------------------


def test_an_arm_is_replayed_scored_and_described(suite: Suite) -> None:
    root = write_arm("cassettes/good", suite, "bugfix", good)
    arm = runner.evaluate_arm("bugfix", suite, ALL, root, workers=2)
    assert (arm.name, arm.feature, arm.split, arm.source) == ("good", "bugfix", ALL, "cassettes/good")
    assert arm.fingerprint == cassettes.fingerprint(root)
    assert (arm.model, arm.prompt, arm.prompt_sha256) == (MODEL, prompts.CURRENT["bugfix"], prompts.current_template("bugfix").sha256())
    assert arm.synthetic is True and arm.suite_sha256 == suite.sha256 == arm.recorded_suite_sha256 and arm.notes == []
    assert [r.item_id for r in arm.results] == sorted(f"bugfix:{t['id']}" for t in TINY_TASKS)
    assert all(r.correct and r.think_valid for r in arm.results)
    assert arm.metrics["resolved_rate"] == 1.0
    assert arm.metrics["by_tag"] == {tag: {"items": 1, "resolved": 1} for t in TINY_TASKS for tag in t["tags"]}
    assert {r.split for r in arm.results} == {DEV, HELDOUT}
    assert [r.item_id for r in runner.evaluate_arm("bugfix", suite, HELDOUT, root).results] == ["bugfix:t-clamp"]


def test_the_result_does_not_depend_on_the_number_of_workers(suite: Suite) -> None:
    for feature in FEATURES:
        root = write_arm(f"cassettes/mixed-{feature}", suite, feature, mixed)
        one = runner.evaluate_arm(feature, suite, ALL, root, workers=1)
        four = runner.evaluate_arm(feature, suite, ALL, root, workers=4)
        assert stable(one) == stable(four)
        assert one.metrics == four.metrics
    bugfix = {r.item_id: r for r in runner.evaluate_arm("bugfix", suite, ALL, "cassettes/mixed-bugfix", workers=4).results}
    assert bugfix["bugfix:t-clamp"].correct
    assert bugfix["bugfix:t-count-vowels"].shown_passed and not bugfix["bugfix:t-count-vowels"].hidden_passed
    assert bugfix["bugfix:t-running-total"].sast_block == ("eval-exec",) and bugfix["bugfix:t-running-total"].shown_passed is None
    assert not bugfix["bugfix:t-parse-port"].schema_valid and bugfix["bugfix:t-parse-port"].finish_reason == "length"


def test_missing_recordings_are_refused_and_can_be_dropped(suite: Suite) -> None:
    root = write_arm("cassettes/partial", suite, "bugfix", good, skip=frozenset({"bugfix:t-parse-port"}))
    with pytest.raises(RefusedError, match=r"no recording for 1 item\(s\) \(bugfix:t-parse-port\)"):
        runner.evaluate_arm("bugfix", suite, ALL, root)
    arm = runner.evaluate_arm("bugfix", suite, ALL, root, allow_missing=True)
    assert [r.item_id for r in arm.results] == ["bugfix:t-clamp", "bugfix:t-count-vowels", "bugfix:t-running-total"]
    assert len(arm.notes) == 1 and "bugfix:t-parse-port" in arm.notes[0]
    # Items outside the split need no recording.
    assert len(runner.evaluate_arm("bugfix", suite, HELDOUT, root).results) == 1


def test_a_recording_made_for_another_request_is_refused(suite: Suite, tmp_path: Path) -> None:
    root = write_arm("cassettes/old", suite, "bugfix", good)
    clamp = next(t for t in TINY_TASKS if t["id"] == "t-clamp")
    changed = edited_suite(tmp_path, "t-clamp", buggy_code=clamp["buggy_code"].replace('"""\n', '"""\n    # keep x in range\n', 1))
    with pytest.raises(RefusedError, match=r"1 recording\(s\) answered a different request \(bugfix:t-clamp\)"):
        runner.evaluate_arm("bugfix", changed, ALL, root)
    arm = runner.evaluate_arm("bugfix", changed, ALL, root, allow_missing=True)
    assert "bugfix:t-clamp" not in {r.item_id for r in arm.results} and len(arm.results) == 3


def test_a_recording_on_another_suite_version_with_the_same_prompts_is_rescored_with_a_note(suite: Suite, tmp_path: Path) -> None:
    root = write_arm("cassettes/good", suite, "bugfix", good)
    clamp = next(t for t in TINY_TASKS if t["id"] == "t-clamp")
    drifted = edited_suite(tmp_path, "t-clamp", hidden_tests=[*clamp["hidden_tests"], {"call": "clamp(0, 0, 3)", "expected": "0"}])
    assert drifted.sha256 != suite.sha256
    arm = runner.evaluate_arm("bugfix", drifted, ALL, root)
    assert arm.recorded_suite_sha256 == suite.sha256 and arm.suite_sha256 == drifted.sha256
    assert len(arm.notes) == 1 and "another version of the suite" in arm.notes[0]
    assert all(r.correct for r in arm.results)


def test_a_set_that_is_not_one_or_lacks_the_feature_is_refused(suite: Suite) -> None:
    root = write_arm("cassettes/judge-only", suite, "judge", good)
    with pytest.raises(RefusedError, match="holds no bugfix recordings"):
        runner.evaluate_arm("bugfix", suite, ALL, root)
    Path("empty").mkdir()
    with pytest.raises(RefusedError):
        runner.evaluate_arm("bugfix", suite, ALL, "empty")
    with pytest.raises(RefusedError):
        runner.evaluate_arm("translate", suite, ALL, root)


def test_a_sandbox_that_does_not_start_stops_the_evaluation(suite: Suite, monkeypatch: pytest.MonkeyPatch) -> None:
    root = write_arm("cassettes/good", suite, "bugfix", good)

    def no_child(*args: Any, **kwargs: Any) -> sandbox.SandboxResult:
        return sandbox.SandboxResult(started=False, exit_code=None, timed_out=False, duration_s=0.0, infra_error="fork failed")

    for module in (sandbox, verdicts):
        if hasattr(module, "run_calls"):
            monkeypatch.setattr(module, "run_calls", no_child)
    with pytest.raises(RunError, match="fork failed"):
        runner.evaluate_arm("bugfix", suite, ALL, root, workers=3)
    # The judge runs no code, so it is unaffected.
    write_arm("cassettes/judge", suite, "judge", good)
    assert runner.evaluate_arm("judge", suite, ALL, "cassettes/judge").metrics["balanced_accuracy"] == 1.0


# ---- gate_run -----------------------------------------------------------------------------------------


def gate_arms(suite: Suite, feature: str, candidate: Answer, baseline: Answer) -> tuple[Path, Path]:
    cand = write_arm(f"cassettes/{feature}-candidate", suite, feature, candidate, arm="candidate")
    base = write_arm(f"cassettes/{feature}-baseline", suite, feature, baseline, arm="baseline", prompt=f"{feature}/v1")
    return cand, base


def failed_gates(report: dict[str, Any]) -> set[str]:
    return {g["gate"] for g in report["gates"] if g["enforced"] and not g["passed"]}


@pytest.mark.parametrize("feature", ["bugfix", "judge"])
def test_the_gate_promotes_a_better_candidate(suite: Suite, feature: str) -> None:
    cand, base = gate_arms(suite, feature, good, overfit)
    report = runner.gate_run(feature, suite, DEV, cand, base, GateConfig(), resamples=200, items_out="items")
    assert report["promote"] is True and failed_gates(report) == set()
    assert (report["feature"], report["split"]) == (feature, DEV)
    n_items = 3 if feature == "bugfix" else 6
    assert report["suite"] == {"name": "tiny", "sha256": suite.sha256, "split_sha256": tasks.split_sha256(suite, DEV), "tasks": 3,
                               "items": n_items, "heldout_pct": 50, "synthetic": True}
    assert report["provenance"] == {"harness_version": __version__, "sast": sast.describe(), "exec_rule": verdicts.EXEC_RULE,
                                    "prompt_current": prompts.CURRENT[feature]}
    assert report["candidate"]["name"] == "candidate" and report["baseline"]["prompt"] == f"{feature}/v1"
    assert report["comparison"]["n"] == n_items
    for role in ("candidate", "baseline"):
        rows = [json.loads(line) for line in Path("items", f"{feature}.{role}.items.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [r["item_id"] for r in rows] == sorted(r["item_id"] for r in rows) and len(rows) == n_items
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("feature", ["bugfix", "judge"])
def test_the_gate_rejects_a_worse_candidate(suite: Suite, feature: str) -> None:
    cand, base = gate_arms(suite, feature, broken, good)
    report = runner.gate_run(feature, suite, DEV, cand, base, GateConfig(), resamples=200)
    assert report["promote"] is False
    assert {"primary", "schema_validity"} <= failed_gates(report)


def test_a_blocked_patch_is_a_new_violation(suite: Suite) -> None:
    def eval_on_running_total(item: Item) -> ChatResponse:
        return fix(RUNNING_TOTAL_EVAL) if item.task.id == "t-running-total" else good(item)

    cand, base = gate_arms(suite, "bugfix", eval_on_running_total, good)
    report = runner.gate_run("bugfix", suite, DEV, cand, base, GateConfig(), resamples=200)
    assert report["promote"] is False and "new_sast_violations" in failed_gates(report)


def test_the_report_is_byte_identical_on_a_replay(suite: Suite) -> None:
    cand, base = gate_arms(suite, "judge", mixed, overfit)
    first = dumps_report(runner.gate_run("judge", suite, DEV, cand, base, GateConfig(), resamples=300))
    second = dumps_report(runner.gate_run("judge", suite, DEV, cand, base, GateConfig(), resamples=300, workers=1))
    assert first == second


def test_allow_missing_keeps_the_comparison_paired(suite: Suite) -> None:
    cand = write_arm("cassettes/cand", suite, "bugfix", good, skip=frozenset({"bugfix:t-parse-port"}))
    base = write_arm("cassettes/base", suite, "bugfix", overfit, prompt="bugfix/v1")
    with pytest.raises(RefusedError):
        runner.gate_run("bugfix", suite, DEV, cand, base, GateConfig(), resamples=100)
    report = runner.gate_run("bugfix", suite, DEV, cand, base, GateConfig(), resamples=100, allow_missing=True)
    assert report["comparison"]["n"] == 2
    assert any("bugfix:t-parse-port" in note for note in report["candidate"]["notes"])
    assert any("the candidate has no fitting recording" in note for note in report["baseline"]["notes"])


def test_arms_the_gate_refuses_are_a_refusal(suite: Suite, monkeypatch: pytest.MonkeyPatch) -> None:
    cand, base = gate_arms(suite, "judge", good, good)

    def refuse(candidate: Any, baseline: Any) -> None:
        raise GateRefused("different item sets")

    monkeypatch.setattr(runner.gate_mod, "check_comparable", refuse)
    with pytest.raises(RefusedError, match="different item sets"):
        runner.gate_run("judge", suite, DEV, cand, base, GateConfig())


def test_bad_gate_arguments_are_refused(suite: Suite) -> None:
    cand, base = gate_arms(suite, "judge", good, good)
    with pytest.raises(RefusedError):
        runner.gate_run("judge", suite, ALL, cand, base, GateConfig())
    with pytest.raises(RefusedError):
        runner.gate_run("judge", suite, DEV, cand, base, GateConfig(tie="maybe"))
    with pytest.raises(RefusedError, match="ledger"):
        runner.gate_run("judge", suite, DEV, cand, base, GateConfig(), record_evaluation=True)


def test_without_a_baseline_the_floor_decides(suite: Suite) -> None:
    cand = write_arm("cassettes/cand", suite, "bugfix", good)
    report = runner.gate_run("bugfix", suite, DEV, cand, None, GateConfig(), resamples=100, items_out="items")
    assert report["baseline"] is None and report["comparison"] is None and report["promote"] is True
    primary = next(g for g in report["gates"] if g["gate"] == "primary")
    assert primary["bound"] == 0.5
    assert Path("items/bugfix.candidate.items.jsonl").is_file() and not Path("items/bugfix.baseline.items.jsonl").exists()
    worse = write_arm("cassettes/worse", suite, "bugfix", overfit)
    assert runner.gate_run("bugfix", suite, DEV, worse, None, GateConfig(), resamples=100)["promote"] is False


def test_the_held_out_openings_come_from_the_ledger(suite: Suite, monkeypatch: pytest.MonkeyPatch) -> None:
    cand, base = gate_arms(suite, "bugfix", good, overfit)
    seen: list[dict[str, Any]] = []
    real = runner.gate_mod.report_warnings

    def spy(*args: Any, **kwargs: Any) -> list[str]:
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(runner.gate_mod, "report_warnings", spy)
    path = Path("ledger/ledger.jsonl")
    runner.gate_run("bugfix", suite, HELDOUT, cand, base, GateConfig(), resamples=50, ledger=Ledger.open(path))
    assert seen[-1] == {"current_prompt": prompts.CURRENT["bugfix"], "heldout_openings": None}
    assert not path.exists()
    report = runner.gate_run("bugfix", suite, HELDOUT, cand, base, GateConfig(), resamples=50, ledger=Ledger.open(path), record_evaluation=True)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in rows] == ["header", "evaluation"]
    assert rows[1]["split_sha256"] == report["suite"]["split_sha256"] == tasks.split_sha256(suite, HELDOUT)
    runner.gate_run("bugfix", suite, HELDOUT, cand, base, GateConfig(), resamples=50, ledger=Ledger.open(path))
    assert seen[-1]["heldout_openings"] == 1
    runner.gate_run("bugfix", suite, DEV, cand, base, GateConfig(), resamples=50, ledger=Ledger.open(path))
    assert seen[-1]["heldout_openings"] is None


# ---- record_arm ---------------------------------------------------------------------------------------


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """An OpenAI-compatible endpoint on 127.0.0.1 that answers with the reference fix (or a
    "true" verdict) and refuses every request about parse_port with HTTP 400."""
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    seen: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - the http.server name
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "authorization": self.headers.get("Authorization"), "body": body})
            text = "\n".join(m["content"] for m in body["messages"])
            task = next(t for t in TINY_TASKS if f"def {t['entry_point']}(" in text)
            if task["entry_point"] == "parse_port":
                self.reply(400, {"error": {"message": "the stub refuses this one"}})
                return
            content = json.dumps({"verdict": True}) if '"verdict"' in text else json.dumps({"explanation": "stub", "patched_code": task["reference_code"]})
            self.reply(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}})

        def reply(self, status: int, obj: dict[str, Any]) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - keep the test output quiet
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


def test_record_writes_a_set_that_replays_and_resumes(suite: Suite, stub: SimpleNamespace) -> None:
    client = OpenAICompatClient(stub.url, api_key="test-key-1", max_retries=0, timeout_s=10)
    template = prompts.current_template("bugfix")
    settings = FEATURES["bugfix"].default_settings
    options: dict[str, Any] = {"arm": "live", "model": "stub-model", "template": template, "settings": settings, "workers": 2}

    first = runner.record_arm("bugfix", suite, ALL, client, "cassettes/live", **options)
    assert (first.recorded, first.skipped_existing, first.failed) == (3, 0, 1)
    assert len(first.failures) == 1 and first.failures[0].startswith("bugfix:t-parse-port: ")
    assert stub.url not in first.failures[0] and "<endpoint>" in first.failures[0], "the endpoint URL may be a secret"
    assert first.to_json() == {"arm": "live", "feature": "bugfix", "split": ALL, "out": "cassettes/live", "prompt": template.id,
                               "model": "stub-model", "recorded": 3, "skipped_existing": 0, "failed": 1, "failures": first.failures}
    assert {s["path"] for s in stub.seen} == {"/v1/chat/completions"}
    assert {s["authorization"] for s in stub.seen} == {"Bearer test-key-1"}
    item = FEATURES["bugfix"].items(suite, ALL)[0]
    assert prompts.build_request(template, item, "stub-model", settings).payload() in [s["body"] for s in stub.seen]

    asked = len(stub.seen)
    second = runner.record_arm("bugfix", suite, ALL, client, "cassettes/live", **options)
    assert (second.recorded, second.skipped_existing, second.failed) == (0, 3, 1)
    assert len(stub.seen) == asked + 1, "only the item that failed is asked again"

    manifest = json.loads(Path("cassettes/live/manifest.json").read_text(encoding="utf-8"))
    assert manifest["synthetic"] is False and manifest["features"]["bugfix"]["model"] == "stub-model"
    with pytest.raises(RefusedError):
        runner.evaluate_arm("bugfix", suite, ALL, "cassettes/live")
    arm = runner.evaluate_arm("bugfix", suite, ALL, "cassettes/live", allow_missing=True)
    assert arm.name == "live" and arm.synthetic is False and len(arm.results) == 3 and all(r.correct for r in arm.results)


class FakeClient:
    """Answers every request with a "true" verdict, or raises ClientError for listed items."""

    def __init__(self, fail: frozenset[str] = frozenset()) -> None:
        self.fail = fail
        self.asked: list[str] = []

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.asked.append(request.item_id)
        if request.item_id in self.fail:
            raise ClientError("HTTP 503 after 3 retries")
        return verdict(True)


def test_record_counts_failures_and_refuses_a_conflicting_resume(suite: Suite) -> None:
    template = prompts.current_template("judge")
    options: dict[str, Any] = {"arm": "j", "template": template, "settings": FEATURES["judge"].default_settings}
    client = FakeClient(fail=frozenset({"judge:t-clamp:wrong"}))
    summary = runner.record_arm("judge", suite, DEV, client, "cassettes/j", model="m1", **options)
    assert (summary.recorded, summary.failed) == (6, 0) and sorted(client.asked) == [i.id for i in FEATURES["judge"].items(suite, DEV)]
    summary = runner.record_arm("judge", suite, HELDOUT, client, "cassettes/j", model="m1", **options)
    assert (summary.recorded, summary.failed, summary.failures) == (1, 1, ["judge:t-clamp:wrong: HTTP 503 after 3 retries"])
    with pytest.raises(RefusedError):
        runner.record_arm("judge", suite, ALL, FakeClient(), "cassettes/j", model="another-model", **options)


def test_record_refuses_a_template_of_another_feature(suite: Suite) -> None:
    with pytest.raises(RefusedError, match="judge/v2 belongs to judge"):
        runner.record_arm("bugfix", suite, ALL, FakeClient(), "x", arm="a", model="m", template=prompts.get_template("judge/v2"), settings=ModelSettings())
    assert RecordSummary("a", "judge", DEV, "x", "judge/v2", "m").to_json()["failures"] == []
