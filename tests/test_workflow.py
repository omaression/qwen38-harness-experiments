"""The CI workflow, .github/workflows/ci.yml, checked without a YAML parser (the harness and its
tests need nothing beyond the standard library and pytest):

  - static text checks: the triggers and the dispatch inputs, read-only permissions, the
    concurrency group, the jobs and what they need, the Python matrix, the action versions,
    secrets only in the live jobs, no ${{ }} expression pasted into a script, and the self-test's
    expectations;
  - every llm-eval command line in the workflow parses against cli.build_parser;
  - the regression-gate job's shell steps, run in order with bash and jq on a copy of the
    repository's suite, cassettes and ledger: they pass on the committed cassettes, and the
    self-test fails when the "regressed" candidate is not rejected, or not by the expected gates;
  - the live jobs' steps: the secrets check, and the whole live evaluation run against a local
    OpenAI-compatible endpoint (a fake model that answers from the demo suite), which must
    record both arms, gate them and leak neither the API key nor the endpoint URL.

The workflow is written in a small, regular subset of YAML (block mappings, `- ` step lists,
`run: |` literal blocks); the helpers below read exactly that subset.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from llm_eval_harness import cli

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
TEXT = WORKFLOW.read_text(encoding="utf-8")
LINES = TEXT.splitlines()
JOBS = ("tests", "regression-gate", "experiments", "live-config", "live-eval")
LIVE_JOBS = ("live-config", "live-eval")
SECRETS = ("LLM_EVAL_BASE_URL", "LLM_EVAL_API_KEY", "LLM_EVAL_CANDIDATE_MODEL", "LLM_EVAL_BASELINE_MODEL")
#: What the self-test expects of the regressed candidate (tests/test_demo.py checks the same sets
#: on the reports themselves).
EXPECTED_FAILED = {"bugfix": ["primary", "schema_validity", "new_sast_violations"], "judge": ["primary", "schema_validity"]}
ACTIONS = {"actions/checkout@v7", "actions/setup-python@v7", "actions/upload-artifact@v7"}


# ---- reading the workflow ---------------------------------------------------------------------


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _block(lines: list[str], start: int) -> list[str]:
    """The lines nested under lines[start] (blank lines included, trailing blanks dropped)."""
    base = _indent(lines[start])
    out = []
    for line in lines[start + 1:]:
        if line.strip() and _indent(line) <= base:
            break
        out.append(line)
    while out and not out[-1].strip():
        out.pop()
    return out


def _entry(lines: list[str], key: str, indent: int) -> list[str]:
    """The block of the mapping entry `key:` at this indentation (KeyError when absent)."""
    for number, line in enumerate(lines):
        if _indent(line) == indent and line.strip().split(":", 1)[0] == key and line.strip().startswith(f"{key}:"):
            return [line] + _block(lines, number)
    raise KeyError(key)


def jobs() -> dict[str, list[str]]:
    """Job id -> its lines (the `  <id>:` line first), in file order."""
    body = _entry(LINES, "jobs", 0)[1:]
    out: dict[str, list[str]] = {}
    for number, line in enumerate(body):
        if _indent(line) == 2 and line.strip().endswith(":") and not line.strip().startswith("#"):
            out[line.strip()[:-1]] = [line] + _block(body, number)
    return out


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def steps(job: str) -> list[dict[str, Any]]:
    """The steps of a job: {"name", "uses", "if", "id", "run", "with", "env"} (absent keys left
    out; "run" is the script text, dedented; "with" and "env" are {key: raw value})."""
    lines = _entry(jobs()[job], "steps", 4)[1:]
    starts = [n for n, line in enumerate(lines) if line.lstrip().startswith("- ") and _indent(line) == 6]
    out = []
    for first, end in zip(starts, starts[1:] + [len(lines)]):
        chunk = [lines[first].replace("- ", "  ", 1)] + lines[first + 1:end]
        step: dict[str, Any] = {}
        number = 0
        while number < len(chunk):
            line = chunk[number]
            if _indent(line) == 8 and ":" in line and not line.strip().startswith("#"):
                key, _, value = line.strip().partition(":")
                nested = _block(chunk, number)
                if key == "run" and value.strip() == "|":
                    width = min(_indent(x) for x in nested if x.strip())
                    step["run"] = "\n".join(x[width:] for x in nested) + "\n"
                elif key in ("with", "env"):
                    step[key] = {x.strip().partition(":")[0]: x.strip().partition(":")[2].strip() for x in nested if x.strip()}
                else:
                    step[key] = _unquote(value)
                number += 1 + len(nested)
            else:
                number += 1
        out.append(step)
    return out


def run_scripts() -> list[tuple[str, str]]:
    """(job, script) for every step that runs a shell script."""
    return [(job, step["run"]) for job in jobs() for step in steps(job) if "run" in step]


# ---- static checks ------------------------------------------------------------------------------


def test_the_workflow_is_read_by_these_helpers() -> None:
    assert list(jobs()) == list(JOBS)
    for job in JOBS:
        assert steps(job), job
        for step in steps(job):
            assert ("uses" in step) != ("run" in step), f"{job}: a step runs a script or uses an action: {step}"


def test_triggers_and_dispatch_inputs() -> None:
    on = "\n".join(_entry(LINES, "on", 0))
    assert re.search(r"^  push:\n    branches: \[main\]$", on, re.M)
    assert re.search(r"^  pull_request:$", on, re.M)
    dispatch = _entry(LINES, "on", 0)
    inputs = _entry(_entry(dispatch, "workflow_dispatch", 2), "inputs", 4)
    feature, split = "\n".join(_entry(inputs, "feature", 6)), "\n".join(_entry(inputs, "split", 6))
    assert "type: choice" in feature and "options: [bugfix, judge]" in feature
    assert "type: choice" in split and "options: [dev, heldout]" in split
    assert "default: dev" in split, "a live experiment must not spend the held-out split by default"


def test_permissions_are_read_only() -> None:
    assert _entry(LINES, "permissions", 0) == ["permissions:", "  contents: read"]
    assert TEXT.count("permissions:") == 1, "no job may widen the workflow's permissions"
    assert not re.search(r":\s*write\b|write-all", TEXT)


def test_a_newer_run_cancels_a_stale_one() -> None:
    concurrency = "\n".join(_entry(LINES, "concurrency", 0))
    assert "group: " in concurrency and "github.ref" in concurrency
    assert "cancel-in-progress: true" in concurrency


def test_scripts_run_under_bash_with_pipefail() -> None:
    # An explicit `shell: bash` makes the runner use `bash --noprofile --norc -eo pipefail`, which
    # is how run_step below executes the scripts; without it, steps get `bash -e` only.
    assert _entry(LINES, "defaults", 0) == ["defaults:", "  run:", "    shell: bash"]
    assert "shell:" not in "\n".join(line for job in jobs().values() for line in job), "no job overrides the shell"


def test_the_jobs_and_what_they_need() -> None:
    table = jobs()
    head = {job: "\n".join(line for line in lines if _indent(line) == 4) for job, lines in table.items()}
    assert "needs" not in head["tests"]
    assert "    needs: tests" in head["regression-gate"]
    assert "    if: github.event_name == 'workflow_dispatch'" in head["live-config"]
    assert "    needs: live-config" in head["live-eval"]
    assert "    if: needs.live-config.outputs.configured == 'true'" in head["live-eval"]
    for job in JOBS:
        assert "    runs-on: ubuntu-latest" in head[job]
        assert re.search(r"^    timeout-minutes: \d+$", head[job], re.M), f"{job} has no time limit"


def test_the_python_matrix_is_quoted() -> None:
    # Unquoted, YAML reads 3.10 as the number 3.1.
    assert '        python-version: ["3.10", "3.11", "3.12"]' in jobs()["tests"]
    assert all(step.get("with", {}).get("python-version") == '"3.12"'
               for job in ("regression-gate", "live-eval") for step in steps(job) if "setup-python" in step.get("uses", ""))


def test_the_tests_job() -> None:
    scripts = [step["run"].strip() if "run" in step else "" for step in steps("tests")]
    for command in ("python -m pip install -e '.[test]'", "python -m pytest -q", "llm-eval tasks validate --suite tasks/demo", "llm-eval ledger verify"):
        assert command in scripts, command


def test_actions_are_pinned_to_their_current_major_versions() -> None:
    used = {step["uses"] for job in JOBS for step in steps(job) if "uses" in step}
    assert used == ACTIONS
    for job in JOBS:
        for step in steps(job):
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step.get("with", {}).get("persist-credentials") == "false", f"{job}: the checkout keeps no token"


def test_secrets_appear_only_in_the_live_jobs() -> None:
    table = jobs()
    for job in ("tests", "regression-gate"):
        assert "secrets." not in "\n".join(table[job]), f"{job} must run without secrets"
    outside = [line for line in LINES if "secrets." in line and not any(line in table[job] for job in LIVE_JOBS)]
    assert not outside
    live = "\n".join(table["live-eval"])
    for name in SECRETS:
        assert f"{name}: ${{{{ secrets.{name} }}}}" in live
    for name in SECRETS[:3]:
        assert f"secrets.{name}" in "\n".join(table["live-config"])


def test_no_expression_is_pasted_into_a_script() -> None:
    # A ${{ }} inside `run:` is substituted into the script's text before the shell sees it: an
    # input or secret holding quotes would then be shell code. Everything goes through env.
    for job, script in run_scripts():
        assert "${{" not in script, f"{job}: {script.splitlines()[0]}"
    live = "\n".join(jobs()["live-eval"])
    assert "FEATURE: ${{ inputs.feature }}" in live and "SPLIT: ${{ inputs.split }}" in live


def test_the_regression_gate_job() -> None:
    gate_steps = steps("regression-gate")
    names = [step.get("name", step.get("uses")) for step in gate_steps]
    scripts = {step.get("name"): step.get("run", "") for step in gate_steps}
    gate = scripts["Gate the candidate and the regressed candidate against the current baseline"]
    assert "for f in bugfix judge; do" in gate
    assert gate.count("--split heldout") == 2 and gate.count("--baseline @current") == 2
    assert '> "reports/$f-candidate.json"' in gate and '> "reports/$f-regressed.json"' in gate
    assert "--candidate cassettes/candidate " in gate and "--candidate cassettes/candidate-regressed " in gate
    assert "--items-out reports/items/candidate" in gate and "--items-out reports/items/regressed" in gate
    # The summary and the upload run whatever happened; the checks come last.
    always = [step for step in gate_steps if step.get("if") == "always()"]
    assert [step["name"] for step in always] == ["Write the job summary", "Upload the reports"]
    assert always[1]["uses"] == "actions/upload-artifact@v7"
    assert names.index("Write the job summary") > names.index("Gate the candidate and the regressed candidate against the current baseline")
    assert names[-2:] == ["Enforce the verdict on the candidate", "Self-test that the gate rejects the regressed candidate"]
    assert '>> "$GITHUB_STEP_SUMMARY"' in scripts["Write the job summary"]
    upload = next(step for step in gate_steps if step.get("uses", "").startswith("actions/upload-artifact"))
    assert upload["with"]["path"] == "reports/"
    assert "jq -e '.promote == true'" in scripts["Enforce the verdict on the candidate"]
    assert "::error title=Regression gate::" in scripts["Enforce the verdict on the candidate"]


def test_the_self_test_expectations() -> None:
    script = next(step["run"] for step in steps("regression-gate") if step.get("name", "").startswith("Self-test"))
    assert "jq -e '.promote == false'" in script
    for feature, gates in EXPECTED_FAILED.items():
        assert f"expect_rejected {feature} '{json.dumps(gates, separators=(',', ':'))}'" in script
    assert script.count("::error title=Gate self-test::") == 2
    assert script.rstrip().endswith('exit "$status"')


def test_the_live_job() -> None:
    scripts = {step.get("name"): step.get("run", "") for step in steps("live-eval")}
    candidate = scripts["Record the candidate with the current prompt"]
    assert "llm-eval record" in candidate and "--prompt" not in candidate, "the candidate uses the current prompt"
    assert '--split "$SPLIT"' in candidate and '--model "$LLM_EVAL_CANDIDATE_MODEL"' in candidate
    baseline = scripts["Record the baseline with the current baseline's prompt"]
    assert 'baseline_dir=$(llm-eval ledger current --feature "$FEATURE")' in baseline
    assert '--prompt-from "$baseline_dir"' in baseline
    assert '--model "${LLM_EVAL_BASELINE_MODEL:-$LLM_EVAL_CANDIDATE_MODEL}"' in baseline
    gate = scripts["Gate the candidate against the baseline"]
    assert "--candidate artifacts/live-candidate" in gate and "--baseline artifacts/live-baseline" in gate and '--split "$SPLIT"' in gate
    assert "--record-evaluation" not in TEXT, "CI never writes the ledger"
    always = [step for step in steps("live-eval") if step.get("if") == "always()"]
    assert [step["name"] for step in always] == ["Write the job summary", "Upload the recordings and the report"]
    assert always[1]["uses"] == "actions/upload-artifact@v7" and always[1]["with"]["path"] == "artifacts/"
    assert "jq -e '.promote == true' artifacts/report.json" in scripts["Enforce the verdict"]
    check = next(step for step in steps("live-config") if step.get("id") == "check")
    assert 'echo "configured=true" >> "$GITHUB_OUTPUT"' in check["run"] and 'echo "configured=false" >> "$GITHUB_OUTPUT"' in check["run"]
    assert "::notice title=Live evaluation skipped::" in check["run"]


# ---- every llm-eval command line parses ----------------------------------------------------------

#: Placeholder values for the shell variables the scripts use.
VARIABLES = {"f": "bugfix", "FEATURE": "bugfix", "SPLIT": "dev", "feature": "bugfix", "split": "dev"}


def _substitute(command: str) -> str:
    def value(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2) or ""
        return VARIABLES.get(name.split(":-")[0].split("[")[0], "x")

    return re.sub(r"\$\{([^}]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", value, command)


def llm_eval_commands() -> list[str]:
    commands = []
    for _job, script in run_scripts():
        for line in script.replace("\\\n", " ").splitlines():
            for match in re.finditer(r"llm-eval ([^>|;&)]*)", line):
                commands.append(match.group(1).strip())
    return commands


def test_every_llm_eval_command_line_parses() -> None:
    commands = llm_eval_commands()
    used = set()
    for command in commands:
        args = shlex.split(_substitute(command))
        cli.build_parser().parse_args(args)  # argparse exits 2 on an unknown or misspelt flag
        used.add(" ".join(args[:2]) if args[0] in ("tasks", "ledger") else args[0])
    assert used == {"tasks validate", "ledger verify", "ledger current", "gate", "summary", "record"}


# ---- running the regression-gate job's steps -------------------------------------------------------

needs_shell = pytest.mark.skipif(not (shutil.which("bash") and shutil.which("jq")), reason="needs bash and jq, as on the CI runner")


def _job_environment(workspace: Path) -> dict[str, str]:
    """The environment of a step on the runner, with `llm-eval` running this checkout's harness."""
    bin_dir = workspace / ".bin"
    bin_dir.mkdir(exist_ok=True)
    shim = bin_dir / "llm-eval"
    shim.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -m llm_eval_harness "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLM_EVAL_", "GITHUB_")) and k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
    env.update({
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "GITHUB_STEP_SUMMARY": str(workspace / "step-summary.md"),
    })
    return env


def run_step(workspace: Path, script: str) -> subprocess.CompletedProcess[str]:
    """One `run:` script as the runner runs it: bash --noprofile --norc -eo pipefail."""
    path = workspace / ".step.sh"
    path.write_text(script, encoding="utf-8")
    return subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", str(path)], cwd=workspace,
                          env=_job_environment(workspace), capture_output=True, text=True, timeout=600)


def _gate_job_scripts() -> list[tuple[str, str]]:
    """(name, script) of the regression-gate job's own steps, install excluded."""
    return [(step["name"], step["run"]) for step in steps("regression-gate") if "run" in step and "pip install" not in step["run"]]


@pytest.fixture(scope="module")
def gate_job(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[tuple[str, subprocess.CompletedProcess[str]]]]:
    """The regression-gate job's steps, run once in order on a copy of the suite, the cassettes
    and the ledger."""
    workspace = tmp_path_factory.mktemp("runner")
    for part in ("tasks", "cassettes", "ledger"):
        shutil.copytree(ROOT / part, workspace / part)
    results = [(name, run_step(workspace, script)) for name, script in _gate_job_scripts()]
    return workspace, results


@needs_shell
def test_the_regression_gate_steps_pass_on_the_committed_cassettes(gate_job) -> None:
    workspace, results = gate_job
    for name, result in results:
        assert result.returncode == 0, f"{name}:\n{result.stdout}\n{result.stderr}"
    for feature in EXPECTED_FAILED:
        for arm in ("candidate", "regressed"):
            assert json.loads((workspace / "reports" / f"{feature}-{arm}.json").read_text(encoding="utf-8"))["feature"] == feature
        assert (workspace / "reports" / "items" / "candidate" / f"{feature}.candidate.items.jsonl").is_file()
    summary = (workspace / "step-summary.md").read_text(encoding="utf-8")
    assert summary.startswith("# LLM regression gate\n") and "**Synthetic demo data" in summary
    assert summary.count("| PROMOTE |") == 2 and summary.count("| REJECT |") == 2
    outputs = "\n".join(result.stdout for _name, result in results)
    assert "the bugfix candidate passed the gate" in outputs and "the judge candidate passed the gate" in outputs
    assert "the regressed bugfix candidate was rejected as expected" in outputs
    assert "the regressed judge candidate was rejected as expected" in outputs
    assert "::error" not in outputs


def _self_test(workspace: Path) -> subprocess.CompletedProcess[str]:
    name, script = _gate_job_scripts()[-1]
    assert name.startswith("Self-test")
    return run_step(workspace, script)


@needs_shell
def test_the_self_test_fails_when_the_regressed_candidate_promotes(gate_job, tmp_path: Path) -> None:
    workspace, _results = gate_job
    (tmp_path / "reports").mkdir()
    for feature in EXPECTED_FAILED:
        promoted = workspace / "reports" / f"{feature}-candidate.json"
        shutil.copyfile(promoted, tmp_path / "reports" / f"{feature}-regressed.json")
    result = _self_test(tmp_path)
    assert result.returncode == 1
    for feature in EXPECTED_FAILED:
        assert f"::error title=Gate self-test::the regressed {feature} candidate was promoted" in result.stdout


@needs_shell
def test_the_self_test_fails_when_an_expected_gate_passes(gate_job, tmp_path: Path) -> None:
    workspace, _results = gate_job
    (tmp_path / "reports").mkdir()
    for feature in EXPECTED_FAILED:
        report = json.loads((workspace / "reports" / f"{feature}-regressed.json").read_text(encoding="utf-8"))
        for gate in report["gates"]:
            if gate["gate"] == "schema_validity":
                gate["passed"] = True  # the gate no longer notices the broken replies
        (tmp_path / "reports" / f"{feature}-regressed.json").write_text(json.dumps(report), encoding="utf-8")
    result = _self_test(tmp_path)
    assert result.returncode == 1
    assert result.stdout.count("but not by every gate of") == 2
    assert "::error title=Gate self-test::the regressed judge candidate was rejected" in result.stdout


@needs_shell
def test_the_enforcement_step_fails_when_the_candidate_is_rejected_by_its_gate(gate_job, tmp_path: Path) -> None:
    workspace, _results = gate_job
    (tmp_path / "reports").mkdir()
    shutil.copyfile(workspace / "reports" / "bugfix-regressed.json", tmp_path / "reports" / "bugfix-candidate.json")
    shutil.copyfile(workspace / "reports" / "judge-candidate.json", tmp_path / "reports" / "judge-candidate.json")
    script = dict(_gate_job_scripts())["Enforce the verdict on the candidate"]
    result = run_step(tmp_path, script)
    assert result.returncode == 1
    assert "::error title=Regression gate::the bugfix candidate failed the gate" in result.stdout
    assert "new_sast_violations" in result.stdout


# ---- running the live jobs' steps against a local endpoint -------------------------------------------


def _live_env(workspace: Path, **variables: str) -> dict[str, str]:
    env = _job_environment(workspace)
    env.update({"GITHUB_OUTPUT": str(workspace / "github-output.txt"), **variables})
    return env


def _run_live_step(workspace: Path, script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    path = workspace / ".live-step.sh"
    path.write_text(script, encoding="utf-8")
    return subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", str(path)], cwd=workspace, env=env,
                          capture_output=True, text=True, timeout=600)


@needs_shell
@pytest.mark.parametrize("missing", ["", "LLM_EVAL_API_KEY"])
def test_the_live_config_step(tmp_path: Path, missing: str) -> None:
    secrets = {"LLM_EVAL_BASE_URL": "http://127.0.0.1:9/v1", "LLM_EVAL_API_KEY": "k", "LLM_EVAL_CANDIDATE_MODEL": "m"}
    if missing:
        secrets[missing] = ""  # an unset secret reaches the step as an empty string
    script = next(step["run"] for step in steps("live-config") if step.get("id") == "check")
    result = _run_live_step(tmp_path, script, _live_env(tmp_path, **secrets))
    assert result.returncode == 0, result.stderr
    output = (tmp_path / "github-output.txt").read_text(encoding="utf-8")
    assert output == ("configured=false\n" if missing else "configured=true\n")
    assert ("::notice title=Live evaluation skipped::" in result.stdout) == bool(missing)


@pytest.fixture
def fake_endpoint() -> Any:
    """An OpenAI-compatible endpoint on 127.0.0.1 that answers from the demo suite: with a v2
    prompt it returns the reference fix and judges correctly; with a v1 prompt it returns the
    overfitted fix and approves every patch. It records the Authorization headers it saw."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from llm_eval_harness import tasks as tasks_mod

    suite = tasks_mod.load_suite(ROOT / "tasks" / "demo")
    by_code = {task.buggy_code.strip("\n"): task for task in suite.tasks}
    seen: list[str | None] = []

    def answer(body: dict[str, Any]) -> str:
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        blocks = re.findall(r"```python\n(.*?)\n```", user, re.S)
        task = by_code[blocks[0]]
        careful = "Hidden tests check other inputs" in system or "Check other inputs yourself" in system
        if "patched_code" in system:
            return json.dumps({"explanation": "Fixed the bug.", "patched_code": task.reference_code if careful else task.wrong_code})
        return json.dumps({"verdict": blocks[2] == task.reference_code.strip("\n") if careful else True})

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(self.headers.get("Authorization"))
            payload = {"choices": [{"message": {"role": "assistant", "content": answer(body)}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
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
        yield {"url": f"http://127.0.0.1:{server.server_address[1]}/v1", "seen": seen}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@needs_shell
@pytest.mark.parametrize("feature", ["bugfix", "judge"])
def test_the_live_eval_steps_against_a_local_endpoint(tmp_path: Path, fake_endpoint: dict[str, Any], feature: str) -> None:
    for part in ("tasks", "cassettes", "ledger"):
        shutil.copytree(ROOT / part, tmp_path / part)
    key = "test-key-that-must-not-leak"
    env = _live_env(tmp_path, LLM_EVAL_BASE_URL=fake_endpoint["url"], LLM_EVAL_API_KEY=key, LLM_EVAL_CANDIDATE_MODEL="fake-model",
                    LLM_EVAL_BASELINE_MODEL="", FEATURE=feature, SPLIT="dev")
    for step in steps("live-eval"):
        if "run" in step and "pip install" not in step["run"]:
            result = _run_live_step(tmp_path, step["run"], env)
            assert result.returncode == 0, f"{step['name']}:\n{result.stdout}\n{result.stderr}"
    assert set(fake_endpoint["seen"]) == {f"Bearer {key}"}
    report = json.loads((tmp_path / "artifacts" / "report.json").read_text(encoding="utf-8"))
    assert report["promote"] is True and report["split"] == "dev" and report["feature"] == feature
    assert report["candidate"]["synthetic"] is False and report["candidate"]["prompt"] == f"{feature}/v2"
    assert report["baseline"]["prompt"] == f"{feature}/v1", "the baseline is asked with the current baseline's prompt"
    assert report["baseline"]["model"] == "fake-model", "without LLM_EVAL_BASELINE_MODEL the candidate model answers"
    assert report["comparison"]["gains"] > 0 and report["comparison"]["losses"] == 0
    assert (tmp_path / "step-summary.md").read_text(encoding="utf-8").startswith("# Live evaluation\n")
    for path in (tmp_path / "artifacts").rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert key not in text and fake_endpoint["url"] not in text, path.name
