"""sandbox: what the child gets, what it reports, the deadline, the caps, the clean-up and the
start failures. Every test runs real child processes (POSIX only)."""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from llm_eval_harness import sandbox, verdicts
from llm_eval_harness.records import Check, Task
from llm_eval_harness.sandbox import SandboxResult, run_calls

pytestmark = pytest.mark.skipif(os.name != "posix", reason="the sandbox is POSIX-only")

FORGE_PREFIX = "import os, sys\nRESULTS = int(sys.argv[1])\n"


def records(result: SandboxResult) -> list[tuple]:
    return [(r.index, r.ok, r.repr if r.ok else r.exc) for r in result.records]


def gone(pid: int, wait_s: float = 5.0) -> bool:
    """Whether process `pid` has died (a zombie waiting for its new parent counts as dead)."""
    end = time.monotonic() + wait_s
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1][:1]
        except (OSError, IndexError):
            state = ""
        if state == "Z":
            return True
        time.sleep(0.05)
    return False


def script(path: Path, body: str) -> str:
    """An executable shell script standing in for the interpreter."""
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


# ---- values, exceptions, load errors -------------------------------------------------------------


def test_values_and_exceptions_with_their_mro() -> None:
    code = (
        "class TooBig(ValueError):\n    pass\n\n"
        "def f(x):\n    if x > 9:\n        raise TooBig('x' * 500)\n    return [x, str(x), x / 2, None]\n"
    )
    result = run_calls(code, "f", ["f(3)", "f(10)", "f('a')", "undefined(1)"], timeout_s=5)
    assert result.started and result.exit_code == 0 and not result.timed_out
    assert result.infra_error is None and result.load_error is None and result.malformed == 0 and result.duplicates == []
    first, second, third, fourth = result.records
    assert (first.index, first.ok, first.repr) == (0, True, "[3, '3', 1.5, None]")
    assert (second.ok, second.exc, second.mro) == (False, "TooBig", ("TooBig", "ValueError", "Exception", "BaseException"))
    assert len(second.msg) == 200
    assert (third.exc, fourth.exc) == ("TypeError", "NameError")


def test_unicode_values_round_trip() -> None:
    result = run_calls("def f():\n    return 'é ☃ \\U0001f600'\n", "f", ["f()"], timeout_s=5)
    assert result.records[0].repr == repr("é ☃ \U0001f600")


def test_a_call_that_does_not_compile_is_its_own_error() -> None:
    result = run_calls("def f():\n    return 1\n", "f", ["f(", "f()"], timeout_s=5)
    assert records(result) == [(0, False, "SyntaxError"), (1, True, "1")]


def test_system_exit_in_a_call_is_recorded_and_the_next_call_runs() -> None:
    result = run_calls("import sys\ndef f(x):\n    sys.exit(x)\n", "f", ["f(3)", "1 + 1"], timeout_s=5)
    assert records(result) == [(0, False, "SystemExit"), (1, True, "2")]
    assert result.records[0].mro == ("SystemExit", "BaseException")


def test_reprs_are_capped_and_a_failing_repr_is_not_a_literal() -> None:
    code = "class Odd:\n    def __repr__(self):\n        raise ValueError('no repr')\n\ndef f(n):\n    return 'x' * n if n else Odd()\n"
    result = run_calls(code, "f", [f"f({sandbox.MAX_REPR_CHARS * 2})", "f(0)"], timeout_s=5)
    assert len(result.records[0].repr) == sandbox.MAX_REPR_CHARS
    assert result.records[1].repr == "<repr() raised ValueError>"


@pytest.mark.parametrize("code, error", [
    ("def f(:\n    pass\n", "SyntaxError"),
    ("def g():\n    return 1\n", "NameError"),
    ("f = 3\n", "TypeError"),
    ("import no_such_module_anywhere\ndef f(): pass\n", "ModuleNotFoundError"),
    ("raise SystemExit(0)\n", "SystemExit"),
    ("import requests\ndef f(): pass\n", "ModuleNotFoundError"),  # no site-packages in the child
])
def test_load_errors(code: str, error: str) -> None:
    result = run_calls(code, "f", ["f()"], timeout_s=5)
    assert result.load_error is not None and result.load_error.startswith(error + ":")
    assert result.records == [] and result.infra_error is None


def test_os_exit_at_import_gives_no_records() -> None:
    result = run_calls("import os\nos._exit(0)\ndef f():\n    return 1\n", "f", ["f()"], timeout_s=5)
    assert result.exit_code == 0 and result.records == [] and result.load_error is None and result.infra_error is None


def test_the_code_reading_stdin_gets_end_of_file() -> None:
    result = run_calls("def f():\n    return input()\n", "f", ["f()"], timeout_s=5)
    assert records(result) == [(0, False, "EOFError")]


# ---- the deadline and the process group ------------------------------------------------------


def test_an_infinite_loop_times_out() -> None:
    start = time.monotonic()
    result = run_calls("def f():\n    while True:\n        pass\n", "f", ["1", "f()", "2"], timeout_s=1.0)
    assert time.monotonic() - start < 1.0 + 2.0
    assert result.timed_out and result.exit_code is not None and result.exit_code < 0
    assert records(result) == [(0, True, "1")], "the calls before the hang count"
    assert result.infra_error is None


FORK = (
    "import os, time\n"
    "def spawn():\n"
    "    pid = os.fork()\n"
    "    if pid == 0:\n"
    "        time.sleep(60)\n"
    "        os._exit(0)\n"
    "    return pid\n"
    "def hang():\n"
    "    time.sleep(60)\n"
)


def test_a_forked_grandchild_dies_when_the_child_exits() -> None:
    result = run_calls(FORK, "spawn", ["spawn()"], timeout_s=10)
    assert not result.timed_out and result.duration_s < 5
    assert gone(int(result.records[0].repr))


def test_a_forked_grandchild_dies_at_the_deadline() -> None:
    result = run_calls(FORK, "spawn", ["spawn()", "hang()"], timeout_s=1.0)
    assert result.timed_out
    assert gone(int(result.records[0].repr))


# ---- the environment and the work directory ------------------------------------------------


def test_the_child_inherits_no_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_EVAL_API_KEY", "not-for-the-child")
    code = "import os\ndef env(name):\n    return os.environ.get(name)\n"
    calls = ["env('LLM_EVAL_API_KEY')", "sorted(os.environ)", "os.path.realpath(env('HOME')) == os.path.realpath(os.getcwd())",
             "env('TMPDIR') == env('HOME')", "env('PATH') == os.defpath"]
    result = run_calls(code, "env", calls, timeout_s=5)
    assert [r.repr for r in result.records] == ["None", repr(sorted(sandbox.child_env(Path("w")))), "True", "True", "True"]


def test_child_env_is_built_from_nothing() -> None:
    env = sandbox.child_env(Path("work"))
    assert env == {"PATH": os.defpath, "LANG": "C.UTF-8", "HOME": "work", "TMPDIR": "work", "PYTHONDONTWRITEBYTECODE": "1"}


def test_the_child_runs_isolated_without_site_packages() -> None:
    code = "import sys\ndef flags():\n    return (sys.flags.isolated, sys.flags.no_site, 'site' in sys.modules)\n"
    result = run_calls(code, "flags", ["flags()"], timeout_s=5)
    assert result.records[0].repr == "(1, 1, False)"


def test_the_work_directory_is_removed_even_when_made_unreadable(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep\n", encoding="utf-8")
    code = (
        "import os\n"
        "def f(outside):\n"
        "    os.makedirs('locked/inner')\n"
        "    open('locked/inner/file.txt', 'w').close()\n"
        "    os.chmod('locked/inner', 0)\n"
        "    os.chmod('locked', 0o500)\n"
        "    os.symlink(outside, 'link-out')\n"
        "    return os.getcwd()\n"
    )
    result = run_calls(code, "f", [f"f({str(outside)!r})"], timeout_s=5)
    workdir = Path(json.loads(result.records[0].repr.replace("'", '"')))
    assert not workdir.exists()
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep\n", "a symlink is removed, never followed"
    assert not any("could not be removed" in note for note in result.notes)


# ---- output, forged results and caps ------------------------------------------------------


def test_prints_do_not_corrupt_results_and_are_tailed() -> None:
    code = (
        "import sys\n"
        "def f(x):\n"
        "    print('{\"i\": 0, \"ok\": true, \"repr\": \"999\"}')\n"
        "    print('to stderr', x, file=sys.stderr)\n"
        "    return x\n"
    )
    result = run_calls(code, "f", ["f(1)", "f(2)"], timeout_s=5)
    assert records(result) == [(0, True, "1"), (1, True, "2")]
    assert result.malformed == 0 and result.duplicates == []
    assert result.stdout_tail.count('"repr": "999"') == 2
    assert result.stderr_tail == "to stderr 1\nto stderr 2\n"


def test_forged_result_lines_are_duplicates_or_malformed() -> None:
    forged = [
        b'{"i": 0, "ok": true, "repr": "3"}',  # the right answer, written before the runner's own record
        b"not json at all",
        b'{"i": 7, "ok": true, "repr": "1"}',  # no such test
        b'{"i": true, "ok": true, "repr": "1"}',  # a bool is not an index
        b'{"load_error": "FakeError: x", "i": 1}',
    ]
    payload = b"".join(line + b"\n" for line in forged)
    code = FORGE_PREFIX + f"os.write(RESULTS, {payload!r})\n" + "def f():\n    return 0\n"
    result = run_calls(code, "f", ["f()", "f()"], timeout_s=5)
    assert result.duplicates == [0]
    assert result.malformed == 4
    assert result.load_error is None
    assert records(result)[1] == (1, True, "0")


def test_huge_output_is_truncated_with_a_note() -> None:
    code = "def f():\n    print('x' * 3_000_000)\n    print('the end')\n    return 1\n"
    result = run_calls(code, "f", ["f()"], timeout_s=10)
    assert records(result) == [(0, True, "1")]
    assert len(result.stdout_tail) == sandbox.TAIL_CHARS and result.stdout_tail.endswith("the end\n")
    assert any(note.startswith("stdout:") and "discarded" in note for note in result.notes)


def test_the_results_pipe_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "MAX_OUTPUT_BYTES", 4_000)
    code = FORGE_PREFIX + "os.write(RESULTS, b'x' * 10_000 + b'\\n')\ndef f():\n    return 1\n"
    result = run_calls(code, "f", ["f()"], timeout_s=5)
    assert result.records == [] and result.malformed == 1
    assert any(note.startswith("results:") for note in result.notes)


def test_output_that_is_not_utf8_is_replaced() -> None:
    result = run_calls("import os\ndef f():\n    os.write(1, b'bad \\xff\\xfe bytes')\n    return 1\n", "f", ["f()"], timeout_s=5)
    assert records(result) == [(0, True, "1")] and "bad" in result.stdout_tail and "\ufffd" in result.stdout_tail


def test_a_fifo_or_symlink_in_place_of_the_output_file_is_not_read(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET CONTENT\n", encoding="utf-8")
    code = (
        "import os\n"
        "def f(secret):\n"
        "    print('mine')\n"
        "    os.remove('stdout.txt')\n"
        "    os.symlink(secret, 'stdout.txt')\n"
        "    os.remove('stderr.txt')\n"
        "    os.mkfifo('stderr.txt')\n"
        "    return 1\n"
    )
    result = run_calls(code, "f", [f"f({str(secret)!r})"], timeout_s=5)
    assert records(result) == [(0, True, "1")]
    assert "SECRET" not in result.stdout_tail and result.stderr_tail == ""
    assert result.duration_s < 5


def test_closing_the_results_fd_loses_the_results_but_does_not_hang() -> None:
    result = run_calls(FORGE_PREFIX + "os.close(RESULTS)\ndef f():\n    return 1\n", "f", ["f()"], timeout_s=5)
    assert result.records == [] and result.infra_error is None and not result.timed_out


# ---- strict parsing ------------------------------------------------------------------------------


def parse(lines: list[bytes], n_calls: int = 2) -> tuple[bool, SandboxResult]:
    result = SandboxResult(started=True, exit_code=0, timed_out=False, duration_s=0.0)
    ready = sandbox._parse_results(b"\n".join(lines) + b"\n", n_calls, result)
    return ready, result


READY = b'{"ready": true}'


def test_parsing_keeps_the_first_record_per_index() -> None:
    ready, result = parse([READY, b'{"i": 1, "ok": true, "repr": "1"}', b'{"i": 1, "ok": true, "repr": "2"}',
                           b'{"i": 0, "ok": false, "exc": "E", "mro": ["E", "Exception"], "msg": "m"}'])
    assert ready and [(r.index, r.repr) for r in result.records] == [(1, "1"), (0, None)]
    assert result.duplicates == [1] and result.malformed == 0
    assert result.records[1].mro == ("E", "Exception")


@pytest.mark.parametrize("line", [
    b'{"i": 0, "ok": true, "repr": "1", "extra": 1}',
    b'{"i": 0, "ok": true}',
    b'{"i": 0, "ok": 1, "repr": "1"}',
    b'{"i": 0, "ok": true, "repr": 1}',
    b'{"i": 0, "ok": false, "exc": "E", "mro": "E", "msg": ""}',
    b'{"i": 0, "ok": false, "exc": "E", "mro": [1], "msg": ""}',
    b'{"i": 0, "ok": true, "repr": "1", "repr": "2"}',
    b'{"i": NaN, "ok": true, "repr": "1"}',
    b'{"i": -1, "ok": true, "repr": "1"}',
    b'{"i": 2, "ok": true, "repr": "1"}',
    b'{"i": 0.0, "ok": true, "repr": "1"}',
    b'[0, true, "1"]',
    b'{"i": 0, "ok": true, "repr": "\xff"}',
    b'{"ready": 1}',
    b'{"load_error": 3}',
    b'{"i": 0, "ok": true, "repr": "' + b"x" * (sandbox.MAX_REPR_CHARS + 1) + b'"}',
    b'[' * 100_000,
])
def test_malformed_lines_are_counted(line: bytes) -> None:
    ready, result = parse([READY, line])
    assert ready and result.records == [] and result.malformed == 1


def test_records_before_ready_and_extra_ready_or_load_errors_are_malformed() -> None:
    ready, result = parse([b'{"i": 0, "ok": true, "repr": "1"}', READY, READY, b'{"load_error": "A: a"}', b'{"load_error": "B: b"}'])
    assert ready and result.records == [] and result.malformed == 3 and result.load_error == "A: a"


def test_no_ready_record_means_not_started() -> None:
    ready, _result = parse([b'{"i": 0, "ok": true, "repr": "1"}'])
    assert not ready


# ---- start failures, platforms and the job ---------------------------------------------------


def test_a_missing_interpreter_is_an_infra_error() -> None:
    result = run_calls("def f(): pass\n", "f", ["f()"], python="/nonexistent/python3")
    assert not result.started and result.exit_code is None and result.records == []
    assert result.infra_error is not None and "/nonexistent" not in result.infra_error


def test_an_interpreter_that_runs_nothing_is_an_infra_error(tmp_path: Path) -> None:
    fake = script(tmp_path / "fake-python", "cat > /dev/null\nexit 0\n")
    result = run_calls("def f(): pass\n", "f", ["f()"], python=fake)
    assert result.started and result.exit_code == 0 and result.records == []
    assert result.infra_error is not None and "did not start" in result.infra_error


def test_an_unusable_temporary_directory_is_an_infra_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> str:
        raise PermissionError(13, "Permission denied", "/somewhere")

    monkeypatch.setattr(sandbox.tempfile, "mkdtemp", refuse)
    result = run_calls("def f(): pass\n", "f", ["f()"])
    assert not result.started and result.infra_error == "could not create a temporary directory: Permission denied"


def test_off_posix_the_sandbox_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_posix", lambda: False)
    with pytest.raises(sandbox.SandboxUnavailable):
        run_calls("def f(): pass\n", "f", ["f()"])


def test_the_job_holds_only_the_code_and_the_calls(tmp_path: Path) -> None:
    job_file, args_file = tmp_path / "job.json", tmp_path / "args.txt"
    fake = script(tmp_path / "capture", f"printf '%s\\n' \"$@\" > '{args_file}'\ncat > '{job_file}'\nexit 0\n")
    task = Task(
        id="t-marker", entry_point="f", buggy_code="def f(x):\n    return 0\n",
        shown_tests=(Check("f(1)", expected="'EXPECTED-MARKER-ONE'"),),
        hidden_tests=(Check("f(2)", expected="'EXPECTED-MARKER-TWO'"), Check("f(3)", raises="MarkerError")),
        reference_code="def f(x):\n    return 1\n", wrong_code="def f(x):\n    return 2\n",
    )
    verdict = verdicts.exec_verdict(task, "def f(x):\n    return x\n", python=fake)
    assert verdict.infra_error is not None and not verdict.passed
    raw = job_file.read_bytes()
    assert b"MARKER" not in raw and b"expected" not in raw and b"raises" not in raw
    assert json.loads(raw) == {"code": "def f(x):\n    return x\n", "entry_point": "f", "calls": ["f(1)", "f(2)", "f(3)"]}
    args = args_file.read_text(encoding="utf-8").split()
    assert args[:3] == ["-I", "-S", "_runner.py"] and args[3].isdigit() and len(args) == 4


def test_the_child_program_is_standard_library_only() -> None:
    import ast

    tree = ast.parse(sandbox.CHILD_SOURCE)
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert imported <= set(sys.stdlib_module_names)
    assert str(sandbox.MAX_REPR_CHARS) in sandbox.CHILD_SOURCE and "@" not in sandbox.CHILD_SOURCE


def test_runs_in_threads_do_not_interfere() -> None:
    outcomes: dict[int, list[tuple]] = {}

    def one(n: int) -> None:
        outcomes[n] = records(run_calls(f"def f():\n    return {n}\n", "f", ["f()", "f() * 2"], timeout_s=10))

    threads = [threading.Thread(target=one, args=(n,)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes == {n: [(0, True, str(n)), (1, True, str(2 * n))] for n in range(6)}


# ---- resource limits (Linux applies them all) ---------------------------------------------------


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS is honoured on Linux")
def test_the_memory_limit_applies() -> None:
    result = run_calls("def f():\n    return len(bytearray(2 * 1024 ** 3))\n", "f", ["f()", "1"], timeout_s=5)
    assert records(result) == [(0, False, "MemoryError"), (1, True, "1")]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_FSIZE behaves this way on Linux")
def test_the_file_size_limit_applies() -> None:
    code = "def f():\n    with open('big.bin', 'wb') as h:\n        h.write(b'x' * (2 * FILE_LIMIT))\n".replace("FILE_LIMIT", str(sandbox.FILE_LIMIT_BYTES))
    result = run_calls(code, "f", ["f()"], timeout_s=5)
    assert records(result) == [(0, False, "OSError")]


def test_no_core_files_and_a_cpu_limit_are_set() -> None:
    code = "import resource\ndef limit(name):\n    return resource.getrlimit(getattr(resource, name))\n"
    result = run_calls(code, "limit", ["limit('RLIMIT_CORE')", "limit('RLIMIT_CPU')"], timeout_s=2.5)
    assert result.records[0].repr == "(0, 0)"
    assert result.records[1].repr == "(4, 4)", "ceil(timeout) + 1 CPU seconds"
