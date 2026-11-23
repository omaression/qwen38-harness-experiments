"""Run model-written code in a throwaway directory, in a child process, under a deadline.

    run_calls(code, entry_point, calls, timeout_s=10.0) -> SandboxResult

This module only reports what happened. Whether a test passed is decided by
verdicts.exec_verdict in the parent process, from values the child returned: the expected values
of the tests never enter the child, so code under test cannot "pass" by writing a fake report,
exiting early with status 0, or skipping a test. It can only pass by returning the right values.

WHAT THE CHILD GETS
  - a fresh temporary directory (tempfile.mkdtemp) as its working directory, HOME and TMPDIR,
    removed afterwards whatever happened;
  - an environment built from nothing (child_env): PATH=os.defpath, LANG=C.UTF-8, HOME, TMPDIR,
    PYTHONDONTWRITEBYTECODE=1. Nothing is inherited, so no API key or token reaches it;
  - the interpreter in isolated mode without site-packages: [python, "-I", "-S", runner, fd]
    (python defaults to sys.executable): no PYTHON* variables, no user site, no third-party
    packages, the script's directory not on sys.path;
  - POSIX resource limits, best effort (a limit the platform refuses is skipped): CPU seconds
    (timeout_s + 1), address space (MEMORY_LIMIT_BYTES), file size (FILE_LIMIT_BYTES), no core
    files; and a session of its own (start_new_session=True), so that at the deadline the whole
    process group is killed (os.killpg SIGKILL), grandchildren included;
  - the job on stdin, as one JSON object: {"code": ..., "entry_point": ..., "calls": [...]}.
    The job has no expected values.

WHAT THE CHILD DOES (CHILD_SOURCE, written into the temporary directory as _runner.py)
  1. reads the job; points its own fds 1 and 2 at files in the temporary directory, so prints of
     the code under test cannot interleave with results; then writes {"ready": true}, before any
     of the code runs. Without that record the run is an infra_error (the runner itself did not
     start: a broken interpreter, a limit too small to start one), never a failed patch;
  2. compiles and execs `code` as module "solution" (a fresh module object in sys.modules); if
     that raises, or `entry_point` is not a callable in the module, it writes one record
     {"load_error": "<ExceptionName>: <message>"} and exits;
  3. evaluates each call expression with eval(compile(call, "<test i>", "eval"), namespace) in
     the module's namespace, in order, and after each writes one JSON line to the results fd:
         {"i": 0, "ok": true, "repr": repr(value)}           (repr cut at MAX_REPR_CHARS)
         {"i": 1, "ok": false, "exc": "ValueError", "mro": ["ValueError", "Exception", ...],
          "msg": "<first 200 characters of str(exc)>"}
     flushed immediately, so the calls that finished count even if a later one hangs or the
     process dies. Every exception a call raises is recorded the same way, SystemExit and
     KeyboardInterrupt included (exc "SystemExit"), and the next call runs: a call that raises
     SystemExit has raised, it has not passed. os._exit ends the process, and the calls not yet
     reported are missing. A value whose repr() raises is reported as "<repr() raised X>",
     which is not a literal, so it can never pass as a value or as an expected exception.
  4. flushes and leaves with os._exit(0): no atexit handler or thread of the code runs after it.
  The results fd is a pipe the parent created and passed with pass_fds; its number is argv[1].

WHAT THE PARENT DOES
  Reads the results pipe and the captured output (each capped at MAX_OUTPUT_BYTES; the rest is
  discarded and noted), waits until the deadline, kills the process group on timeout, and parses
  the result lines strictly: a line that is not one well-formed record is counted in `malformed`;
  an index reported twice is listed in `duplicates` (verdicts fails that test). The exit status is
  recorded and decides nothing. A child that could not be started at all (no interpreter, fork
  failure, unwritable temporary directory) gives infra_error and no records: that is a fault of
  the machine, never of the patch, and the caller must not score the item.
  When the child exits before the deadline, its process group is killed at once as well, so a
  process the code forked cannot outlive the run or keep writing results; what is left in the
  pipes is then read for at most a second. Code under test can write to the results fd itself
  (its number is in argv): a forged line for an index the runner reports too makes that index a
  duplicate, so forging can only fail a test, never pass one.

LIMITS (docs/DESIGN.md, "Sandbox"). This is not a security boundary. The child runs as the same
user, can read every file that user can read, can open network connections, and can use as much
wall time as the deadline allows. It keeps a careless or overfitted patch from touching the
working tree, the environment's secrets or the verdict; it does not stop a hostile one. Score
untrusted models inside a container or VM with no network (for example, `docker run
--network none`). POSIX only: on other platforms run_calls raises SandboxUnavailable.
"""

from __future__ import annotations

import json
import math
import os
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    import resource
except ImportError:  # not POSIX; run_calls refuses such platforms anyway
    resource = None  # type: ignore[assignment]

DEFAULT_TIMEOUT_S = 10.0
#: Captured bytes kept per stream (results pipe, stdout, stderr).
MAX_OUTPUT_BYTES = 1_000_000
#: The longest repr the child sends for one value.
MAX_REPR_CHARS = 20_000
#: Address-space limit of the child (RLIMIT_AS), where the platform supports it.
MEMORY_LIMIT_BYTES = 1 << 30
#: Largest file the child may write (RLIMIT_FSIZE).
FILE_LIMIT_BYTES = 10 << 20
#: Characters of the code's own stdout / stderr kept for the item log.
TAIL_CHARS = 2_000

#: Characters of an exception's message the child sends.
_MSG_CHARS = 200
#: How often the parent checks whether the child has exited while nothing else happens.
_TICK_S = 0.05
#: How long the parent reads what is left in the pipes once the process group is dead.
_DRAIN_S = 1.0
_RUNNER, _STDOUT, _STDERR = "_runner.py", "stdout.txt", "stderr.txt"
#: waitid(WNOWAIT) sees that the child exited without reaping it, so its process group id
#: cannot be reused before the group is killed. (CPython does not offer os.waitid on macOS.)
_CAN_PEEK = all(hasattr(os, name) for name in ("waitid", "WNOWAIT", "WEXITED", "WNOHANG", "P_PID"))


class SandboxUnavailable(RuntimeError):
    """The platform lacks what the sandbox needs (POSIX process groups and pass_fds)."""


@dataclass
class CallRecord:
    """What the child reported for one call."""

    index: int
    ok: bool
    repr: str | None = None
    exc: str | None = None
    mro: tuple[str, ...] = ()
    msg: str = ""


@dataclass
class SandboxResult:
    """How one sandbox run went.

      started       the child process was created
      exit_code     its exit status (negative: killed by that signal); None if it never ran
      timed_out     the deadline hit and the process group was killed
      duration_s    wall time from start to the end of the child
      records       well-formed call records in the order received (first record per index)
      duplicates    indexes reported more than once
      malformed     result lines that were not well-formed records
      load_error    the code did not load, or defines no callable entry point
      stdout_tail, stderr_tail   the last TAIL_CHARS of the code's own output
      infra_error   the child could not be started: no verdict may be drawn
      notes         anything else worth logging (output truncated, a limit not applied, ...)
    """

    started: bool
    exit_code: int | None
    timed_out: bool
    duration_s: float
    records: list[CallRecord] = field(default_factory=list)
    duplicates: list[int] = field(default_factory=list)
    malformed: int = 0
    load_error: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    infra_error: str | None = None
    notes: list[str] = field(default_factory=list)


_CHILD_TEMPLATE = r'''"""The sandbox's child process (llm_eval_harness.sandbox), run as: python -I -S _runner.py <fd>."""
import json
import os
import sys
import traceback
import types

MAX_REPR_CHARS = @MAX_REPR_CHARS@
MSG_CHARS = @MSG_CHARS@


def describe(exc):
    try:
        return str(exc)[:MSG_CHARS]
    except BaseException:
        return "<str() of the exception failed>"


def value_repr(value):
    try:
        return repr(value)[:MAX_REPR_CHARS]
    except BaseException as exc:
        return "<repr() raised " + type(exc).__name__ + ">"


def exception_record(i, exc):
    try:
        cls = type(exc)
        mro = [str(c.__name__) for c in cls.__mro__ if c is not object]
        return {"i": i, "ok": False, "exc": str(cls.__name__), "mro": mro, "msg": describe(exc)}
    except BaseException:
        return {"i": i, "ok": False, "exc": "<unknown>", "mro": ["<unknown>"], "msg": ""}


def main():
    fd = int(sys.argv[1])
    write, dumps = os.write, json.dumps

    def send(record):
        data = (dumps(record) + "\n").encode("ascii")
        while data:
            data = data[write(fd, data):]

    job = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    code, entry_point, calls = job["code"], job["entry_point"], job["calls"]
    sys.stdout.flush()
    sys.stderr.flush()
    for target, name in ((1, "stdout.txt"), (2, "stderr.txt")):
        handle = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.dup2(handle, target)
        os.close(handle)
    send({"ready": True})

    module = types.ModuleType("solution")
    module.__file__ = "solution.py"
    sys.modules["solution"] = module
    namespace = module.__dict__
    try:
        exec(compile(code, "solution.py", "exec"), namespace)
        if entry_point not in namespace:
            raise NameError("the code defines no " + repr(entry_point))
        if not callable(namespace[entry_point]):
            raise TypeError(repr(entry_point) + " is not callable")
    except BaseException as exc:
        send({"load_error": type(exc).__name__ + ": " + describe(exc)})
        return
    for i, call in enumerate(calls):
        try:
            value = eval(compile(call, "<test " + str(i) + ">", "eval"), namespace)
        except BaseException as exc:
            send(exception_record(i, exc))
        else:
            send({"i": i, "ok": True, "repr": value_repr(value)})


if __name__ == "__main__":
    status = 0
    try:
        main()
    except BaseException:
        status = 70
        try:
            traceback.print_exc()
        except BaseException:
            pass
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except BaseException:
        pass
    os._exit(status)
'''

#: The child program (see WHAT THE CHILD DOES). Standard library only.
CHILD_SOURCE = _CHILD_TEMPLATE.replace("@MAX_REPR_CHARS@", str(MAX_REPR_CHARS)).replace("@MSG_CHARS@", str(_MSG_CHARS))


def child_env(workdir: Path) -> dict[str, str]:
    """The child's whole environment (see WHAT THE CHILD GETS)."""
    home = str(workdir)
    return {"PATH": os.defpath, "LANG": "C.UTF-8", "HOME": home, "TMPDIR": home, "PYTHONDONTWRITEBYTECODE": "1"}


def run_calls(
    code: str,
    entry_point: str,
    calls: Sequence[str],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    python: str | None = None,
) -> SandboxResult:
    """Exec `code` in a sandboxed child and evaluate each call expression in its namespace.
    `python` defaults to sys.executable. Never raises for anything the code does; raises
    SandboxUnavailable on a platform without POSIX process control."""
    if not _posix():
        raise SandboxUnavailable("the sandbox needs POSIX process groups and pass_fds (Linux or macOS)")
    result = SandboxResult(started=False, exit_code=None, timed_out=False, duration_s=0.0)
    start = time.monotonic()
    try:
        workdir = Path(tempfile.mkdtemp(prefix="llm-eval-sandbox-"))
    except OSError as exc:
        result.infra_error = f"could not create a temporary directory: {_reason(exc)}"
        return result
    try:
        _run(workdir, code, entry_point, list(calls), max(0.0, float(timeout_s)), _resolve(python or _default_python()), result, start)
    finally:
        if not _remove_tree(workdir):
            result.notes.append("the temporary directory could not be removed completely")
    return result


def _run(workdir: Path, code: str, entry_point: str, calls: list[str], timeout_s: float, python: str, result: SandboxResult, start: float) -> None:
    """One run, filling `result`. Every failure to start becomes result.infra_error."""
    try:
        (workdir / _RUNNER).write_text(CHILD_SOURCE, encoding="utf-8")
        job = json.dumps({"code": code, "entry_point": entry_point, "calls": calls}).encode("ascii")
        read_fd, write_fd = os.pipe()
    except (OSError, TypeError, ValueError) as exc:
        result.infra_error = f"could not prepare the sandbox: {_reason(exc)}"
        return
    apply_limits, notes = _limits(timeout_s)
    result.notes.extend(notes)
    try:
        proc = subprocess.Popen(
            [python, "-I", "-S", _RUNNER, str(write_fd)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=workdir,
            env=child_env(workdir),
            pass_fds=(write_fd,),
            start_new_session=True,
            preexec_fn=apply_limits,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        os.close(read_fd)
        result.infra_error = f"could not start the interpreter {Path(python).name!r}: {_reason(exc)}"
        return
    finally:
        os.close(write_fd)

    result.started = True
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    out_fd, err_fd = proc.stdout.fileno(), proc.stderr.fileno()
    sinks = {read_fd: _Sink(keep_head=True), out_fd: _Sink(), err_fd: _Sink()}
    try:
        result.timed_out = _pump(proc, sinks, job, start + timeout_s)
    finally:
        _kill_group(proc)
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            result.notes.append("the child did not end after SIGKILL")
        result.exit_code = proc.returncode
        result.duration_s = time.monotonic() - start
        if not _drain(sinks, _DRAIN_S):
            result.notes.append("a process outside the sandbox's group kept its pipes open; their rest was not read")
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            _close_quietly(stream)
        os.close(read_fd)

    results, out_pipe, err_pipe = sinks[read_fd], sinks[out_fd], sinks[err_fd]
    if results.dropped:
        result.notes.append(f"results: {results.dropped} bytes after the first {MAX_OUTPUT_BYTES} discarded")
    result.stdout_tail = _tail("stdout", out_pipe, workdir / _STDOUT, result.notes)
    result.stderr_tail = _tail("stderr", err_pipe, workdir / _STDERR, result.notes)
    if result.exit_code is not None and result.exit_code < 0 and not result.timed_out:
        result.notes.append(f"the child was killed by {_signal_name(-result.exit_code)}")
    ready = _parse_results(bytes(results.data), len(calls), result)
    if not ready:
        result.records.clear()
        result.duplicates.clear()
        result.malformed, result.load_error = 0, None
        if result.timed_out:
            result.infra_error = "the sandbox runner did not start before the deadline"
        else:
            last = result.stderr_tail.strip().splitlines()[-1:] or [""]
            result.infra_error = f"the sandbox runner did not start (exit status {result.exit_code}) {last[0][:200]}".rstrip()


# ---- the parent's side of the pipes ----------------------------------------------------------


class _Sink:
    """The bytes read from one pipe: at most MAX_OUTPUT_BYTES of them, the first ones (the
    results, whose order matters) or the last ones (output, whose end explains most)."""

    def __init__(self, keep_head: bool = False) -> None:
        self.keep_head = keep_head
        self.data = bytearray()
        self.total = 0
        self.eof = False

    def add(self, chunk: bytes) -> None:
        self.total += len(chunk)
        if self.keep_head:
            room = MAX_OUTPUT_BYTES - len(self.data)
            if room > 0:
                self.data += chunk[:room]
        else:
            self.data += chunk
            if len(self.data) > MAX_OUTPUT_BYTES:
                del self.data[: len(self.data) - MAX_OUTPUT_BYTES]

    @property
    def dropped(self) -> int:
        return self.total - len(self.data)


def _pump(proc: subprocess.Popen[bytes], sinks: dict[int, _Sink], job: bytes, deadline: float) -> bool:
    """Feed the job to the child's stdin and read its pipes until it exits (False) or the
    deadline hits (True). Never blocks on a pipe: everything goes through one selector."""
    assert proc.stdin is not None
    selector = selectors.DefaultSelector()
    stdin_fd = proc.stdin.fileno()
    pending = memoryview(job)
    try:
        for fd in sinks:
            selector.register(fd, selectors.EVENT_READ)
        os.set_blocking(stdin_fd, False)
        selector.register(stdin_fd, selectors.EVENT_WRITE)
        while True:
            if _exited(proc):
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            wait = min(_TICK_S, remaining)
            if not selector.get_map():
                time.sleep(wait)
                continue
            for key, _events in selector.select(wait):
                if key.fd == stdin_fd:
                    pending = _feed(stdin_fd, pending)
                    if not pending:
                        selector.unregister(stdin_fd)
                        _close_quietly(proc.stdin)
                elif not _read_into(key.fd, sinks[key.fd]):
                    selector.unregister(key.fd)
    finally:
        selector.close()


def _feed(fd: int, pending: memoryview) -> memoryview:
    """Write what the pipe takes now; the rest stays pending. A closed pipe drops it all."""
    try:
        written = os.write(fd, pending[:65536])
    except BlockingIOError:
        return pending
    except OSError:
        return pending[:0]
    return pending[written:]


def _read_into(fd: int, sink: _Sink) -> bool:
    """One read from a readable pipe into its sink; False at end of file."""
    try:
        chunk = os.read(fd, 65536)
    except BlockingIOError:
        return True
    except OSError:
        chunk = b""
    if not chunk:
        sink.eof = True
        return False
    sink.add(chunk)
    return True


def _drain(sinks: dict[int, _Sink], grace_s: float) -> bool:
    """Read what is left in the pipes once the process group is dead. False when a pipe was
    still open after grace_s (a process that left the group holds it)."""
    selector = selectors.DefaultSelector()
    try:
        for fd, sink in sinks.items():
            if not sink.eof:
                selector.register(fd, selectors.EVENT_READ)
        end = time.monotonic() + grace_s
        while selector.get_map():
            remaining = end - time.monotonic()
            if remaining <= 0:
                return False
            for key, _events in selector.select(remaining):
                if not _read_into(key.fd, sinks[key.fd]):
                    selector.unregister(key.fd)
        return True
    finally:
        selector.close()


def _exited(proc: subprocess.Popen[bytes]) -> bool:
    """Whether the child has exited, without reaping it where the platform allows."""
    if _CAN_PEEK:
        try:
            return os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
        except ChildProcessError:
            return True
        except OSError:
            pass
    return proc.poll() is not None


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    """SIGKILL the child's whole process group (its pgid is its pid: start_new_session)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass


def _limits(timeout_s: float) -> tuple[Callable[[], None] | None, list[str]]:
    """The preexec_fn that applies the resource limits, and notes on limits this platform lacks.
    The values are computed here, in the parent; the function only calls setrlimit, so it stays
    light in the forked child. A limit above the current hard limit is lowered to it."""
    if resource is None:
        return None, ["resource limits not applied: this platform has no resource module"]
    wanted = (
        ("RLIMIT_CPU", math.ceil(timeout_s) + 1),
        ("RLIMIT_AS", MEMORY_LIMIT_BYTES),
        ("RLIMIT_FSIZE", FILE_LIMIT_BYTES),
        ("RLIMIT_CORE", 0),
    )
    plan: list[tuple[int, int]] = []
    notes: list[str] = []
    for name, value in wanted:
        which = getattr(resource, name, None)
        if which is None:
            notes.append(f"{name} not applied: not available on this platform")
            continue
        try:
            _soft, hard = resource.getrlimit(which)
        except (OSError, ValueError):
            notes.append(f"{name} not applied: getrlimit failed")
            continue
        if hard != resource.RLIM_INFINITY:
            value = min(value, hard)
        plan.append((which, int(value)))
    setrlimit = resource.setrlimit

    def apply_limits() -> None:
        for which, value in plan:
            try:
                setrlimit(which, (value, value))
            except (OSError, ValueError):
                pass  # best effort: the platform refused this limit

    return apply_limits, notes


# ---- reading what the child left ------------------------------------------------------------


def _parse_results(data: bytes, n_calls: int, result: SandboxResult) -> bool:
    """Fill result.records, duplicates, malformed and load_error from the results pipe. Returns
    whether the runner reported ready. Every line must be exactly one record of the protocol:
    a call record before "ready", a second "ready" or load_error, an index out of range, a
    wrong key or type, a duplicate JSON key or a NaN all make a line malformed."""
    ready = False
    first: dict[int, CallRecord] = {}
    duplicates: set[int] = set()
    for line in data.split(b"\n"):
        if not line.strip():
            continue
        obj = _decode(line)
        kind = _kind(obj, n_calls)
        if kind == "ready" and not ready:
            ready = True
        elif kind is None or kind == "ready" or not ready:
            result.malformed += 1
        elif kind == "load_error":
            if result.load_error is None:
                result.load_error = obj["load_error"]  # type: ignore[index]
            else:
                result.malformed += 1
        else:
            assert obj is not None
            record = CallRecord(
                index=obj["i"], ok=obj["ok"], repr=obj.get("repr"), exc=obj.get("exc"),
                mro=tuple(obj.get("mro", ())), msg=str(obj.get("msg", ""))[:_MSG_CHARS],
            )
            if record.index in first:
                duplicates.add(record.index)
            else:
                first[record.index] = record
                result.records.append(record)
    result.duplicates = sorted(duplicates)
    return ready


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _decode(line: bytes) -> dict[str, Any] | None:
    try:
        obj = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_pairs, parse_constant=_no_constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    return obj if isinstance(obj, dict) else None


def _kind(obj: dict[str, Any] | None, n_calls: int) -> str | None:
    """"ready", "load_error" or "call" for a well-formed record, None for anything else."""
    if obj is None:
        return None
    keys = obj.keys()
    if keys == {"ready"}:
        return "ready" if obj["ready"] is True else None
    if keys == {"load_error"}:
        return "load_error" if type(obj["load_error"]) is str else None
    index = obj.get("i")
    if type(index) is not int or not 0 <= index < n_calls:
        return None
    if keys == {"i", "ok", "repr"}:
        good = obj["ok"] is True and type(obj["repr"]) is str and len(obj["repr"]) <= MAX_REPR_CHARS
    elif keys == {"i", "ok", "exc", "mro", "msg"}:
        mro = obj["mro"]
        good = (obj["ok"] is False and type(obj["exc"]) is str and type(obj["msg"]) is str
                and type(mro) is list and all(type(name) is str for name in mro))
    else:
        good = False
    return "call" if good else None


def _tail(name: str, pipe: _Sink, path: Path, notes: list[str]) -> str:
    """The last TAIL_CHARS of one stream: what reached the pipe before the child redirected it,
    then the file the code's own output went to."""
    data, size = _read_file_end(path)
    total, kept = pipe.total + size, len(pipe.data) + len(data)
    if total > kept:
        notes.append(f"{name}: {total - kept} of {total} bytes of output discarded (at most {MAX_OUTPUT_BYTES} kept)")
    text = (bytes(pipe.data) + data)[-4 * TAIL_CHARS:].decode("utf-8", errors="replace")
    return text[-TAIL_CHARS:]


def _read_file_end(path: Path) -> tuple[bytes, int]:
    """(the last MAX_OUTPUT_BYTES of a regular file, its size). The child may have replaced
    the file: a symlink is not followed, and a FIFO or directory is not read."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return b"", 0
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return b"", 0
        if info.st_size > MAX_OUTPUT_BYTES:
            os.lseek(fd, info.st_size - MAX_OUTPUT_BYTES, os.SEEK_SET)
        chunks, budget = [], MAX_OUTPUT_BYTES
        while budget > 0:
            chunk = os.read(fd, min(budget, 1 << 16))
            if not chunk:
                break
            chunks.append(chunk)
            budget -= len(chunk)
        return b"".join(chunks), info.st_size
    except OSError:
        return b"", 0
    finally:
        os.close(fd)


def _remove_tree(path: Path) -> bool:
    """Remove the work directory, also when the code made parts of it unreadable. Symbolic
    links are removed, never followed. Returns whether it is gone."""
    shutil.rmtree(path, ignore_errors=True)
    if not os.path.lexists(path):
        return True
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    for root, dirs, _files in os.walk(path):
        for name in dirs:
            child = os.path.join(root, name)
            if not os.path.islink(child):
                try:
                    os.chmod(child, 0o700)
                except OSError:
                    pass
    shutil.rmtree(path, ignore_errors=True)
    return not os.path.lexists(path)


# ---- small helpers ----------------------------------------------------------------------------


def _resolve(python: str) -> str:
    """An absolute interpreter path: the child's working directory and PATH are not ours."""
    if os.sep in python:
        return os.path.abspath(python)
    return shutil.which(python) or python


def _reason(exc: BaseException) -> str:
    """An OSError's reason without the file name (no absolute path in a message)."""
    if isinstance(exc, OSError) and exc.strerror:
        return exc.strerror
    return f"{type(exc).__name__}: {exc}"


def _signal_name(number: int) -> str:
    try:
        return signal.Signals(number).name
    except ValueError:
        return f"signal {number}"


def _close_quietly(stream: Any) -> None:
    try:
        stream.close()
    except OSError:
        pass


def _default_python() -> str:
    return sys.executable or "python3"


def _posix() -> bool:
    return os.name == "posix"
