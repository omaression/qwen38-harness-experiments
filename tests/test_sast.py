"""sast: the rule table, every rule's positive and negative cases, name resolution, shadowing,
the regex fallback, added_findings and describe()."""

from __future__ import annotations

import textwrap
import threading

import pytest

from llm_eval_harness import sast, util
from llm_eval_harness.sast import BLOCK, FLAG

from conftest import TINY_TASKS


def rules_of(code: str) -> list[str]:
    return [f.rule for f in sast.scan(textwrap.dedent(code))]


# ---- the rule table ---------------------------------------------------------------------------


def test_rule_table_is_the_documented_one() -> None:
    table = [(r.id, r.severity, r.cwe) for r in sast.RULES]
    assert table == [
        ("eval-exec", BLOCK, "CWE-95"),
        ("os-command", BLOCK, "CWE-78"),
        ("shell-true", BLOCK, "CWE-78"),
        ("unsafe-deserialization", BLOCK, "CWE-502"),
        ("process-control", BLOCK, None),
        ("interpreter-tamper", BLOCK, None),
        ("subprocess", FLAG, "CWE-78"),
        ("dynamic-import", FLAG, "CWE-470"),
        ("file-write", FLAG, "CWE-73"),
        ("network", FLAG, None),
        ("introspection", FLAG, None),
        ("exit-call", FLAG, None),
    ]
    assert all(r.description for r in sast.RULES)


def test_describe_is_stable_and_fingerprints_the_rules() -> None:
    first, second = sast.describe(), sast.describe()
    assert first == second
    assert first["ruleset"] == sast.RULESET == "sast/1"
    assert first["rules"] == [{"id": r.id, "severity": r.severity, "cwe": r.cwe} for r in sast.RULES]
    assert first["fingerprint"] == util.sha256_json(first["rules"])
    assert len(str(first["fingerprint"])) == 64


# ---- one positive case (at least) per rule -------------------------------------------------

POSITIVE = [
    ("eval-exec", "eval(s)"),
    ("eval-exec", "exec(code, {})"),
    ("eval-exec", "list(map(eval, xs))"),
    ("os-command", "import os\nos.system(cmd)"),
    ("os-command", "import os\nos.popen(cmd).read()"),
    ("os-command", "import os\nos.execv('/bin/sh', ['sh'])"),
    ("os-command", "import os\nos.spawnlp(os.P_WAIT, 'ls', 'ls')"),
    ("os-command", "import os\nos.posix_spawn('/bin/ls', ['ls'], {})"),
    ("os-command", "import pty\npty.spawn('sh')"),
    ("shell-true", "import subprocess\nsubprocess.run(cmd, shell=True)"),
    ("shell-true", "helper(cmd, shell=True)"),
    ("shell-true", "helper(cmd, **{'shell': True})"),
    ("unsafe-deserialization", "import pickle\npickle.loads(blob)"),
    ("unsafe-deserialization", "import pickle\npickle.load(handle)"),
    ("unsafe-deserialization", "import cPickle\ncPickle.loads(blob)"),
    ("unsafe-deserialization", "import _pickle\n_pickle.loads(blob)"),
    ("unsafe-deserialization", "import dill\ndill.loads(blob)"),
    ("unsafe-deserialization", "import marshal\nmarshal.loads(blob)"),
    ("unsafe-deserialization", "import shelve\nshelve.open(path)"),
    ("unsafe-deserialization", "import yaml\nyaml.unsafe_load(text)"),
    ("unsafe-deserialization", "import yaml\nyaml.load(text)"),
    ("unsafe-deserialization", "import yaml\nyaml.load(text, Loader=yaml.Loader)"),
    ("unsafe-deserialization", "import yaml\nyaml.load_all(text, yaml.FullLoader)"),
    ("process-control", "import os\nos._exit(0)"),
    ("process-control", "import os\nos.kill(pid, 9)"),
    ("process-control", "import os\nos.killpg(pid, 9)"),
    ("process-control", "import atexit\natexit.register(cleanup)"),
    ("process-control", "import signal\nsignal.signal(signal.SIGALRM, handler)"),
    ("process-control", "import signal\nsignal.alarm(1)"),
    ("interpreter-tamper", "import sys\nsys.settrace(tracer)"),
    ("interpreter-tamper", "import sys\nsys.setprofile(tracer)"),
    ("interpreter-tamper", "import threading\nthreading.settrace(tracer)"),
    ("interpreter-tamper", "import threading\nthreading.setprofile(tracer)"),
    ("interpreter-tamper", "import sys\nsys.modules['pytest'] = fake"),
    ("interpreter-tamper", "import sys\ndel sys.modules['json']"),
    ("interpreter-tamper", "import sys\nsys.modules.pop('json')"),
    ("interpreter-tamper", "import sys\nsys.modules.update({'json': fake})"),
    ("interpreter-tamper", "import sys\nsys.modules.setdefault('json', fake)"),
    ("interpreter-tamper", "import sys\nsys.modules = {}"),
    ("interpreter-tamper", "import builtins\nbuiltins.len = fake_len"),
    ("interpreter-tamper", "__builtins__['len'] = fake_len"),
    ("interpreter-tamper", "import builtins\nsetattr(builtins, 'len', fake_len)"),
    ("interpreter-tamper", "import builtins\ndelattr(builtins, 'len')"),
    ("subprocess", "import subprocess\nsubprocess.run(['ls'])"),
    ("subprocess", "import subprocess\nsubprocess.Popen(['ls']).wait()"),
    ("subprocess", "from subprocess import check_output\ncheck_output(['ls'])"),
    ("dynamic-import", "__import__('json')"),
    ("dynamic-import", "import importlib\nimportlib.import_module('json')"),
    ("file-write", "open(path, 'w')"),
    ("file-write", "open(path, mode='a')"),
    ("file-write", "open(path, 'r+')"),
    ("file-write", "open(path, 'xb')"),
    ("file-write", "import io\nio.open(path, 'w')"),
    ("file-write", "from pathlib import Path\nPath(p).write_text('x')"),
    ("file-write", "target.write_bytes(b'x')"),
    ("file-write", "import os\nos.remove(path)"),
    ("file-write", "import os\nos.unlink(path)"),
    ("file-write", "import os\nos.rename(a, b)"),
    ("file-write", "import os\nos.replace(a, b)"),
    ("file-write", "import os\nos.rmdir(path)"),
    ("file-write", "import shutil\nshutil.rmtree(path)"),
    ("file-write", "import shutil\nshutil.move(a, b)"),
    ("network", "import socket"),
    ("network", "import ssl"),
    ("network", "import urllib.request"),
    ("network", "from urllib.request import urlopen"),
    ("network", "from urllib import request"),
    ("network", "from http import client"),
    ("network", "import http.client as hc"),
    ("network", "import ftplib, smtplib, telnetlib"),
    ("network", "import requests"),
    ("network", "import httpx"),
    ("network", "__import__('socket')"),
    ("introspection", "import sys\nsys._getframe(1)"),
    ("introspection", "import inspect\ninspect.currentframe()"),
    ("introspection", "import inspect\ninspect.stack()"),
    ("introspection", "import inspect\ninspect.getouterframes(frame)"),
    ("introspection", "import gc\ngc.get_objects()"),
    ("introspection", "import gc\ngc.get_referrers(x)"),
    ("exit-call", "import sys\nsys.exit(1)"),
    ("exit-call", "exit()"),
    ("exit-call", "quit(0)"),
]


@pytest.mark.parametrize("rule, code", POSITIVE, ids=[f"{r}:{c.splitlines()[-1][:30]}" for r, c in POSITIVE])
def test_each_rule_matches(rule: str, code: str) -> None:
    assert rule in rules_of(code)


def test_every_rule_has_a_positive_case() -> None:
    assert {rule for rule, _ in POSITIVE} == {r.id for r in sast.RULES}


def test_a_decorator_reference_is_a_finding() -> None:
    code = """
        import atexit

        @atexit.register
        def goodbye():
            pass
    """
    assert rules_of(code) == ["process-control"]


NEGATIVE = [
    ("eval-exec", "import ast\nast.literal_eval(s)"),
    ("eval-exec", "from ast import literal_eval\nliteral_eval(s)"),
    ("eval-exec", "x = 'eval(s)'  # eval(s) in a string and a comment"),
    ("eval-exec", "evaluate(s)\nmy_exec(s)"),
    ("os-command", "import os\nos.path.join(a, b)"),
    ("os-command", "import os\nos.getcwd()"),
    ("shell-true", "import subprocess\nsubprocess.run(['ls'], shell=False)"),
    ("unsafe-deserialization", "import pickle\npickle.dumps(obj)"),
    ("unsafe-deserialization", "import json\njson.loads(text)"),
    ("unsafe-deserialization", "import yaml\nyaml.safe_load(text)"),
    ("unsafe-deserialization", "import yaml\nyaml.load(text, Loader=yaml.SafeLoader)"),
    ("unsafe-deserialization", "import yaml\nyaml.load(text, yaml.CSafeLoader)"),
    ("unsafe-deserialization", "import yaml\nfrom yaml import BaseLoader\nyaml.load(text, Loader=BaseLoader)"),
    ("process-control", "import os\nos.getpid()"),
    ("interpreter-tamper", "import sys\nx = sys.modules.get('json')"),
    ("interpreter-tamper", "import builtins\nn = builtins.len(xs)"),
    ("interpreter-tamper", "import sys\nif sys.modules == {}: pass"),
    ("file-write", "open(path)"),
    ("file-write", "open(path, 'rb')"),
    ("file-write", "open(path, mode)"),
    ("network", "import urllib.parse"),
    ("network", "from urllib import parse"),
    ("network", "from urllib.parse import quote"),
    ("network", "import json, re, math"),
    ("introspection", "import inspect\ninspect.signature(f)"),
    ("exit-call", "import sys\nsys.exc_info()"),
]


@pytest.mark.parametrize("rule, code", NEGATIVE, ids=[f"{r}:{c.splitlines()[-1][:30]}" for r, c in NEGATIVE])
def test_lookalikes_do_not_match(rule: str, code: str) -> None:
    assert rule not in rules_of(code)


# ---- name resolution ---------------------------------------------------------------------------

RESOLVED = [
    ("import alias", "import os as o\no.system(c)", "os-command"),
    ("import of a submodule binds the package", "import os.path\nos.system(c)", "os-command"),
    ("from-import alias", "from os import system as run\nrun(c)", "os-command"),
    ("from-import", "from pickle import loads\nloads(b)", "unsafe-deserialization"),
    ("star import", "from os import *\nsystem(c)", "os-command"),
    ("star import of subprocess", "from subprocess import *\nrun(['ls'])", "subprocess"),
    ("assignment of a builtin", "e = eval\ne(s)", "eval-exec"),
    ("assignment of a function", "import pickle\np = pickle.loads\np(b)", "unsafe-deserialization"),
    ("chained assignments", "import os\na = os\nb = a\nc = b.system\nc(x)", "os-command"),
    ("annotated assignment", "import os\nrun: object = os.system\nrun(c)", "os-command"),
    ("walrus", "import os\nif (run := os.system):\n    run(c)", "os-command"),
    ("self-assignment at module level", "eval = eval\neval(s)", "eval-exec"),
    ("alias swapped back", "e = eval\neval = e\neval(s)", "eval-exec"),
    ("parameter default", "import os\ndef f(c, run=os.system):\n    return run(c)", "os-command"),
    ("lambda default", "f = lambda s, e=eval: e(s)", "eval-exec"),
    ("getattr on a module", "import os\ngetattr(os, 'system')(c)", "os-command"),
    ("getattr on builtins", "import builtins\ngetattr(builtins, 'exec')(s)", "eval-exec"),
    ("getattr on __builtins__", "getattr(__builtins__, 'eval')(s)", "eval-exec"),
    ("builtins module", "import builtins\nbuiltins.eval(s)", "eval-exec"),
    ("builtins alias", "import builtins as b\nb.exec(s)", "eval-exec"),
    ("from builtins", "from builtins import eval as ev\nev(s)", "eval-exec"),
    ("__builtins__ attribute", "__builtins__.eval(s)", "eval-exec"),
    ("__builtins__ subscript", "__builtins__['exec'](s)", "eval-exec"),
    ("sys.modules lookup", "import sys\nsys.modules['os'].system(c)", "os-command"),
    ("__import__ with a constant", "__import__('os').system(c)", "os-command"),
    ("import_module with a constant", "import importlib\nimportlib.import_module('os').popen(c)", "os-command"),
    ("posix is os", "import posix\nposix.system(c)", "os-command"),
    ("import inside a function", "def f(c):\n    import os\n    return os.system(c)", "os-command"),
    ("module import used in a function", "import os\ndef f(c):\n    return os.system(c)", "os-command"),
    ("global declaration", "def f():\n    global run\n    import os\n    run = os.system\ndef g(c):\n    return run(c)", "os-command"),
    ("sound within a scope", "import os\nos = Wrapper()\nos.system(c)", "os-command"),
    ("module name used without import", "os.system(c)", "os-command"),
    ("builtin outside a function with that parameter", "def f(eval):\n    return eval\nx = eval('1')", "eval-exec"),
    ("class attribute is not visible in methods", "class C:\n    eval = None\n    def m(self, s):\n        return eval(s)", "eval-exec"),
    ("comprehension variable stays inside", "[0 for eval in ()]\neval(s)", "eval-exec"),
    ("passed as a callback", "import os\nlist(map(os.system, cmds))", "os-command"),
]


@pytest.mark.parametrize("label, code, rule", RESOLVED, ids=[r[0] for r in RESOLVED])
def test_names_are_resolved(label: str, code: str, rule: str) -> None:
    assert rule in rules_of(code), label


SHADOWED = [
    ("a def named eval", "def eval(x):\n    return x\neval(s)"),
    ("a parameter named eval", "def f(eval, s):\n    return eval(s)"),
    ("a lambda parameter named exec", "f = lambda exec: exec(1)"),
    ("an attribute named eval", "obj.eval(s)\nself.exec(code)"),
    ("a method named eval", "class C:\n    def eval(self, s):\n        return s\nC().eval(s)"),
    ("system on an object", "obj.system(c)\nobj.popen(c)"),
    ("system on a parameter named os", "def f(os, c):\n    return os.system(c)"),
    ("os shadowed by a parameter in a function", "import os\ndef f(os, c):\n    return os.system(c)"),
    ("a local system = print", "from os import system\ndef f():\n    system = print\n    system('x')"),
    ("system without any import", "system(c)\npopen(c)"),
    ("a loop variable named eval", "for eval in handlers:\n    eval(s)"),
    ("a with target named exec", "with ctx() as exec:\n    exec(s)"),
    ("an except name", "try:\n    pass\nexcept Exception as eval:\n    eval(s)"),
    ("an import from another module", "from mylib import eval\neval(s)"),
    ("a relative import", "from .helpers import system\nsystem(c)"),
    ("a class named exec", "class exec:\n    pass\nexec(1)"),
    ("getattr with a computed name", "import os\ngetattr(os, 'sys' + 'tem')(c)"),
    ("a shadowed getattr", "import os\ndef getattr(o, n):\n    return print\ngetattr(os, 'system')(c)"),
]


@pytest.mark.parametrize("label, code", SHADOWED, ids=[s[0] for s in SHADOWED])
def test_rebound_names_are_not_the_builtin_or_the_import(label: str, code: str) -> None:
    found = rules_of(code)
    assert "eval-exec" not in found and "os-command" not in found, (label, found)


def test_a_long_chain_of_aliases_is_followed_in_either_order() -> None:
    links = [f"a{k + 1} = a{k}\n" for k in range(200)]
    forward = "a0 = eval\n" + "".join(links) + "a200(s)\n"
    backward = "a200(s)\n" + "".join(reversed(links)) + "a0 = eval\n"
    for code in (forward, backward):
        assert "a200(s)" in [f.snippet for f in sast.scan(code) if f.rule == "eval-exec"]


def test_alias_cycles_terminate() -> None:
    assert rules_of("a = b\nb = a\na(s)\n") == []
    assert "os-command" in rules_of("import os\na = b\nb = a\nb = os.system\na(c)\n")


def test_a_match_capture_is_a_binding() -> None:
    code = "match value:\n    case [eval]:\n        eval(s)\n"
    assert "eval-exec" not in rules_of(code)


# ---- findings: snippets, lines, order ---------------------------------------------------------


def test_findings_carry_line_snippet_and_mode_and_are_sorted() -> None:
    code = "import os, subprocess\n\n\ndef f(c):\n    subprocess.run(c, shell=True)\n    return os.system(c)\n"
    found = sast.scan(code)
    assert [(f.rule, f.line, f.snippet, f.mode) for f in found] == [
        ("shell-true", 5, "subprocess.run(c, shell=True)", "ast"),
        ("subprocess", 5, "subprocess.run(c, shell=True)", "ast"),
        ("os-command", 6, "os.system(c)", "ast"),
    ]
    assert found[0].to_json() == {"rule": "shell-true", "severity": BLOCK, "cwe": "CWE-78", "line": 5, "snippet": "subprocess.run(c, shell=True)", "mode": "ast"}


def test_snippets_collapse_whitespace_and_are_capped() -> None:
    code = "eval(\n    'x' +\n    'y'\n)\n" + "eval('" + "a" * 500 + "')\n"
    snippets = [f.snippet for f in sast.scan(code)]
    assert snippets[0] == "eval('x' + 'y')"
    assert len(snippets[1]) == sast.SNIPPET_CHARS


def test_writes_into_sys_modules_are_quoted_as_the_statement() -> None:
    assert [f.snippet for f in sast.scan("import sys\nsys.modules['x'] = 1\n")] == ["sys.modules['x'] = 1"]


# ---- the regex fallback ----------------------------------------------------------------------


def test_code_that_does_not_parse_is_scanned_with_regexes() -> None:
    code = "def f(:\n    eval(x)\n    import os; os.system('ls')\n    pickle.loads(b)\n    subprocess.run(c)\n"
    found = sast.scan(code)
    assert {f.mode for f in found} == {"regex"}
    assert [(f.rule, f.line) for f in found] == [("eval-exec", 2), ("os-command", 3), ("unsafe-deserialization", 4)]
    assert all(f.severity == BLOCK for f in found), "flag rules do not run in regex mode"


def test_regex_mode_sees_comments_and_strings() -> None:
    found = sast.scan("def f(:\n    pass  # eval(x)\n")
    assert [(f.rule, f.snippet) for f in found] == [("eval-exec", "eval(")]


@pytest.mark.parametrize("code", [
    "def broken(:\n    obj.eval(x)\n    def eval(y): pass\n    literal_eval(s)\n",
    "def broken(:\n    yaml.load(s, Loader=yaml.SafeLoader)\n    sys.modules == {}\n",
])
def test_regex_mode_negatives(code: str) -> None:
    assert sast.scan(code) == []


@pytest.mark.parametrize("code, rule", [
    ("def broken(:\n    __builtins__['eval'](s)\n", "eval-exec"),
    ("def broken(:\n    shell=True\n", "shell-true"),
    ("def broken(:\n    yaml.load(s)\n", "unsafe-deserialization"),
    ("def broken(:\n    atexit.register(f)\n", "process-control"),
    ("def broken(:\n    sys.modules['x'] = m\n", "interpreter-tamper"),
    ("def broken(:\n    del sys.modules['x']\n", "interpreter-tamper"),
    ("def broken(:\n    builtins.len = f\n", "interpreter-tamper"),
    ("def broken(:\n    pty.spawn('sh')\n", "os-command"),
])
def test_regex_mode_positives(code: str, rule: str) -> None:
    assert rule in [f.rule for f in sast.scan(code)]


# ---- never raises ------------------------------------------------------------------------------


@pytest.mark.parametrize("code", [
    "",
    "   \n\n",
    "\x00eval(x)",
    "x = (" * 500 + "1" + ")" * 500,
    "+".join(["1"] * 20_000),
    "[" * 3000 + "]" * 3000,
    "def f():\n    return '\\N{SNOWMAN}' + 'é' + '\udcff'\n",
    "\ufeffimport os\nos.system(c)\n",
    "if True:\n\tpass\n        pass\n",
])
def test_scan_never_raises(code: str) -> None:
    for finding in sast.scan(code):
        assert finding.rule in {r.id for r in sast.RULES}


def test_non_text_input_has_no_findings() -> None:
    assert sast.scan(None) == []  # type: ignore[arg-type]
    assert sast.scan(123) == []  # type: ignore[arg-type]
    assert sast.added_findings("x = 1", None) == []  # type: ignore[arg-type]


def test_scans_in_threads_agree() -> None:
    code = "import os\ndef f(c):\n    return os.system(c)\nx = '\\d'\n"  # an invalid escape warns while parsing
    expected = sast.scan(code)
    results: list[object] = []
    threads = [threading.Thread(target=lambda: results.append(sast.scan(code))) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [expected] * 8


# ---- added findings and blocking ---------------------------------------------------------------


def test_added_findings_is_a_multiset_difference() -> None:
    original = "import os\ndef f(c):\n    os.system(c)\n    return 1\n"
    same = "import os\ndef f(c):\n    os.system(c)\n    return 2\n"
    one_more = "import os\ndef f(c):\n    os.system(c)\n    os.system(c)\n    return 1\n"
    other_call = "import os\ndef f(c):\n    os.system(c + ' -l')\n    return 1\n"
    assert sast.added_findings(original, same) == []
    added = sast.added_findings(original, one_more)
    assert [(f.rule, f.snippet) for f in added] == [("os-command", "os.system(c)")]
    assert [f.snippet for f in sast.added_findings(original, other_call)] == ["os.system(c + ' -l')"]
    assert sast.added_findings(one_more, original) == []


def test_added_findings_of_a_clean_original_are_all_findings() -> None:
    patched = "import pickle, sys\ndef f(b):\n    sys.exit(pickle.loads(b))\n"
    assert sast.added_findings("def f(b):\n    return b\n", patched) == sast.scan(patched)


def test_blocking_keeps_block_findings() -> None:
    found = sast.scan("import os, sys\nos.system(c)\nsys.exit(0)\n")
    assert [f.rule for f in sast.blocking(found)] == ["os-command"]
    assert {f.rule for f in found} == {"os-command", "exit-call"}


# ---- the admission test on the tiny fixture -----------------------------------------------------


@pytest.mark.parametrize("task", TINY_TASKS, ids=[t["id"] for t in TINY_TASKS])
def test_no_block_finding_in_the_tiny_tasks(task: dict) -> None:
    for field in ("buggy_code", "reference_code", "wrong_code"):
        assert sast.blocking(sast.scan(task[field])) == [], field
        assert all(f.mode == "ast" for f in sast.scan(task[field]))
