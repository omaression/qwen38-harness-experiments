"""The repository's fixed contracts, checked from the first commit on:

  - every module imports, and imports nothing but the standard library and this package;
  - the command lines that CI, the README and docs/DESIGN.md use parse against cli.build_parser;
  - the shared records behave as documented (request hash, round trips);
  - the util helpers that hashes depend on;
  - no tracked text file holds a home-directory path, an e-mail address or a key-like token.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import sys
from pathlib import Path

import pytest

import llm_eval_harness
from llm_eval_harness import cli, util
from llm_eval_harness.records import ChatRequest, ChatResponse, ItemResult, Message, ModelSettings

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "llm_eval_harness"
MODULES = sorted(p.stem for p in PACKAGE.glob("*.py") if p.stem != "__main__")


# ---- imports ------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(f"llm_eval_harness.{name}" if name != "__init__" else "llm_eval_harness")


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_package_uses_only_the_standard_library(path: Path) -> None:
    allowed = set(sys.stdlib_module_names) | {"__future__", "llm_eval_harness"}
    foreign = _imported_roots(path) - allowed
    assert not foreign, f"{path.name} imports {sorted(foreign)}: the harness is standard library only"


# ---- the command-line contract ----------------------------------------------------------------

DOCUMENTED_COMMANDS = [
    "tasks validate --suite tasks/demo",
    "tasks validate --suite tasks/demo --no-exec --timeout 5",
    "tasks split --suite tasks/demo",
    "run --feature bugfix --cassettes cassettes/candidate --split dev --items-out reports/candidate.items.jsonl",
    "run --feature judge --cassettes cassettes/baseline --split all --workers 2 --timeout 5",
    "record --feature bugfix --model my-model --out artifacts/cassettes/live-candidate --split dev --prompt bugfix/v2",
    "record --feature judge --model my-model --out artifacts/cassettes/live-baseline --prompt-from cassettes/baseline --base-url http://localhost:8000/v1",
    """record --feature bugfix --model m --out d --seed none --extra '{"chat_template_kwargs": {"enable_thinking": true}}'""",
    "gate --feature bugfix --suite tasks/demo --split heldout --candidate cassettes/candidate --baseline @current --ledger ledger/ledger.jsonl --items-out reports/items",
    "gate --feature judge --candidate cassettes/candidate-regressed --baseline cassettes/baseline",
    "gate --feature bugfix --candidate cassettes/candidate --baseline none --floor 0.5",
    "gate --feature bugfix --candidate c --baseline b --margin 0.02 --tie fail --alpha 0.1 --require-gain --resamples 500 --seed 3",
    "gate --feature bugfix --candidate c --baseline b --schema-min 0.9 --think-min 0 --max-new-sast 1 --allow-missing --record-evaluation",
    "summary reports/bugfix-candidate.json reports/bugfix-regressed.json --title Gate",
    "ledger verify --ledger ledger/ledger.jsonl",
    "ledger status",
    "ledger current --feature bugfix",
    "ledger promote --feature bugfix --to cassettes/candidate --report reports/bugfix-candidate.json --reason better",
    "ledger promote --feature judge --to cassettes/baseline --initial --reason first",
    "ledger rollback --feature bugfix --reason regression-in-production",
]


@pytest.mark.parametrize("command", DOCUMENTED_COMMANDS)
def test_documented_command_lines_parse(command: str) -> None:
    import shlex

    cli.build_parser().parse_args(shlex.split(command))


# Every command the README and docs/DESIGN.md show, found in their text, so that a renamed or
# misspelt flag in the documentation fails here rather than in a reader's shell.
DOCS = ("README.md", "docs/DESIGN.md")
_SHELL_BLOCK = re.compile(r"^```(?:bash|sh)\n(.*?)^```$", re.S | re.M)
_INLINE_COMMAND = re.compile(r"(?<!`)`(llm-eval [^`]+?)`(?!`)")


def _shell_words(text: str) -> list[str]:
    """The words of a shell script: comments dropped, `\\` line continuations joined, quotes
    removed, and operators (`>`, `|`, `$(`, and `;` for the end of a line) as words of their own.
    A quoted string may span lines."""
    import shlex

    out, quote, k = [], "", 0
    while k < len(text):
        ch = text[k]
        if quote:
            if ch == quote:
                quote = ""
            elif ch == "\\" and quote == '"':
                out.append(text[k : k + 2])
                k += 2
                continue
        elif ch in "'\"":
            quote = ch
        elif ch == "\\" and text[k + 1 : k + 2] == "\n":
            out.append(" ")
            k += 2
            continue
        elif ch == "#" and (k == 0 or text[k - 1] in " \t\n"):
            end = text.find("\n", k)
            k = len(text) if end < 0 else end
            continue
        elif ch == "\n":
            out.append(" ; ")
            k += 1
            continue
        out.append(ch)
        k += 1
    lexer = shlex.shlex("".join(out), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _commands_in(words: list[str]) -> list[list[str]]:
    """The argument lists of `llm-eval ...` and `python -m llm_eval_harness ...`, each up to the
    next shell operator."""
    out, k = [], 0
    while k < len(words):
        if words[k] == "llm-eval":
            start = k + 1
        elif words[k] == "-m" and words[k + 1 : k + 2] == ["llm_eval_harness"]:
            start = k + 2
        else:
            k += 1
            continue
        end = start
        while end < len(words) and not set(words[end]) <= set("|&;<>()"):
            end += 1
        out.append(words[start:end])
        k = end
    return out


def _concrete(args: list[str]) -> list[str]:
    """Shell variables replaced by a value of the kind their option takes."""
    kinds = {"--feature": "bugfix", "--split": "dev"}
    return [kinds.get(args[k - 1], "x") if "$" in arg and k else arg for k, arg in enumerate(args)]


def _subcommand_parser(words: list[str]):
    import argparse

    parser = cli.build_parser()
    for word in words:
        sub = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        if word.startswith("-") or not sub or word not in sub[0].choices:
            break
        parser = sub[0].choices[word]
    return parser


@pytest.mark.parametrize("doc", DOCS)
def test_every_command_in_the_docs_parses(doc: str) -> None:
    text = (ROOT / doc).read_text(encoding="utf-8")
    commands = [args for block in _SHELL_BLOCK.findall(text) for args in _commands_in(_shell_words(block))]
    assert commands, f"no llm-eval command found in the shell blocks of {doc}"
    for args in commands:
        try:
            cli.build_parser().parse_args(_concrete(args))
        except SystemExit:
            pytest.fail(f"{doc}: `llm-eval {' '.join(args)}` does not parse against cli.build_parser")
    spans = _INLINE_COMMAND.findall(text)
    for span in spans:  # often partial (`llm-eval gate --record-evaluation`): check the flags it names
        words = _shell_words(span)[1:]
        known = _subcommand_parser(words)._option_string_actions
        unknown = [w for w in (w.split("=", 1)[0] for w in words if w.startswith("--")) if w not in known]
        assert not unknown, f"{doc}: `{span}` names options its command does not have: {unknown}"


def test_the_docs_command_finder_finds_what_it_should() -> None:
    words = _shell_words('x=$(llm-eval ledger current --feature "$F") # llm-eval in a comment\n'
                         "llm-eval gate --feature \"$f\" \\\n  --candidate c --baseline b > r.json | jq '.a,\n .b'\n"
                         "llm-eval summary r.json --title 'Gate # 1'   # it's a comment\n"
                         "llm-eval tasks validate\n"
                         "docker run img python -m llm_eval_harness tasks split")  # fmt: skip
    assert _commands_in(words) == [
        ["ledger", "current", "--feature", "$F"],
        ["gate", "--feature", "$f", "--candidate", "c", "--baseline", "b"],
        ["summary", "r.json", "--title", "Gate # 1"],
        ["tasks", "validate"],
        ["tasks", "split"],
    ]
    assert _concrete(["gate", "--feature", "$f", "--candidate", "$DIR"]) == ["gate", "--feature", "bugfix", "--candidate", "x"]
    assert "--record-evaluation" in _subcommand_parser(["gate"])._option_string_actions
    assert "--record-evaluation" not in _subcommand_parser(["run"])._option_string_actions


@pytest.mark.parametrize(
    "command",
    [
        "gate --feature bugfix --candidate c",  # no --baseline
        "gate --feature other --candidate c --baseline b",  # unknown feature
        "gate --feature bugfix --cand c --baseline b",  # abbreviations are refused
        "gate --feature bugfix --candidate c --baseline b --margin -0.1",
        "gate --feature bugfix --candidate c --baseline b --tie maybe",
        "gate --feature bugfix --candidate c --baseline b --split all",  # the gate scores one split
        "record --feature bugfix --model m --out d --prompt bugfix/v2 --prompt-from cassettes/baseline",
        "record --feature bugfix --model m --out d --api-key sk-should-not-exist",  # keys never come from flags
        "ledger rollback --feature bugfix",  # a rollback needs a reason
    ],
)
def test_bad_command_lines_exit_2(command: str) -> None:
    import shlex

    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(shlex.split(command))
    assert exc.value.code == 2


def test_gate_defaults_match_the_documented_rule() -> None:
    args = cli.build_parser().parse_args(["gate", "--feature", "bugfix", "--candidate", "c", "--baseline", "b"])
    assert (args.split, args.margin, args.tie, args.alpha, args.schema_min, args.think_min, args.max_new_sast) == ("heldout", 0.0, "pass", 0.05, 0.95, 0.90, 0)
    assert args.require_gain is False and args.record_evaluation is False


def test_console_script_and_version() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'llm-eval = "llm_eval_harness.cli:main"' in text
    assert re.fullmatch(r"\d+\.\d+\.\d+", llm_eval_harness.__version__)


# ---- the shared records -----------------------------------------------------------------------


def _request(item_id: str = "bugfix:t-clamp", **settings: object) -> ChatRequest:
    return ChatRequest(item_id=item_id, model="m", messages=(Message("system", "s"), Message("user", "u")), settings=ModelSettings(**settings))


def test_request_hash_is_the_payload_hash_and_ignores_the_item_id() -> None:
    a, b = _request("bugfix:a"), _request("bugfix:b")
    assert a.sha256() == b.sha256() == util.sha256_text(util.canonical_json(a.payload()))
    assert "item_id" not in a.payload()
    assert _request(temperature=0.5).sha256() != a.sha256()
    assert _request(seed=None).payload().keys() == {"model", "messages", "temperature", "max_tokens"}


def test_settings_extra_may_not_override_the_body() -> None:
    with pytest.raises(ValueError):
        _request(extra={"model": "other"}).payload()
    assert _request(extra={"chat_template_kwargs": {"enable_thinking": True}}).payload()["chat_template_kwargs"] == {"enable_thinking": True}


def test_response_and_item_result_round_trip() -> None:
    response = ChatResponse(content="{}", reasoning=None, finish_reason="stop", usage={"completion_tokens": 3}, latency_s=0.5)
    assert ChatResponse.from_json(json.loads(json.dumps(response.to_json()))) == response
    row = ItemResult(item_id="bugfix:t", feature="bugfix", task_id="t", cluster="t", split="heldout", correct=True, schema_valid=True,
                     lenient_valid=True, think_present=False, think_valid=True, think_words=0, sast_block=("eval-exec",), problems=("x",))
    assert ItemResult.from_json(json.loads(json.dumps(row.to_json()))) == row


def test_tiny_fixture_tasks_serialise(tiny_tasks) -> None:
    for task in tiny_tasks:
        record = task.to_json()
        assert record["id"] == task.id and len(record["hidden_tests"]) >= 2
        assert json.loads(util.canonical_json(record)) == record


# ---- util -------------------------------------------------------------------------------------


def test_canonical_json_is_order_independent_and_refuses_nan() -> None:
    assert util.canonical_json({"b": 1, "a": [1, "é"]}) == '{"a":[1,"é"],"b":1}'
    with pytest.raises(ValueError):
        util.canonical_json({"x": float("nan")})


def test_jsonl_round_trip_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    util.write_jsonl(path, [{"b": 2, "a": 1}, {"c": "ü"}])
    assert util.read_jsonl(path) == [{"a": 1, "b": 2}, {"c": "ü"}]
    path.write_text('{"a": 1}\n[1, 2]\n', encoding="utf-8")
    with pytest.raises(util.JsonlError, match=r":2: "):
        util.read_jsonl(path)


def test_display_path_is_never_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a").mkdir()
    assert util.display_path(tmp_path / "a") == "a"
    assert util.display_path("a") == "a"
    shown = util.display_path(Path("/") / "somewhere" / "else")
    assert not shown.startswith("/") and shown.endswith("else")


# ---- cassette fingerprints (the ledger depends on them) ----------------------------------------


def test_cassette_fingerprint(tmp_path: Path) -> None:
    from llm_eval_harness.cassettes import CassetteError, fingerprint

    with pytest.raises(CassetteError):
        fingerprint(tmp_path)
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "bugfix.jsonl").write_text('{"item_id": "bugfix:a"}\n', encoding="utf-8")
    first = fingerprint(tmp_path)
    (tmp_path / "notes.txt").write_text("not part of the set\n", encoding="utf-8")
    assert fingerprint(tmp_path) == first
    (tmp_path / "bugfix.jsonl").write_text('{"item_id": "bugfix:b"}\n', encoding="utf-8")
    assert fingerprint(tmp_path) != first


# ---- public safety ----------------------------------------------------------------------------

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist", "reports", "artifacts"}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".jsonl", ".txt", ".cfg", ".ini", ".sh", ""}
FORBIDDEN = {
    "a home-directory path": re.compile(r"(?i)/(?:users|home)/[a-z0-9_.-]+/"),
    "an e-mail address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
    "an API key": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})"),
}


def _text_files() -> list[Path]:
    out = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            out.append(path)
    return out


def test_no_private_paths_addresses_or_keys_in_the_tree() -> None:
    found = []
    for path in _text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for what, pattern in FORBIDDEN.items():
            for match in pattern.finditer(text):
                found.append(f"{path.relative_to(ROOT)}: {what}: {match.group(0)[:40]}")
    assert not found, "\n".join(found)
