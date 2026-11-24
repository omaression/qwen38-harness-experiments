"""Import HumanEvalFix (the Python subset of bigcode/humanevalpack) as the task suite tasks/humanevalfix.

    python scripts/import_humanevalfix.py --download             # fetch the pinned parquet file
    python scripts/import_humanevalfix.py --source PATH          # a local .parquet or .jsonl copy
    python scripts/import_humanevalfix.py --source PATH --check  # rebuild in memory, compare bytes

Standard library only, like the harness. The parquet file is read by a small reader below (flat
schemas of string columns; PLAIN and dictionary encodings; data pages v1 and v2; uncompressed,
snappy or gzip pages), so neither pyarrow nor the `datasets` library is needed. --download fetches
the file of the pinned dataset revision over HTTPS with urllib and refuses it unless its sha256
is the pinned one. Run it from the repository root (or give --out); it imports the harness from
src/.

WHAT A HUMANEVALFIX ROW BECOMES (tasks/humanevalfix/README.md explains each choice)
  id              "Python/12" -> "python-12" (the dataset's id, in the harness's id alphabet)
  buggy_code      prompt + buggy_solution: imports, helpers, signature, docstring, buggy body
  reference_code  prompt + canonical_solution
  wrong_code      prompt + a guard + buggy_solution: the buggy body, preceded by an `if` that
                  returns each shown test's expected value for exactly that test's arguments.
                  The harness requires the judge's negative patch to pass every shown test and
                  fail a hidden one, and the buggy solution fails a shown test by construction,
                  so it cannot be the negative patch unchanged. The guard is the overfitted
                  patch the harness's wrong_code stands for, and it keeps the dataset's bug.
  shown_tests     the expressible asserts of `example_test` (the docstring's examples)
  hidden_tests    the expressible asserts of `test`, minus those whose call is a shown test or
                  appears in the docstring or in a rendered prompt (the model would see it)
  tags            bug_type and failure_symptoms, as slugs ("operator-misuse", "infinite-loop")
  group           shared by near-duplicate tasks, so that they land in one bucket: buggy code
                  the harness's twin detector calls near duplicates (tasks.find_twins' measure
                  and threshold), or docstrings at least DOCSTRING_TWIN_THRESHOLD similar

AN EXPRESSIBLE ASSERT is a top-level statement of the test's check() function of the form
`assert f(<args>) == <literal>` (either side, an assert message allowed), where f is the function
under test and every argument is a literal or arithmetic on literals (`3 * 19`). It becomes
{"call": "<entry>(<args>)", "expected": "<literal>"}: the harness evaluates the call in the
sandbox and compares the value with the literal by ==, as the dataset's assert does. Everything
else is dropped, per assert, with a reason: tolerance comparisons (abs(f(x) - y) < 1e-6), `is`
and truthiness asserts, arguments or expectations that are not literals, asserts inside loops, and
`assert True`. Exactness is then checked by execution (below).

VERIFICATION (skipped with --no-verify). Every candidate task runs in the harness's own checker
(verdicts.exec_verdict, the sandbox the README describes; dataset code runs nowhere else):
  - a hidden or shown test the reference fails under the harness's value rule is dropped (the
    conversion of that assert is not exact, e.g. a value whose repr is not a literal);
  - a task is skipped when the buggy code passes every shown test (the examples do not show the
    bug), when fewer than tasks.MIN_HIDDEN_TESTS hidden tests remain, when the buggy or the wrong
    code passes every hidden test, or when any code trips a block-class SAST rule.
The kept tasks then satisfy every rule of `llm-eval tasks validate`. The import report
(IMPORT.json) lists every skipped task and every dropped assert with its reason.

THE SPLIT. --heldout-pct (default 60) is the suite's heldout_pct. The script refuses a value that
gives fewer than --min-heldout (70) held-out tasks: a few-point effect needs that many paired items.
On the 120 kept tasks, 60 puts 74 in the held-out split and 46 in dev.
"""

from __future__ import annotations

import argparse
import difflib
import ast
import gzip
import hashlib
import json
import re
import struct
import sys
import tempfile
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_eval_harness import features, prompts, sast, tasks, verdicts  # noqa: E402
from llm_eval_harness.records import ALL, Suite, Task  # noqa: E402

#: The dataset revision this suite was built from, and its Python file.
DATASET = "bigcode/humanevalpack"
REVISION = "9a41762f73a8cb23bb5811b73d5aab164efcf378"
PARQUET_PATH = "python/test-00000-of-00001.parquet"
PARQUET_SHA256 = "ed5f15d789156e21222bfcd556c425a39042355c84ae1e8b058abd6a3d7f8075"
DOWNLOAD_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{PARQUET_PATH}"

DEFAULT_OUT = ROOT / "tasks" / "humanevalfix"
DEFAULT_HELDOUT_PCT = 60
DEFAULT_MIN_HELDOUT = 70
REPORT_FILE = "IMPORT.json"
REPORT_SCHEMA = "llm_eval_harness.import_report/1"
SUITE_NAME = "humanevalfix"
SUITE_VERSION = 1
WORKERS = 8
COLUMNS = ("task_id", "prompt", "canonical_solution", "buggy_solution", "bug_type", "failure_symptoms", "entry_point", "test", "example_test", "docstring")

DESCRIPTION = (
    "Derived from HumanEvalFix, the Python subset of the bigcode/humanevalpack dataset (MIT license; "
    f"revision {REVISION[:12]}, see NOTICE). Not synthetic: the functions, docstrings, human-written bugs "
    "and tests are the dataset's, converted by scripts/import_humanevalfix.py. buggy_code is the buggy "
    "solution with its signature and docstring, reference_code the canonical solution, shown_tests the "
    "dataset's example asserts, hidden_tests its full-test asserts that compare a call with a literal. "
    "wrong_code is the buggy solution behind a guard that returns the shown tests' expected values: an "
    "overfitted patch that keeps the dataset's bug. Tasks or asserts the format cannot express exactly "
    "are left out (IMPORT.json lists each with its reason). HumanEval is public and widely trained on, "
    "so held-out here means held out from prompt tuning, not unseen by a model."
)


class ConversionError(RuntimeError):
    """The source cannot be read or converted (a message for the user, exit 1)."""


# ==== a small parquet reader (standard library only) ==============================================
#
# Enough of the format for flat tables of string columns, which is what the Hugging Face hub
# serves for this dataset: the footer's thrift (compact protocol) metadata, column chunks made of
# an optional dictionary page and data pages (v1 or v2), definition levels for optional columns,
# PLAIN and RLE_DICTIONARY values, and UNCOMPRESSED, SNAPPY or GZIP pages. Anything else (nested
# columns, other physical types or codecs) is refused with a message, never misread.

_CT_STOP, _CT_TRUE, _CT_FALSE, _CT_BYTE, _CT_I16, _CT_I32, _CT_I64, _CT_DOUBLE, _CT_BINARY, _CT_LIST, _CT_SET, _CT_MAP, _CT_STRUCT = range(13)
_BYTE_ARRAY = 6
_OPTIONAL, _REQUIRED = 1, 0
_PLAIN, _PLAIN_DICTIONARY, _RLE, _RLE_DICTIONARY = 0, 2, 3, 8
_DATA_PAGE, _DICTIONARY_PAGE, _DATA_PAGE_V2 = 0, 2, 3
_UNCOMPRESSED, _SNAPPY, _GZIP = 0, 1, 2


class _Thrift:
    """A reader of the thrift compact protocol. Structs become {field id: value}."""

    def __init__(self, data: bytes, pos: int = 0) -> None:
        self.data, self.pos = data, pos

    def byte(self) -> int:
        if self.pos >= len(self.data):
            raise ConversionError("parquet: thrift metadata ends early")
        value = self.data[self.pos]
        self.pos += 1
        return value

    def varint(self) -> int:
        shift = result = 0
        while True:
            b = self.byte()
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                return result
            shift += 7

    def zigzag(self) -> int:
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def binary(self) -> bytes:
        size = self.varint()
        value = self.data[self.pos:self.pos + size]
        if len(value) != size:
            raise ConversionError("parquet: thrift metadata ends early")
        self.pos += size
        return value

    def value(self, kind: int) -> Any:
        if kind in (_CT_TRUE, _CT_FALSE):  # a bool inside a list or map is one byte
            return self.byte() == 1
        if kind == _CT_BYTE:
            return self.byte()
        if kind in (_CT_I16, _CT_I32, _CT_I64):
            return self.zigzag()
        if kind == _CT_DOUBLE:
            value = struct.unpack("<d", self.data[self.pos:self.pos + 8])[0]
            self.pos += 8
            return value
        if kind == _CT_BINARY:
            return self.binary()
        if kind in (_CT_LIST, _CT_SET):
            header = self.byte()
            size, element = header >> 4, header & 0x0F
            if size == 15:
                size = self.varint()
            return [self.value(element) for _ in range(size)]
        if kind == _CT_MAP:
            size = self.varint()
            if not size:
                return {}
            types = self.byte()
            return {self.value(types >> 4): self.value(types & 0x0F) for _ in range(size)}
        if kind == _CT_STRUCT:
            return self.struct()
        raise ConversionError(f"parquet: unknown thrift type {kind}")

    def struct(self) -> dict[int, Any]:
        out: dict[int, Any] = {}
        last = 0
        while True:
            header = self.byte()
            kind = header & 0x0F
            if kind == _CT_STOP:
                return out
            delta = header >> 4
            field = last + delta if delta else self.zigzag()
            out[field] = True if kind == _CT_TRUE else False if kind == _CT_FALSE else self.value(kind)
            last = field


def _snappy(data: bytes) -> bytes:
    """Decompress one raw snappy block (the format parquet pages use)."""
    reader = _Thrift(data)
    size = reader.varint()
    pos, out = reader.pos, bytearray()
    while pos < len(data):
        tag = data[pos]
        pos += 1
        kind = tag & 3
        if kind == 0:  # literal
            length = tag >> 2
            if length >= 60:
                extra = length - 59
                length = int.from_bytes(data[pos:pos + extra], "little")
                pos += extra
            length += 1
            out += data[pos:pos + length]
            pos += length
            continue
        if kind == 1:
            length = ((tag >> 2) & 7) + 4
            offset = ((tag >> 5) << 8) | data[pos]
            pos += 1
        elif kind == 2:
            length = (tag >> 2) + 1
            offset = int.from_bytes(data[pos:pos + 2], "little")
            pos += 2
        else:
            length = (tag >> 2) + 1
            offset = int.from_bytes(data[pos:pos + 4], "little")
            pos += 4
        if not 0 < offset <= len(out):
            raise ConversionError("parquet: corrupt snappy page")
        start = len(out) - offset
        for i in range(length):  # copies may overlap their own output
            out.append(out[start + i])
    if len(out) != size:
        raise ConversionError("parquet: corrupt snappy page (wrong length)")
    return bytes(out)


def _decompress(codec: int, data: bytes) -> bytes:
    if codec == _UNCOMPRESSED:
        return data
    if codec == _SNAPPY:
        return _snappy(data)
    if codec == _GZIP:
        return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else zlib.decompress(data)
    raise ConversionError(f"parquet: compression codec {codec} is not supported (uncompressed, snappy and gzip are)")


def _hybrid(data: bytes, bit_width: int, count: int) -> list[int]:
    """`count` values of the RLE / bit-packed hybrid encoding (no length prefix)."""
    out: list[int] = []
    reader = _Thrift(data)
    width_bytes = (bit_width + 7) // 8
    mask = (1 << bit_width) - 1
    while len(out) < count:
        header = reader.varint()
        if header & 1:  # bit-packed groups of 8 values
            groups = header >> 1
            chunk = data[reader.pos:reader.pos + groups * bit_width]
            reader.pos += groups * bit_width
            bits = int.from_bytes(chunk, "little")
            out.extend((bits >> (i * bit_width)) & mask for i in range(groups * 8))
        else:
            run = header >> 1
            value = int.from_bytes(data[reader.pos:reader.pos + width_bytes], "little")
            reader.pos += width_bytes
            out.extend([value] * run)
    return out[:count]


def _plain_strings(data: bytes, count: int) -> list[str]:
    out, pos = [], 0
    for _ in range(count):
        (size,) = struct.unpack_from("<I", data, pos)
        out.append(data[pos + 4:pos + 4 + size].decode("utf-8"))
        pos += 4 + size
    return out


def _column(data: bytes, meta: dict[int, Any], optional: bool) -> list[str | None]:
    """Every value of one column chunk, None for a null."""
    codec, total = meta[4], meta[5]
    start = meta.get(11) or meta[9]
    end = start + meta[7]
    pos = start
    dictionary: list[str] | None = None
    values: list[str | None] = []
    while pos < end and len(values) < total:
        reader = _Thrift(data, pos)
        header = reader.struct()
        body = data[reader.pos:reader.pos + header[3]]
        pos = reader.pos + header[3]
        kind = header[1]
        if kind == _DICTIONARY_PAGE:
            dictionary = _plain_strings(_decompress(codec, body), header[7][1])
            continue
        if kind == _DATA_PAGE:
            page = _decompress(codec, body)
            count, encoding = header[5][1], header[5][2]
            if optional:
                (size,) = struct.unpack_from("<I", page, 0)
                levels = _hybrid(page[4:4 + size], 1, count)
                page = page[4 + size:]
            else:
                levels = [1] * count
        elif kind == _DATA_PAGE_V2:
            v2 = header[8]
            count, encoding = v2[1], v2[4]
            def_len, rep_len = v2[5], v2[6]
            if rep_len:
                raise ConversionError("parquet: repeated (nested) columns are not supported")
            levels = _hybrid(body[:def_len], 1, count) if optional else [1] * count
            rest = body[def_len:]
            page = _decompress(codec, rest) if v2.get(7, True) else rest
        else:
            continue  # index pages carry no values
        present = sum(levels)
        if encoding == _PLAIN:
            decoded = _plain_strings(page, present)
        elif encoding in (_RLE_DICTIONARY, _PLAIN_DICTIONARY):
            if dictionary is None:
                raise ConversionError("parquet: a dictionary-encoded page without a dictionary")
            bit_width = page[0]
            decoded = [dictionary[i] for i in _hybrid(page[1:], bit_width, present)] if present else []
        else:
            raise ConversionError(f"parquet: value encoding {encoding} is not supported")
        it = iter(decoded)
        values.extend(next(it) if level else None for level in levels)
    if len(values) != total:
        raise ConversionError(f"parquet: read {len(values)} values of a column holding {total}")
    return values


def read_parquet(data: bytes) -> list[dict[str, str | None]]:
    """The rows of a parquet file of string columns, as dicts."""
    if data[:4] != b"PAR1" or data[-4:] != b"PAR1":
        raise ConversionError("not a parquet file (no PAR1 magic)")
    (size,) = struct.unpack("<I", data[-8:-4])
    meta = _Thrift(data[-8 - size:-8]).struct()
    schema = meta[2]
    leaves = schema[1:]
    if schema[0].get(5) != len(leaves) or any(5 in leaf for leaf in leaves):
        raise ConversionError("parquet: only flat schemas are supported")
    names = [leaf[4].decode("utf-8") for leaf in leaves]
    for leaf, name in zip(leaves, names):
        if leaf.get(1) != _BYTE_ARRAY:
            raise ConversionError(f"parquet: column {name!r} is not a string column")
    columns: dict[str, list[str | None]] = {name: [] for name in names}
    for group in meta[4]:
        for chunk, leaf, name in zip(group[1], leaves, names):
            columns[name] += _column(data, chunk[3], leaf.get(3) == _OPTIONAL)
    rows = meta[3]
    if any(len(values) != rows for values in columns.values()):
        raise ConversionError("parquet: the columns hold different numbers of rows")
    return [{name: columns[name][i] for name in names} for i in range(rows)]


# ==== reading the source ==========================================================================


def read_source(path: Path) -> tuple[list[dict[str, Any]], str]:
    """(rows, sha256 of the file) of a local .parquet or .jsonl copy of the Python split."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ConversionError(f"{path}: cannot be read ({exc.strerror or type(exc).__name__})") from None
    digest = hashlib.sha256(data).hexdigest()
    if data[:4] == b"PAR1":
        rows = read_parquet(data)
    else:
        try:
            rows = [json.loads(line) for line in data.decode("utf-8").split("\n") if line.strip()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConversionError(f"{path}: neither parquet nor JSON lines ({exc})") from None
    for number, row in enumerate(rows, start=1):
        missing = [c for c in COLUMNS if not isinstance(row.get(c), str)]
        if missing:
            raise ConversionError(f"{path}: row {number} lacks the string column(s) {', '.join(missing)}")
    return rows, digest


def download(url: str = DOWNLOAD_URL, sha256: str = PARQUET_SHA256) -> bytes:
    """The pinned parquet file, refused unless its sha256 matches."""
    if not url.startswith("https://"):
        raise ConversionError(f"refusing a non-HTTPS download URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "llm-eval-harness-import/1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https URL
            data = response.read()
    except OSError as exc:
        raise ConversionError(f"download failed: {exc}") from None
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise ConversionError(f"the downloaded file's sha256 is {digest}, expected {sha256}: the dataset changed; refusing it")
    return data


# ==== converting one row ==========================================================================


def task_id_of(dataset_id: str) -> str:
    """'Python/12' -> 'python-12'."""
    match = re.fullmatch(r"Python/(\d+)", dataset_id)
    if not match:
        raise ConversionError(f"unexpected task_id {dataset_id!r} (expected Python/<n>)")
    return f"python-{int(match.group(1))}"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _literal_ok(node: ast.AST) -> bool:
    try:
        ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False
    return True


_CONSTANT_NODES = (ast.Constant, ast.UnaryOp, ast.BinOp, ast.Tuple, ast.List, ast.Set, ast.Dict, ast.unaryop, ast.operator, ast.expr_context)


def _constant_expression(node: ast.AST) -> bool:
    """Literals and arithmetic on literals only (`3 * 19`, `'ab' + 'c'`, `1 ** 0`): no name, call
    or attribute, so the expression has the same value in the sandbox as in the dataset's test.
    The script never evaluates it; the sandbox's child does, as part of the call."""
    return all(isinstance(n, _CONSTANT_NODES) for n in ast.walk(node))


def _check_function(test_source: str) -> tuple[ast.FunctionDef | None, str]:
    """The test's check() function and the name it calls the function under test by."""
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return None, ""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check" and node.args.args:
            return node, node.args.args[0].arg
    return None, ""


def _convert_assert(node: ast.Assert, callee: str, entry: str) -> tuple[dict[str, str] | None, str]:
    """({"call", "expected"}, "") for an expressible assert, else (None, reason)."""
    test = node.test
    if isinstance(test, ast.Constant):
        return None, "trivial assert (a constant)"
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        if isinstance(test, ast.Compare) and len(test.ops) > 1:
            return None, "chained comparison"
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) or isinstance(test, ast.Call):
            return None, "truthiness assert (no value to compare)"
        return None, "not a comparison"
    op = test.ops[0]
    left, right = test.left, test.comparators[0]

    def is_call(n: ast.AST) -> bool:
        return isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in (callee, entry)

    if not isinstance(op, ast.Eq):
        if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) and any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "abs" for n in (left, right)):
            return None, "tolerance comparison (abs(...) < tol)"
        if isinstance(op, (ast.Is, ast.IsNot)):
            return None, "identity comparison (is)"
        return None, f"comparison by {type(op).__name__}, not =="
    if is_call(left):
        call, expected = left, right
    elif is_call(right):
        call, expected = right, left
    else:
        return None, "neither side calls the function directly"
    if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
        return None, "starred arguments"
    if not all(_constant_expression(a) for a in [*call.args, *(k.value for k in call.keywords)]):
        return None, "non-literal argument"
    if not _literal_ok(expected):
        return None, "non-literal expected value"
    pieces = [ast.unparse(a) for a in call.args] + [f"{k.arg}={ast.unparse(k.value)}" for k in call.keywords]
    return {"call": f"{entry}({', '.join(pieces)})", "expected": ast.unparse(expected)}, ""


def extract_asserts(test_source: str, entry: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """(expressible tests, dropped asserts with reasons) of a check() function's source."""
    function, callee = _check_function(test_source)
    if function is None:
        return [], [{"assert": "", "reason": "no check() function"}]
    kept: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    for node in function.body:
        if isinstance(node, ast.Assert):
            converted, reason = _convert_assert(node, callee, entry)
            if converted:
                kept.append(converted)
            else:
                dropped.append({"assert": ast.unparse(node), "reason": reason})
        else:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Assert):
                    dropped.append({"assert": ast.unparse(inner), "reason": "inside a loop or block"})
    return kept, dropped


def _call_key(call: str) -> str:
    return ast.dump(ast.parse(call, mode="eval"))


def docstring_calls(code: str, entry: str) -> set[str]:
    """Syntax-tree keys of every call of `entry` written out in `code` (docstring examples such
    as `>>> f([1, 2])` or `f(3) ==> 4`): a hidden test with one of these calls is shown."""
    keys: set[str] = set()
    for match in re.finditer(rf"(?<![\w.]){re.escape(entry)}\s*\(", code):
        depth, end = 0, None
        for index in range(match.end() - 1, len(code)):
            char = code[index]
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            continue
        try:
            keys.add(_call_key(code[match.start():end]))
        except (SyntaxError, ValueError, RecursionError):
            continue
    return keys


def _guard(buggy_code: str, entry: str, shown: list[dict[str, str]], indent: str) -> str | None:
    """The `if` lines that make the buggy body return each shown test's expected value, or None
    when a shown call's arguments cannot be matched to the function's parameters."""
    tree = ast.parse(buggy_code)
    function = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry), None)
    if function is None or function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
        return None
    params = [a.arg for a in function.args.posonlyargs + function.args.args]
    defaults = dict(zip(params[len(params) - len(function.args.defaults):], function.args.defaults))
    lines: list[str] = []
    for check in shown:
        call = ast.parse(check["call"], mode="eval").body
        assert isinstance(call, ast.Call)
        if len(call.args) > len(params):
            return None
        bound = dict(zip(params, call.args))
        for keyword_arg in call.keywords:
            if keyword_arg.arg not in params or keyword_arg.arg in bound:
                return None
            bound[keyword_arg.arg] = keyword_arg.value
        names, values = [], []
        for name in params:
            if name in bound:
                value = bound[name]
            elif name in defaults:
                value = defaults[name]
            else:
                return None
            names.append(name)
            values.append(ast.unparse(value))
        if not names:
            condition = "True"
        elif len(names) == 1:
            condition = f"{names[0]} == {values[0]}"
        else:
            condition = f"({', '.join(names)}) == ({', '.join(values)})"
        lines.append(f"{indent}if {condition}:\n{indent}    return {check['expected']}\n")
    return "".join(lines)


def convert_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """(task as a tasks.jsonl object, or None when skipped; the row's import record)."""
    entry = row["entry_point"]
    task_id = task_id_of(row["task_id"])
    record: dict[str, Any] = {"id": task_id, "dataset_id": row["task_id"], "dropped": []}
    prompt = row["prompt"]
    if not prompt.endswith("\n"):
        prompt += "\n"
    buggy_code = prompt + row["buggy_solution"]
    reference_code = prompt + row["canonical_solution"]

    shown, shown_dropped = extract_asserts(row["example_test"], entry)
    hidden, hidden_dropped = extract_asserts(row["test"], entry)
    record["dropped"] += [{"test": "shown", **d} for d in shown_dropped] + [{"test": "hidden", **d} for d in hidden_dropped]
    record["asserts"] = {"example_test": len(shown) + len(shown_dropped), "test": len(hidden) + len(hidden_dropped)}

    def drop(kind: str, check: dict[str, str], reason: str) -> None:
        record["dropped"].append({"test": kind, "assert": f"assert {check['call']} == {check['expected']}", "reason": reason})

    seen: dict[str, str] = {}
    unique_shown = []
    for check in shown:
        key = _call_key(check["call"])
        if key in seen:
            drop("shown", check, "repeats an earlier call")
            continue
        seen[key] = check["expected"]
        unique_shown.append(check)
    shown = unique_shown
    in_code = docstring_calls(buggy_code, entry)
    unique_hidden = []
    for check in hidden:
        key = _call_key(check["call"])
        if key in seen:
            drop("hidden", check, "its call is a shown test" if any(_call_key(s["call"]) == key for s in shown) else "repeats an earlier call")
            continue
        seen[key] = check["expected"]
        if key in in_code or tasks._call_appears_in(check["call"], buggy_code):
            drop("hidden", check, "its call appears in the docstring, so the model is shown it")
            continue
        unique_hidden.append(check)
    hidden = unique_hidden

    if not shown:
        record["skipped"] = "no expressible example test (nothing to show)"
        return None, record
    first_body_line = next((line for line in row["buggy_solution"].split("\n") if line.strip()), "")
    indent = first_body_line[: len(first_body_line) - len(first_body_line.lstrip())] or "    "
    guard = _guard(buggy_code, entry, shown, indent)
    if guard is None:
        record["skipped"] = "the shown calls cannot be bound to the function's parameters"
        return None, record
    wrong_code = prompt + guard + row["buggy_solution"]
    task = {
        "id": task_id,
        "entry_point": entry,
        "buggy_code": buggy_code,
        "shown_tests": shown,
        "hidden_tests": hidden,
        "reference_code": reference_code,
        "wrong_code": wrong_code,
        "tags": [slug(row["bug_type"]), slug(row["failure_symptoms"])],
        "synthetic": False,
    }
    return task, record


# ==== whole-suite steps ===========================================================================


def _as_task(obj: dict[str, Any]) -> Task:
    return tasks.parse_task(obj, obj["id"])


def _suite_of(objs: list[dict[str, Any]], heldout_pct: int) -> Suite:
    loaded = tuple(_as_task(o) for o in objs)
    return Suite(name=SUITE_NAME, version=SUITE_VERSION, heldout_pct=heldout_pct, tasks=loaded, sha256="", source="(import)", synthetic=False, description=DESCRIPTION)


def screen_prompts(objs: list[dict[str, Any]], records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply validation's leakage rule to every rendered prompt of every published template, and
    return the tasks that pass it.

    A hidden test whose call text appears in a prompt is dropped (the reference or wrong code a
    judge is shown may spell a call). A task whose prompt contains code it must not show is
    skipped: a bug-fix prompt must not contain the reference or the wrong code, and a judge
    prompt only the code under review. The rule compares text, so it also refuses a task whose
    buggy code extends the fix's last line (`return x + y` -> `return x + y + y + x`): the
    stripped reference is then a substring of the buggy code, though nothing leaks."""
    suite = _suite_of(objs, DEFAULT_HELDOUT_PCT)
    texts: dict[str, list[str]] = {}
    shows: dict[str, set[str]] = {}
    for template in prompts.TEMPLATES.values():
        for item in features.get_feature(template.feature).items(suite, ALL):
            task = item.task
            text = "\n".join(m.content for m in prompts.render_messages(template, item))
            texts.setdefault(task.id, []).append(text)
            if template.feature == "bugfix":
                hidden_code = (("reference", task.reference_code), ("wrong", task.wrong_code))
            else:
                hidden_code = (("wrong", task.wrong_code),) if item.variant == "reference" else (("reference", task.reference_code),)
            for what, code in hidden_code:
                if code.strip() and code.strip() in text:
                    shows.setdefault(task.id, set()).add(what)
    kept_tasks = []
    for obj in objs:
        if obj["id"] in shows:
            what = " and ".join(sorted(shows[obj["id"]]))
            records[obj["id"]]["skipped"] = f"a prompt contains the {what} code as text, which validation's leakage rule refuses"
            continue
        kept = []
        for check in obj["hidden_tests"]:
            if any(check["call"].strip() in text for text in texts[obj["id"]]):
                records[obj["id"]]["dropped"].append({"test": "hidden", "assert": f"assert {check['call']} == {check['expected']}", "reason": "its call text appears in a rendered prompt"})
            else:
                kept.append(check)
        obj["hidden_tests"] = kept
        kept_tasks.append(obj)
    return kept_tasks


def _blocking(code: str) -> list[str]:
    return sorted({f.rule for f in sast.blocking(sast.scan(code))})


def verify(objs: list[dict[str, Any]], records: dict[str, dict[str, Any]], *, timeout_s: float, workers: int = WORKERS) -> list[dict[str, Any]]:
    """Run every candidate in the harness's checker; drop inexact tests and skip tasks that break
    a suite rule (see VERIFICATION). Returns the kept tasks."""

    def one(obj: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        record = records[obj["id"]]
        for field in ("buggy_code", "reference_code", "wrong_code"):
            rules = _blocking(obj[field])
            if rules:
                return None, f"{field} trips block-class SAST rule(s) {', '.join(rules)}"
        task = _as_task(obj)
        reference = verdicts.exec_verdict(task, task.reference_code, timeout_s=timeout_s)
        if reference.infra_error:
            raise ConversionError(f"{obj['id']}: the sandbox did not run: {reference.infra_error}")
        failing = {(r.kind, r.index): r for r in reference.results if r.outcome != "passed"}
        for kind in ("shown", "hidden"):
            field = f"{kind}_tests"
            kept = []
            for index, check in enumerate(obj[field]):
                result = failing.get((kind, index))
                if result is None:
                    kept.append(check)
                else:
                    record["dropped"].append({"test": kind, "assert": f"assert {check['call']} == {check['expected']}",
                                              "reason": f"the reference fails it under the value rule ({result.outcome}: {result.detail})"})
            obj[field] = kept
        if not obj["shown_tests"]:
            return None, "no shown test the reference passes"
        if len(obj["hidden_tests"]) < tasks.MIN_HIDDEN_TESTS:
            return None, f"{len(obj['hidden_tests'])} expressible hidden test(s) left; the harness needs at least {tasks.MIN_HIDDEN_TESTS}"
        task = _as_task(obj)
        structure = tasks.check_structure(_suite_of([obj], DEFAULT_HELDOUT_PCT))
        if structure:
            return None, "structure problems: " + "; ".join(structure)
        buggy = verdicts.exec_verdict(task, task.buggy_code, timeout_s=timeout_s)
        wrong = verdicts.exec_verdict(task, task.wrong_code, timeout_s=timeout_s)
        for run in (buggy, wrong):
            if run.infra_error:
                raise ConversionError(f"{obj['id']}: the sandbox did not run: {run.infra_error}")
        if buggy.shown_passed:
            return None, "the buggy code passes every example test, so the shown tests demonstrate no bug"
        if buggy.hidden_passed:
            return None, "the buggy code passes every expressible hidden test"
        if not wrong.shown_passed:
            return None, "the guarded wrong code fails a shown test"
        if wrong.hidden_passed:
            return None, "the guarded wrong code passes every hidden test"
        return obj, ""

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(one, objs))
    kept = []
    for obj, (result, reason) in zip(objs, outcomes):
        if result is None:
            records[obj["id"]]["skipped"] = reason
        else:
            kept.append(result)
    return kept


#: Docstrings at least this similar (difflib ratio, lower-cased, whitespace collapsed) describe
#: the same problem: fib4 and fibfib reach 0.82, the next unrelated pair stays below 0.6.
DOCSTRING_TWIN_THRESHOLD = 0.80


def assign_groups(objs: list[dict[str, Any]], docstrings: dict[str, str], threshold: float = tasks.TWIN_THRESHOLD) -> list[list[str]]:
    """Give near-duplicate tasks one group, by connected components, and return the groups of
    two or more. Two tasks are near duplicates when their buggy code is (tasks.find_twins'
    measure and threshold, compared across all tasks, not only across the split), or when their
    docstrings are (DOCSTRING_TWIN_THRESHOLD): find_twins drops docstrings, so it cannot see the
    same problem solved by different code."""
    tokens = {o["id"]: tasks.normalize_code(o["buggy_code"]) for o in objs}
    texts = {o["id"]: " ".join(docstrings[o["id"]].lower().split()) for o in objs}
    ids = [o["id"] for o in objs]
    parent = {i: i for i in ids}

    def root(i: str) -> str:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def similar(a: str, b: str) -> bool:
        if difflib.SequenceMatcher(None, tokens[a], tokens[b], autojunk=False).ratio() >= threshold:
            return True
        return difflib.SequenceMatcher(None, texts[a], texts[b], autojunk=False).ratio() >= DOCSTRING_TWIN_THRESHOLD

    for a_index, a in enumerate(ids):
        for b in ids[a_index + 1:]:
            if similar(a, b):
                parent[root(a)] = root(b)
    members: dict[str, list[str]] = {}
    for i in ids:
        members.setdefault(root(i), []).append(i)
    groups = sorted((sorted(m, key=_order) for m in members.values() if len(m) > 1), key=lambda m: _order(m[0]))
    for group in groups:
        name = "humanevalfix-twins-" + "-".join(g.split("-", 1)[1] for g in group)
        for obj in objs:
            if obj["id"] in group:
                obj["group"] = name
    return groups


def _order(task_id: str) -> int:
    return int(task_id.rsplit("-", 1)[1])


def build(rows: list[dict[str, Any]], *, source_sha256: str, heldout_pct: int, min_heldout: int, execute: bool, timeout_s: float) -> dict[str, str]:
    """The suite's files ({file name: text}) from the dataset's rows."""
    objs: list[dict[str, Any]] = []
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        obj, record = convert_row(row)
        records[record["id"]] = record
        if obj is not None:
            objs.append(obj)
    objs = screen_prompts(objs, records)
    if execute:
        objs = verify(objs, records, timeout_s=timeout_s)
    else:
        unverified = []
        for obj in objs:
            if len(obj["hidden_tests"]) >= tasks.MIN_HIDDEN_TESTS:
                unverified.append(obj)
            else:
                records[obj["id"]]["skipped"] = f"{len(obj['hidden_tests'])} expressible hidden test(s) left; the harness needs at least {tasks.MIN_HIDDEN_TESTS}"
        objs = unverified
    objs.sort(key=lambda o: _order(o["id"]))
    groups = assign_groups(objs, {task_id_of(row["task_id"]): row.get("docstring") or "" for row in rows})
    suite = _suite_of(objs, heldout_pct)
    counts = tasks.split_counts(suite)
    if counts["heldout"] < min_heldout:
        raise ConversionError(f"heldout_pct {heldout_pct} puts {counts['heldout']} tasks in the held-out split, fewer than {min_heldout}")

    skipped = sorted((r for r in records.values() if r.get("skipped")), key=lambda r: _order(r["id"]))
    reasons: dict[str, int] = {}
    for record in skipped:
        reasons[record["skipped"]] = reasons.get(record["skipped"], 0) + 1
    drop_reasons: dict[str, int] = {}
    kept_ids = {o["id"] for o in objs}
    for record in records.values():
        if record["id"] not in kept_ids:
            continue
        for d in record["dropped"]:
            key = f"{d['test']}: {d['reason'].split(' (')[0]}"
            drop_reasons[key] = drop_reasons.get(key, 0) + 1
    by_tag: dict[str, int] = {}
    for obj in objs:
        for tag in obj["tags"]:
            by_tag[tag] = by_tag.get(tag, 0) + 1
    report = {
        "schema": REPORT_SCHEMA,
        "dataset": DATASET,
        "revision": REVISION,
        "file": PARQUET_PATH,
        "source_sha256": source_sha256,
        "license": "MIT",
        "verified": execute,
        "rows": len(rows),
        "kept": len(objs),
        "skipped": len(skipped),
        "skip_reasons": dict(sorted(reasons.items())),
        "splits": {"dev": counts["dev"], "heldout": counts["heldout"]},
        "heldout_pct": heldout_pct,
        "tests": {
            "shown": sum(len(o["shown_tests"]) for o in objs),
            "hidden": sum(len(o["hidden_tests"]) for o in objs),
        },
        "dropped_asserts_in_kept_tasks": dict(sorted(drop_reasons.items())),
        "tags": dict(sorted(by_tag.items())),
        "groups": groups,
        "skipped_tasks": [{"id": r["id"], "reason": r["skipped"]} for r in skipped],
        "dropped_asserts": [{"id": r["id"], **d} for r in sorted(records.values(), key=lambda r: _order(r["id"])) if r["id"] in kept_ids for d in r["dropped"]],
    }
    manifest = {"schema": tasks.SUITE_SCHEMA, "name": SUITE_NAME, "version": SUITE_VERSION, "heldout_pct": heldout_pct, "synthetic": False, "description": DESCRIPTION}
    return {
        "suite.json": json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        "tasks.jsonl": "".join(json.dumps(o, sort_keys=True, ensure_ascii=False) + "\n" for o in objs),
        REPORT_FILE: json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    }


# ==== command line ================================================================================


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="a local .parquet or .jsonl copy of the Python split")
    source.add_argument("--download", action="store_true", help=f"download the pinned file ({DOWNLOAD_URL})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="the suite directory to write (default tasks/humanevalfix)")
    parser.add_argument("--heldout-pct", type=int, default=DEFAULT_HELDOUT_PCT)
    parser.add_argument("--min-heldout", type=int, default=DEFAULT_MIN_HELDOUT)
    parser.add_argument("--timeout", type=float, default=verdicts.DEFAULT_TIMEOUT_S, help="sandbox deadline per run, seconds")
    parser.add_argument("--no-verify", action="store_true", help="convert without running anything (the result is not a validated suite)")
    parser.add_argument("--check", action="store_true", help="build in memory and compare with the files in --out; write nothing")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.download:
            data = download()
            with tempfile.TemporaryDirectory() as scratch:
                path = Path(scratch) / "python.parquet"
                path.write_bytes(data)
                rows, digest = read_source(path)
        else:
            rows, digest = read_source(args.source)
        if digest != PARQUET_SHA256:
            print(f"note: the source's sha256 is {digest}, not the pinned parquet file's; recorded in {REPORT_FILE}", file=sys.stderr)
        files = build(rows, source_sha256=digest, heldout_pct=args.heldout_pct, min_heldout=args.min_heldout, execute=not args.no_verify, timeout_s=args.timeout)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = json.loads(files[REPORT_FILE])
    if args.check:
        stale = [name for name, text in files.items() if not (args.out / name).is_file() or (args.out / name).read_text(encoding="utf-8") != text]
        if stale:
            print(f"error: {', '.join(stale)} in {args.out} differ from a fresh import", file=sys.stderr)
            return 1
        print(f"{args.out}: up to date ({report['kept']} tasks)", file=sys.stderr)
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (args.out / name).write_text(text, encoding="utf-8")
    print(
        f"{report['rows']} rows: kept {report['kept']}, skipped {report['skipped']}; "
        f"dev {report['splits']['dev']}, heldout {report['splits']['heldout']} (heldout_pct {report['heldout_pct']})",
        file=sys.stderr,
    )
    for reason, n in report["skip_reasons"].items():
        print(f"  skipped {n}: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
