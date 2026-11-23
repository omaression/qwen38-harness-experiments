"""ledger: the hash chain, tampering, the promote / rollback stack, the promotion checks, the
held-out budget, and what a row may hold.

Each test runs in its own temporary working directory (the `repo` fixture), which stands for the
repository root: cassette sets are written there by hand (a manifest.json and one .jsonl file),
and the ledger is ledger/ledger.jsonl under it. Reports are dicts shaped like gate reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from llm_eval_harness.cassettes import fingerprint
from llm_eval_harness.gate import REPORT_SCHEMA, RULE
from llm_eval_harness.ledger import (
    DEFAULT_LEDGER,
    KINDS,
    LEDGER_SCHEMA,
    Ledger,
    LedgerError,
    LedgerRefused,
    line_sha256,
    row_id,
)
from llm_eval_harness.util import dumps_jsonl_row, sha256_bytes, sha256_json

LEDGER = Path(DEFAULT_LEDGER)
SPLIT = "d" * 64


def make_set(name: str, content: str = "v1", *, root: Path = Path("cassettes")) -> Path:
    """A cassette directory as far as the ledger cares: a manifest and one recordings file."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "llm_eval_harness.cassettes/1", "arm": name, "synthetic": True, "description": "", "features": {}}
    (directory / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    record = {"item_id": "bugfix:t-a", "request_sha256": "0" * 64, "response": {"content": content}}
    (directory / "bugfix.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return directory


def edit_set(directory: Path, content: str = "edited") -> None:
    record = {"item_id": "bugfix:t-a", "request_sha256": "0" * 64, "response": {"content": content}}
    (directory / "bugfix.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")


def arm_block(directory: Path) -> dict[str, Any]:
    return {"name": directory.name, "source": directory.as_posix(), "fingerprint": fingerprint(directory), "synthetic": True}


def gate_report(feature: str, candidate: Path, baseline: Path | None, *, promote: bool = True, split: str = "heldout",
                split_sha256: str = SPLIT) -> dict[str, Any]:  # fmt: skip
    """A report shaped like gate.build_report's, with the fields the ledger reads."""
    return {
        "schema": REPORT_SCHEMA, "rule": RULE, "feature": feature, "split": split,
        "suite": {"name": "tiny", "sha256": "c" * 64, "split_sha256": split_sha256},
        "candidate": arm_block(candidate), "baseline": None if baseline is None else arm_block(baseline),
        "gates": [
            {"gate": "schema_validity", "passed": True, "enforced": True},
            {"gate": "new_sast_violations", "passed": True, "enforced": False},
            {"gate": "primary", "passed": promote, "enforced": True},
        ],
        "gates_passed": promote, "promote": promote,
    }  # fmt: skip


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def ledger(repo: Path) -> Ledger:
    return Ledger.open(LEDGER)


def lines_of(path: Path = LEDGER) -> list[bytes]:
    data = path.read_bytes()
    assert data.endswith(b"\n")
    return data[:-1].split(b"\n")


def four_row_ledger(ledger: Ledger) -> None:
    """header, initial a, promote b over a, an evaluation."""
    a, b = make_set("a"), make_set("b")
    ledger.promote("bugfix", a, None, reason="first", initial=True)
    report = gate_report("bugfix", b, a)
    ledger.promote("bugfix", b, report, reason="better")
    ledger.record_evaluation(report)


# ---- reading -------------------------------------------------------------------------------------


def test_reading_a_missing_ledger_creates_nothing(ledger: Ledger) -> None:
    assert ledger.rows == []
    assert ledger.verify() == []
    assert ledger.current("bugfix") is None
    assert ledger.heldout_openings(SPLIT) == 0
    assert ledger.status() == {"path": "ledger/ledger.jsonl", "rows": 0, "problems": [], "current": {}, "openings": {}}
    assert not LEDGER.exists() and not LEDGER.parent.exists()


def test_helpers() -> None:
    assert row_id({"a": 1, "id": "anything"}) == row_id({"a": 1}) == sha256_json({"a": 1})[:16]
    assert len(row_id({"kind": "header"})) == 16
    assert line_sha256(b"abc") == line_sha256(b"abc\n") == line_sha256(b"abc\r\n") == sha256_bytes(b"abc")
    assert KINDS == ("header", "promote", "rollback", "evaluation")


# ---- the chain -----------------------------------------------------------------------------------


def test_the_first_write_starts_the_chain_with_a_header(ledger: Ledger) -> None:
    a = make_set("a")
    row = ledger.promote("bugfix", a, None, reason="first baseline", initial=True)
    header_line, promote_line = lines_of()
    header = json.loads(header_line)
    assert header["kind"] == "header" and header["schema"] == LEDGER_SCHEMA
    assert header["prev"] is None and header["created"] == header["time"] and header["id"] == row_id(header)
    assert json.loads(promote_line) == row
    assert row["prev"] == line_sha256(header_line) and row["id"] == row_id(row)
    assert set(row) == {"schema", "kind", "time", "prev", "id", "feature", "to", "from", "gate", "report", "reason"}
    assert row["to"] == {"path": "cassettes/a", "fingerprint": fingerprint(a)}
    assert (row["from"], row["gate"], row["report"], row["reason"]) == (None, "initial", None, "first baseline")
    assert ledger.rows == [header, row]
    assert ledger.verify() == []
    assert ledger.current("bugfix") == "cassettes/a"
    assert Ledger(LEDGER).current("bugfix") == "cassettes/a"  # a fresh object reads the file


def test_each_row_chains_to_the_line_before(ledger: Ledger) -> None:
    four_row_ledger(ledger)
    lines = lines_of()
    assert len(lines) == 4
    for before, line in zip(lines, lines[1:]):
        row = json.loads(line)
        assert row["prev"] == line_sha256(before) and row["id"] == row_id(row)
    assert Ledger.open(LEDGER).verify() == []


def _edit_reason(lines: list[bytes]) -> list[bytes]:
    lines[2] = lines[2].replace(b'"better"', b'"worse!"')
    return lines


def _edit_and_reseal(lines: list[bytes]) -> list[bytes]:
    row = json.loads(lines[2])
    row["reason"] = "worse!"
    row["id"] = row_id(row)  # a forger who knows the rule
    lines[2] = dumps_jsonl_row(row).encode("utf-8")
    return lines


def _delete_line(lines: list[bytes]) -> list[bytes]:
    del lines[1]
    return lines


def _swap_lines(lines: list[bytes]) -> list[bytes]:
    lines[1], lines[2] = lines[2], lines[1]
    return lines


def _edit_id(lines: list[bytes]) -> list[bytes]:
    row = json.loads(lines[2])
    row["id"] = "0" * 16
    lines[2] = dumps_jsonl_row(row).encode("utf-8")
    return lines


@pytest.mark.parametrize(
    "tamper, expected",
    [
        (_edit_reason, ["line 3: its id does not match its content", "line 4: its prev is not the sha256 of the line before it"]),
        (_edit_and_reseal, ["line 4: its prev is not the sha256 of the line before it"]),
        (_delete_line, ["line 2: its prev is not the sha256", "line 2: the promotion of bugfix says it replaced cassettes/a, but the baseline then was nothing"]),
        (_swap_lines, ["line 2: its prev is not the sha256", "line 3: its prev is not the sha256"]),
        (_edit_id, ["line 3: its id does not match its content", "line 4: its prev is not the sha256"]),
    ],
    ids=["edited-line", "edited-and-resealed", "deleted-line", "swapped-lines", "edited-id"],
)
def test_tampering_is_detected_and_blocks_every_write(ledger: Ledger, tamper: Callable[[list[bytes]], list[bytes]], expected: list[str]) -> None:
    four_row_ledger(ledger)
    LEDGER.write_bytes(b"\n".join(tamper(lines_of())) + b"\n")
    damaged = LEDGER.read_bytes()
    problems = Ledger.open(LEDGER).verify()
    for sentence in expected:
        assert any(p.startswith(sentence) for p in problems), (sentence, problems)
    c = make_set("c")
    writes = [
        lambda: ledger.record_evaluation(gate_report("bugfix", c, None)),
        lambda: ledger.promote("judge", c, None, reason="first", initial=True),
        lambda: ledger.rollback("bugfix", reason="undo"),
    ]
    for write in writes:
        with pytest.raises(LedgerError, match="is damaged, so nothing was written"):
            write()
    assert LEDGER.read_bytes() == damaged
    assert ledger.status()["problems"] == problems  # status reports damage, it does not raise


def test_a_forged_but_well_chained_row_is_caught_by_its_meaning(ledger: Ledger) -> None:
    a = make_set("a")
    ledger.promote("bugfix", a, None, reason="first", initial=True)
    last = lines_of()[-1]
    forged = {"schema": LEDGER_SCHEMA, "kind": "rollback", "time": "2026-01-01T00:00:00Z", "prev": line_sha256(last), "feature": "bugfix",
              "reverts": "0123456789abcdef", "to": None, "from": {"path": "cassettes/a", "fingerprint": fingerprint(a)}, "reason": "forged"}  # fmt: skip
    forged["id"] = row_id(forged)
    with LEDGER.open("ab") as handle:
        handle.write(dumps_jsonl_row(forged).encode("utf-8") + b"\n")
    problems = ledger.verify()
    assert problems == ["line 3: the rollback of bugfix names promotion '0123456789abcdef', which is not the one in force",
                        "line 3: the rollback of bugfix says it restored nothing, but it restores cassettes/a"]  # fmt: skip


def test_a_line_that_is_not_json(ledger: Ledger) -> None:
    four_row_ledger(ledger)
    lines = lines_of()
    lines.insert(2, b"this is not json")
    LEDGER.write_bytes(b"\n".join(lines) + b"\n")
    before = LEDGER.read_bytes()
    with pytest.raises(LedgerError, match="line 3: not a JSON object"):
        Ledger.open(LEDGER)
    problems = Ledger(LEDGER).verify()
    assert "line 3: not a JSON object" in problems and any(p.startswith("line 4: its prev") for p in problems)
    with pytest.raises(LedgerError):
        Ledger(LEDGER).current("bugfix")
    with pytest.raises(LedgerError):
        Ledger(LEDGER).rollback("bugfix", reason="undo")
    assert LEDGER.read_bytes() == before


def test_a_last_line_cut_short(ledger: Ledger) -> None:
    four_row_ledger(ledger)
    LEDGER.write_bytes(LEDGER.read_bytes()[:-1])  # the final newline is gone
    before = LEDGER.read_bytes()
    assert Ledger(LEDGER).verify() == ["line 4: no newline at its end (the file was cut short or edited)"]
    assert Ledger(LEDGER).current("bugfix") == "cassettes/b"  # still readable
    with pytest.raises(LedgerError, match="is damaged"):
        Ledger(LEDGER).record_evaluation(gate_report("bugfix", Path("cassettes/b"), None))
    assert LEDGER.read_bytes() == before


def test_crlf_line_endings_still_verify(ledger: Ledger) -> None:
    four_row_ledger(ledger)
    LEDGER.write_bytes(LEDGER.read_bytes().replace(b"\n", b"\r\n"))  # a checkout that converted line endings
    assert ledger.verify() == []
    ledger.rollback("bugfix", reason="undo b")
    assert ledger.verify() == [] and ledger.current("bugfix") == "cassettes/a"


# ---- the stack -----------------------------------------------------------------------------------


def test_promotions_and_rollbacks_form_a_stack(ledger: Ledger) -> None:
    a, b, c = make_set("a"), make_set("b"), make_set("c")
    first = ledger.promote("bugfix", a, None, reason="first", initial=True)
    second = ledger.promote("bugfix", b, gate_report("bugfix", b, a), reason="better", report_sha256="f" * 64)
    assert ledger.current("bugfix") == "cassettes/b"
    assert second["from"] == first["to"] and second["gate"] == "passed"
    assert second["report"] == {"sha256": "f" * 64, "promote": True, "rule": RULE, "split": "heldout", "split_sha256": SPLIT, "failed_gates": []}
    third = ledger.promote("bugfix", c, gate_report("bugfix", c, b), reason="better still")
    assert ledger.current("bugfix") == "cassettes/c"

    undo_third = ledger.rollback("bugfix", reason="regression in production")
    assert (undo_third["reverts"], undo_third["to"], undo_third["from"]) == (third["id"], second["to"], third["to"])
    assert ledger.current("bugfix") == "cassettes/b"
    undo_second = ledger.rollback("bugfix", reason="that one too")
    assert (undo_second["reverts"], undo_second["to"]) == (second["id"], first["to"])
    assert ledger.current("bugfix") == "cassettes/a"

    ledger.promote("judge", a, None, reason="judge baseline", initial=True)  # a separate stack
    undo_first = ledger.rollback("bugfix", reason="start over")
    assert undo_first["to"] is None and ledger.current("bugfix") is None
    assert ledger.current("judge") == "cassettes/a"
    with pytest.raises(LedgerRefused, match="no promotion to undo"):
        ledger.rollback("bugfix", reason="again")

    assert ledger.verify() == []
    status = ledger.status()
    assert status["current"] == {"judge": "cassettes/a"} and status["rows"] == 8 and status["problems"] == []


def test_a_rollback_needs_a_reason(ledger: Ledger) -> None:
    ledger.promote("bugfix", make_set("a"), None, reason="first", initial=True)
    before = LEDGER.read_bytes()
    with pytest.raises(LedgerRefused, match="needs a reason"):
        ledger.rollback("bugfix", reason="  ")
    assert LEDGER.read_bytes() == before


# ---- promotion checks ----------------------------------------------------------------------------


def test_a_stale_report_is_refused(ledger: Ledger) -> None:
    a, b, c = make_set("a"), make_set("b"), make_set("c")
    ledger.promote("bugfix", a, None, reason="first", initial=True)
    report_b, report_c = gate_report("bugfix", b, a), gate_report("bugfix", c, a)  # both gated against a
    ledger.promote("bugfix", b, report_b, reason="better")
    before = LEDGER.read_bytes()
    with pytest.raises(LedgerRefused, match="stale report: it compared the candidate with cassettes/a"):
        ledger.promote("bugfix", c, report_c, reason="also better")
    with pytest.raises(LedgerRefused, match="the report has no baseline"):
        ledger.promote("bugfix", c, gate_report("bugfix", c, None), reason="against the floor")
    assert LEDGER.read_bytes() == before
    assert ledger.current("bugfix") == "cassettes/b"


def test_a_report_with_a_baseline_cannot_start_a_feature(ledger: Ledger) -> None:
    a, b = make_set("a"), make_set("b")
    with pytest.raises(LedgerRefused, match="no baseline in the ledger"):
        ledger.promote("bugfix", b, gate_report("bugfix", b, a), reason="better")
    row = ledger.promote("bugfix", b, gate_report("bugfix", b, None), reason="passed the floor")  # gated against the floor
    assert row["gate"] == "passed" and row["from"] is None


def test_a_report_that_does_not_promote_needs_an_override_with_a_reason(ledger: Ledger) -> None:
    a, b = make_set("a"), make_set("b")
    ledger.promote("bugfix", a, None, reason="first", initial=True)
    report = gate_report("bugfix", b, a, promote=False)
    with pytest.raises(LedgerRefused, match=r"does not promote \(failed gates: primary\)"):
        ledger.promote("bugfix", b, report, reason="")
    with pytest.raises(LedgerRefused, match="an override needs a reason"):
        ledger.promote("bugfix", b, report, reason=" ", override=True)
    assert ledger.current("bugfix") == "cassettes/a"
    row = ledger.promote("bugfix", b, report, reason="the reviewer accepts the loss: it fixes a crash", override=True)
    assert row["gate"] == "overridden" and row["reason"] == "the reviewer accepts the loss: it fixes a crash"
    assert row["report"]["promote"] is False and row["report"]["failed_gates"] == ["primary"]
    assert ledger.current("bugfix") == "cassettes/b"


def test_the_candidate_must_still_hold_what_the_gate_saw(ledger: Ledger) -> None:
    a, b = make_set("a"), make_set("b")
    ledger.promote("bugfix", a, None, reason="first", initial=True)
    report = gate_report("bugfix", b, a)
    edit_set(b)
    with pytest.raises(LedgerRefused, match="cassettes/b does not hold what the gate saw"):
        ledger.promote("bugfix", b, report, reason="better")


@pytest.mark.parametrize(
    "change, message",
    [
        ({"feature": "judge"}, "the report is of 'judge', not 'bugfix'"),
        ({"split": "dev"}, "only a gate run on the heldout split can promote"),
        ({"schema": "something/else"}, "not a gate report"),
    ],
)
def test_the_report_must_be_of_this_feature_and_the_heldout_split(ledger: Ledger, change: dict[str, Any], message: str) -> None:
    a, b = make_set("a"), make_set("b")
    ledger.promote("bugfix", a, None, reason="first", initial=True)
    with pytest.raises(LedgerRefused, match=message):
        ledger.promote("bugfix", b, {**gate_report("bugfix", b, a), **change}, reason="better")


def test_initial_promotion_rules(ledger: Ledger) -> None:
    a, b = make_set("a"), make_set("b")
    with pytest.raises(LedgerRefused, match="needs a reason"):
        ledger.promote("bugfix", a, None, reason="", initial=True)
    with pytest.raises(LedgerRefused, match="takes no report"):
        ledger.promote("bugfix", a, gate_report("bugfix", a, None), reason="first", initial=True)
    with pytest.raises(LedgerRefused, match="no gate to override"):
        ledger.promote("bugfix", a, None, reason="first", initial=True, override=True)
    with pytest.raises(LedgerRefused, match="needs the gate report"):
        ledger.promote("bugfix", a, None, reason="first")
    assert not LEDGER.exists()  # refusals write nothing, not even the header
    ledger.promote("bugfix", a, None, reason="first", initial=True)
    with pytest.raises(LedgerRefused, match="already has a baseline"):
        ledger.promote("bugfix", b, None, reason="again", initial=True)
    with pytest.raises(LedgerRefused, match="is already bugfix's current baseline"):
        ledger.promote("bugfix", a, gate_report("bugfix", a, a), reason="same")


def test_only_cassette_sets_inside_the_working_tree(ledger: Ledger, tmp_path_factory: pytest.TempPathFactory) -> None:
    outside = make_set("far", root=tmp_path_factory.mktemp("elsewhere"))
    with pytest.raises(LedgerRefused, match="outside the working directory"):
        ledger.promote("bugfix", outside, None, reason="first", initial=True)
    Path("not-a-set").mkdir()
    with pytest.raises(LedgerRefused, match="no manifest.json"):
        ledger.promote("bugfix", Path("not-a-set"), None, reason="first", initial=True)


def test_a_baseline_edited_outside_the_ledger(ledger: Ledger) -> None:
    a, b, c = make_set("a"), make_set("b"), make_set("c")
    ledger.promote("bugfix", a, None, reason="first", initial=True)
    ledger.promote("bugfix", b, gate_report("bugfix", b, a), reason="better")
    report_c = gate_report("bugfix", c, b)  # made before the edit
    edit_set(b)
    problems = ledger.verify()
    assert len(problems) == 1 and problems[0].startswith("bugfix: the current baseline cassettes/b was changed outside the ledger")
    assert ledger.verify(check_fingerprints=False) == []
    with pytest.raises(LedgerRefused, match="the ledger does not verify"):
        ledger.promote("bugfix", c, report_c, reason="better")
    with pytest.raises(LedgerRefused, match="stale report"):
        ledger.promote("bugfix", c, gate_report("bugfix", c, b), reason="better")  # made after the edit
    ledger.rollback("bugfix", reason="b was edited by hand")  # the repair is allowed
    assert ledger.current("bugfix") == "cassettes/a" and ledger.verify() == []


def test_a_rollback_target_must_be_intact(ledger: Ledger) -> None:
    a, b = make_set("a"), make_set("b")
    ledger.promote("bugfix", a, None, reason="first", initial=True)
    ledger.promote("bugfix", b, gate_report("bugfix", b, a), reason="better")
    edit_set(a)
    with pytest.raises(LedgerRefused, match="cannot roll bugfix back to cassettes/a: it was changed outside the ledger"):
        ledger.rollback("bugfix", reason="undo")
    (b / "manifest.json").unlink()
    assert ledger.verify() == ["bugfix: the current baseline cassettes/b is no longer a cassette set (no manifest.json)"]


# ---- evaluations and the held-out budget ---------------------------------------------------------


def test_heldout_openings_count_distinct_candidates_per_split(ledger: Ledger) -> None:
    b, c, d = make_set("b"), make_set("c"), make_set("d")
    first = ledger.record_evaluation(gate_report("bugfix", b, None, split_sha256="1" * 64))
    ledger.record_evaluation(gate_report("bugfix", b, None, split_sha256="1" * 64))  # the same candidate again
    ledger.record_evaluation(gate_report("bugfix", c, None, promote=False, split_sha256="1" * 64), report_sha256="e" * 64)
    ledger.record_evaluation(gate_report("judge", d, None, split_sha256="2" * 64))
    assert ledger.heldout_openings("1" * 64) == 2
    assert ledger.heldout_openings("2" * 64) == 1
    assert ledger.heldout_openings("3" * 64) == 0
    assert ledger.status()["openings"] == {"1" * 64: 2, "2" * 64: 1}
    assert ledger.current("bugfix") is None  # evaluations promote nothing
    report = gate_report("bugfix", b, None, split_sha256="1" * 64)
    assert first == {**first, "kind": "evaluation", "feature": "bugfix", "split": "heldout", "split_sha256": "1" * 64,
                     "candidate": {"path": "cassettes/b", "fingerprint": fingerprint(b)}, "baseline": None, "promote": True,
                     "report_sha256": sha256_json(report)}  # fmt: skip
    assert ledger.rows[3]["report_sha256"] == "e" * 64
    assert ledger.verify() == []


def test_record_evaluation_refuses_what_is_not_a_report(ledger: Ledger) -> None:
    with pytest.raises(LedgerRefused, match="not a gate report"):
        ledger.record_evaluation({"schema": "other"})
    broken = gate_report("bugfix", make_set("b"), None)
    del broken["suite"]["split_sha256"]
    with pytest.raises(LedgerRefused, match="split_sha256"):
        ledger.record_evaluation(broken)
    assert not LEDGER.exists()


# ---- what a row may hold -------------------------------------------------------------------------


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_rows_hold_no_absolute_path_user_or_host(repo: Path, ledger: Ledger) -> None:
    a, b = make_set("a"), make_set("b")
    ledger.promote("bugfix", repo / a, None, reason="first", initial=True)  # an absolute path inside the tree
    report = gate_report("bugfix", b, a)
    report["candidate"]["source"] = str(repo / b)
    report["baseline"]["source"] = str(repo / a)
    ledger.record_evaluation(report)
    ledger.promote("bugfix", b, report, reason="better")
    ledger.rollback("bugfix", reason="undo")
    text = LEDGER.read_text(encoding="utf-8")
    assert str(repo) not in text
    for row in Ledger.open(LEDGER).rows:
        for value in _strings(row):
            assert not value.startswith("/"), value
        assert not {"user", "username", "host", "hostname"} & set(row)
    evaluation = ledger.rows[2]
    assert evaluation["candidate"]["path"] == "cassettes/b" and evaluation["baseline"]["path"] == "cassettes/a"
    assert ledger.rows[1]["to"]["path"] == "cassettes/a"
