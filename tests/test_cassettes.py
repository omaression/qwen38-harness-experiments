"""Cassettes on disk (cassettes.py): the writer-to-loader round trip, deterministic manifests, the
resume rules, the synthetic flag, concurrent writers, strict reading with file:line messages, and
the fingerprint the ledger relies on."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

import pytest

from llm_eval_harness.cassettes import (
    CASSETTE_SCHEMA,
    MANIFEST_FILE,
    CassetteError,
    CassetteSet,
    CassetteWriter,
    FeatureTape,
    Recording,
    fingerprint,
    load_cassettes,
    read_recordings,
)
from llm_eval_harness.records import ChatRequest, ChatResponse, Message, ModelSettings

SUITE_SHA = "ab" * 32  # stands in for a suite's sha256
OTHER_SHA = "cd" * 32


def request(item_id: str, *, model: str = "demo-model", settings: ModelSettings | None = None, text: str | None = None) -> ChatRequest:
    user = text if text is not None else f"Fix the function of {item_id}."
    return ChatRequest(item_id=item_id, model=model, messages=(Message("system", "You fix bugs."), Message("user", user)),
                       settings=settings or ModelSettings())


def response(content: str = '{"explanation": "e", "patched_code": "c"}', **fields: Any) -> ChatResponse:
    values: dict[str, Any] = {"reasoning": None, "finish_reason": "stop", "usage": {"prompt_tokens": 12, "completion_tokens": 5}, "latency_s": 0.25}
    values.update(fields)
    return ChatResponse(content=content, **values)


def writer(root: Path, **changes: Any) -> CassetteWriter:
    params: dict[str, Any] = {"arm": "candidate", "feature": "bugfix", "prompt": "bugfix/v2", "model": "demo-model",
                              "settings": ModelSettings(), "suite_name": "tiny", "suite_sha256": SUITE_SHA}
    params.update(changes)
    return CassetteWriter(root, **params)


def recorded_set(root: Path, items: tuple[str, ...] = ("bugfix:a", "bugfix:b"), **changes: Any) -> Path:
    w = writer(root, **changes)
    for item_id in items:
        w.write(request(item_id), response(content=f"answer to {item_id}"))
    w.close()
    return root


def manifest_of(root: Path) -> dict[str, Any]:
    return json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))


def snapshot(root: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(root.iterdir()) if p.is_file()}


def row(item_id: str = "bugfix:a", digest: str = "0" * 64, resp: Any = None, **extra: Any) -> str:
    return json.dumps({"item_id": item_id, "request_sha256": digest, "response": {"content": "x"} if resp is None else resp, **extra})


# ---- the round trip ---------------------------------------------------------------------------------


def test_writer_to_loader_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    w = writer(Path("set"), description="a first take")
    first, second = request("bugfix:t-clamp"), request("bugfix:t-parse-port", settings=ModelSettings())
    thinking = response(reasoning="The upper bound is hi.", usage={"prompt_tokens": 1, "completion_tokens": 2, "reasoning_tokens": 1}, latency_s=1.5)
    bare = ChatResponse(content="")
    w.write(first, thinking)
    w.write(second, bare)
    w.close()

    cset = load_cassettes("set")
    assert isinstance(cset, CassetteSet)
    assert (cset.root, cset.source, cset.arm, cset.synthetic, cset.description) == (Path("set"), "set", "candidate", False, "a first take")
    assert cset.fingerprint == fingerprint("set")
    assert dict(cset.features) == {"bugfix": FeatureTape(feature="bugfix", file="bugfix.jsonl", prompt="bugfix/v2", model="demo-model",
                                                         settings=ModelSettings(), suite="tiny", suite_sha256=SUITE_SHA, items=2)}
    recordings = read_recordings(cset, "bugfix")
    assert list(recordings) == ["bugfix:t-clamp", "bugfix:t-parse-port"]
    assert recordings["bugfix:t-clamp"] == Recording("bugfix:t-clamp", first.sha256(), thinking)
    assert recordings["bugfix:t-parse-port"] == Recording("bugfix:t-parse-port", second.sha256(), bare)


def test_rows_are_one_sorted_key_json_object_per_line(tmp_path: Path) -> None:
    root = recorded_set(tmp_path / "set")
    lines = (root / "bugfix.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert list(first) == ["item_id", "request_sha256", "response"]
    assert first["request_sha256"] == request("bugfix:a").sha256()
    assert first["response"] == response(content="answer to bugfix:a").to_json()


def test_text_survives_byte_for_byte(tmp_path: Path) -> None:
    content = 'é — 漢字 line para\x0cform\nnewline "quotes" \\ end'
    w = writer(tmp_path / "set")
    w.write(request("bugfix:u"), response(content=content, reasoning="pensée "))
    w.close()
    got = read_recordings(load_cassettes(tmp_path / "set"), "bugfix")["bugfix:u"].response
    assert (got.content, got.reasoning) == (content, "pensée ")


def test_rows_with_crlf_endings_read(tmp_path: Path) -> None:
    root = recorded_set(tmp_path / "set")
    path = root / "bugfix.jsonl"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert len(read_recordings(load_cassettes(root), "bugfix")) == 2


def test_two_features_share_one_set(tmp_path: Path) -> None:
    root = recorded_set(tmp_path / "set")
    judge = writer(root, feature="judge", prompt="judge/v2")
    judge.write(request("judge:a:reference"), response(content='{"verdict": true}'))
    judge.close()
    cset = load_cassettes(root)
    assert sorted(cset.features) == ["bugfix", "judge"]
    assert (cset.features["bugfix"].items, cset.features["judge"].items) == (2, 1)
    assert list(read_recordings(cset, "judge")) == ["judge:a:reference"]
    assert len(read_recordings(cset, "bugfix")) == 2


def test_an_empty_tape_still_loads(tmp_path: Path) -> None:
    w = writer(tmp_path / "deep" / "set")
    w.close()
    cset = load_cassettes(tmp_path / "deep" / "set")
    assert cset.features["bugfix"].items == 0
    assert (tmp_path / "deep" / "set" / "bugfix.jsonl").read_bytes() == b""
    assert read_recordings(cset, "bugfix") == {}


# ---- deterministic manifests ----------------------------------------------------------------------


def _record_demo(root: Path, order: tuple[str, ...]) -> None:
    for feature in order:
        w = writer(root, arm="baseline", feature=feature, prompt=f"{feature}/v1", synthetic=True, description="made by a script")
        for task in ("t-clamp", "t-count-vowels"):
            item_id = f"bugfix:{task}" if feature == "bugfix" else f"judge:{task}:reference"
            w.write(request(item_id), response(content=f"{feature} {task}", latency_s=0.5))
        w.close()


def test_manifest_bytes_are_deterministic(tmp_path: Path) -> None:
    _record_demo(tmp_path / "a", ("bugfix", "judge"))
    _record_demo(tmp_path / "b", ("judge", "bugfix"))
    a, b = snapshot(tmp_path / "a"), snapshot(tmp_path / "b")
    assert a == b and fingerprint(tmp_path / "a") == fingerprint(tmp_path / "b")

    tape = {"model": "demo-model", "settings": ModelSettings().to_json(), "suite": "tiny", "suite_sha256": SUITE_SHA, "items": 2}
    expected = {
        "schema": CASSETTE_SCHEMA,
        "arm": "baseline",
        "synthetic": True,
        "description": "made by a script",
        "features": {
            "bugfix": {"file": "bugfix.jsonl", "prompt": "bugfix/v1", **tape},
            "judge": {"file": "judge.jsonl", "prompt": "judge/v1", **tape},
        },
    }
    assert a[MANIFEST_FILE].decode("utf-8") == json.dumps(expected, indent=2, sort_keys=True) + "\n"

    # A resume that finds everything recorded rewrites the very same bytes.
    again = writer(tmp_path / "a", arm="baseline", feature="judge", prompt="judge/v1", synthetic=True)
    assert again.has(request("judge:t-clamp:reference"))
    again.close()
    again.close()
    assert snapshot(tmp_path / "a") == a


def test_a_resume_without_a_description_keeps_the_old_one(tmp_path: Path) -> None:
    root = recorded_set(tmp_path / "set", description="kept")
    writer(root).close()
    assert manifest_of(root)["description"] == "kept"
    writer(root, description="replaced").close()
    assert manifest_of(root)["description"] == "replaced"


# ---- resuming -----------------------------------------------------------------------------------


def test_a_resume_indexes_what_is_on_disk(tmp_path: Path) -> None:
    root = recorded_set(tmp_path / "set")
    before = (root / "bugfix.jsonl").read_bytes()
    w = writer(root)
    assert w.has(request("bugfix:a")) and w.has(request("bugfix:b"))
    assert w.response_for(request("bugfix:a")) == response(content="answer to bugfix:a")
    assert not w.has(request("bugfix:c"))
    changed = request("bugfix:a", text="the task was edited")
    assert not w.has(changed) and w.response_for(changed) is None  # same item, another request
    w.write(request("bugfix:c"), response(content="answer to bugfix:c"))
    w.close()
    after = (root / "bugfix.jsonl").read_bytes()
    assert after.startswith(before) and after.count(b"\n") == 3
    cset = load_cassettes(root)
    assert cset.features["bugfix"].items == 3 and len(read_recordings(cset, "bugfix")) == 3


def test_the_writer_refuses_a_second_row_for_an_item(tmp_path: Path) -> None:
    w = writer(tmp_path / "set")
    w.write(request("bugfix:a"), response())
    with pytest.raises(CassetteError, match="already holds item 'bugfix:a' for this very request"):
        w.write(request("bugfix:a"), response(content="again"))
    with pytest.raises(CassetteError, match="for another request: record into a new directory"):
        w.write(request("bugfix:a", text="edited"), response())
    w.close()
    assert len(read_recordings(load_cassettes(tmp_path / "set"), "bugfix")) == 1


@pytest.mark.parametrize(
    "change, named",
    [
        ({"prompt": "bugfix/v1"}, "prompt 'bugfix/v2', not 'bugfix/v1'"),
        ({"model": "other-model"}, "model 'demo-model', not 'other-model'"),
        ({"settings": ModelSettings(temperature=0.7)}, "settings"),
        ({"settings": ModelSettings(seed=None)}, "settings"),
        ({"settings": ModelSettings(extra={"top_p": 0.9})}, "settings"),
        ({"suite_sha256": OTHER_SHA}, "suite"),
        ({"suite_name": None, "suite_sha256": None}, "suite"),
        ({"arm": "baseline"}, "arm 'candidate', not 'baseline'"),
    ],
)
def test_a_conflicting_resume_is_refused_and_writes_nothing(tmp_path: Path, change: dict[str, Any], named: str) -> None:
    root = recorded_set(tmp_path / "set")
    before = snapshot(root)
    other = writer(root, **change)
    with pytest.raises(CassetteError, match="record into a new directory") as refused:
        other.has(request("bugfix:a"))
    assert named in str(refused.value) and "manifest.json" in str(refused.value)
    new = request("bugfix:new", model=other.model, settings=other.settings)
    with pytest.raises(CassetteError, match="record into a new directory"):
        other.write(new, response())
    with pytest.raises(CassetteError, match="record into a new directory"):
        other.close()  # close() repeats the refusal and never overwrites the manifest
    assert snapshot(root) == before


def test_a_feature_may_not_take_another_features_file(tmp_path: Path) -> None:
    root = tmp_path / "set"
    root.mkdir()
    (root / "bugfix.jsonl").write_text("", encoding="utf-8")
    document = {"schema": CASSETTE_SCHEMA, "arm": "candidate", "synthetic": False, "description": "",
                "features": {"judge": {"file": "bugfix.jsonl", "prompt": "judge/v2", "model": "demo-model",
                                       "settings": ModelSettings().to_json(), "items": 0}}}
    (root / MANIFEST_FILE).write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CassetteError, match="keeps feature 'judge' in 'bugfix.jsonl'"):
        writer(root).close()


def test_a_resume_follows_the_file_the_manifest_names(tmp_path: Path) -> None:
    root = tmp_path / "set"
    root.mkdir()
    (root / "fixes.jsonl").write_text(row("bugfix:a", request("bugfix:a").sha256()) + "\n", encoding="utf-8")
    document = {"schema": CASSETTE_SCHEMA, "arm": "candidate", "synthetic": False,
                "features": {"bugfix": {"file": "fixes.jsonl", "prompt": "bugfix/v2", "model": "demo-model",
                                        "settings": ModelSettings().to_json(), "suite": "tiny", "suite_sha256": SUITE_SHA, "items": 1}}}
    (root / MANIFEST_FILE).write_text(json.dumps(document), encoding="utf-8")
    w = writer(root)
    assert w.has(request("bugfix:a"))
    w.write(request("bugfix:b"), response())
    w.close()
    assert not (root / "bugfix.jsonl").exists()
    assert manifest_of(root)["features"]["bugfix"]["file"] == "fixes.jsonl"
    assert len(read_recordings(load_cassettes(root), "bugfix")) == 2


def test_an_incomplete_last_line_is_dropped_on_resume(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = recorded_set(tmp_path / "set")
    path = root / "bugfix.jsonl"
    complete = path.read_bytes()
    path.write_bytes(complete + b'{"item_id": "bugfix:c", "request_sha2')  # an append cut short
    with pytest.raises(CassetteError, match=r"bugfix\.jsonl:3: not JSON"):
        read_recordings(load_cassettes(root), "bugfix")  # the reader never repairs anything
    w = writer(root)
    assert not w.has(request("bugfix:c"))
    assert "dropping an incomplete last line" in capsys.readouterr().err
    assert path.read_bytes() == complete
    w.write(request("bugfix:c"), response())
    w.close()
    assert sorted(read_recordings(load_cassettes(root), "bugfix")) == ["bugfix:a", "bugfix:b", "bugfix:c"]


def test_a_complete_last_line_without_its_newline_is_kept(tmp_path: Path) -> None:
    root = recorded_set(tmp_path / "set")
    path = root / "bugfix.jsonl"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    assert len(read_recordings(load_cassettes(root), "bugfix")) == 2
    w = writer(root)
    assert w.has(request("bugfix:b"))
    w.write(request("bugfix:c"), response())
    w.close()
    assert path.read_bytes().count(b"\n") == 3
    assert len(read_recordings(load_cassettes(root), "bugfix")) == 3


def test_a_damaged_tape_is_not_resumed(tmp_path: Path) -> None:
    root = recorded_set(tmp_path / "set")
    path = root / "bugfix.jsonl"
    path.write_text("not json\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(CassetteError, match=r"bugfix\.jsonl:1: not JSON"):
        writer(root).has(request("bugfix:a"))


# ---- what the writer refuses up front ---------------------------------------------------------------


@pytest.mark.parametrize(
    "change, message",
    [
        ({"feature": "summary"}, "unknown feature"),
        ({"arm": " "}, "arm must be a non-empty string"),
        ({"model": ""}, "model must be a non-empty string"),
        ({"settings": ModelSettings(temperature=0)}, "would replay as"),  # an int temperature hashes as 0, replays as 0.0
        ({"settings": ModelSettings(max_tokens=4096.0)}, "max_tokens must be a positive integer"),  # type: ignore[arg-type]
        ({"settings": ModelSettings(temperature=float("nan"))}, "cannot be stored"),
        ({"settings": ModelSettings(extra={"model": "other"})}, "cannot be stored"),
        ({"settings": {"temperature": 0.0}}, "records.ModelSettings"),
    ],
)
def test_the_writer_refuses_what_could_never_replay(tmp_path: Path, change: dict[str, Any], message: str) -> None:
    with pytest.raises(CassetteError, match=message):
        writer(tmp_path / "set", **change)
    assert not (tmp_path / "set").exists()


@pytest.mark.parametrize(
    "made, message",
    [
        (lambda: request("bugfix:a", model="another-model"), "would never replay"),
        (lambda: request("bugfix:a", settings=ModelSettings(temperature=0.3)), "would never replay"),
    ],
)
def test_write_refuses_a_request_the_tape_does_not_describe(tmp_path: Path, made: Callable[[], ChatRequest], message: str) -> None:
    w = writer(tmp_path / "set")
    with pytest.raises(CassetteError, match=message):
        w.write(made(), response())


@pytest.mark.parametrize(
    "bad, message",
    [
        (response(usage={"completion_tokens": 1.5}), "usage must be an object of integers"),
        (response(latency_s=-1.0), "latency_s must be a number >= 0"),
        (response(latency_s=float("inf")), "cannot be stored as JSON"),
    ],
)
def test_write_refuses_a_response_the_reader_would_refuse(tmp_path: Path, bad: ChatResponse, message: str) -> None:
    w = writer(tmp_path / "set")
    with pytest.raises(CassetteError, match=message):
        w.write(request("bugfix:a"), bad)
    w.close()
    assert read_recordings(load_cassettes(tmp_path / "set"), "bugfix") == {}


# ---- the synthetic flag -------------------------------------------------------------------------------


def test_the_synthetic_flag_is_the_writers_and_never_mixed(tmp_path: Path) -> None:
    demo = recorded_set(tmp_path / "demo", synthetic=True)
    live = recorded_set(tmp_path / "live")
    assert load_cassettes(demo).synthetic is True and manifest_of(demo)["synthetic"] is True
    assert load_cassettes(live).synthetic is False and manifest_of(live)["synthetic"] is False

    for root, flag in ((demo, False), (live, True)):
        before = snapshot(root)
        for feature in ("bugfix", "judge"):  # not even another feature may join with the other flag
            mixed = writer(root, feature=feature, prompt=f"{feature}/v2", synthetic=flag)
            with pytest.raises(CassetteError, match="never mixes them"):
                mixed.close()
        assert snapshot(root) == before


# ---- concurrency ------------------------------------------------------------------------------------


def test_eight_threads_write_one_valid_complete_file(tmp_path: Path) -> None:
    root = tmp_path / "set"
    w = writer(root)
    start = threading.Barrier(8)
    errors: list[BaseException] = []

    def work(thread: int) -> None:
        try:
            start.wait()
            for i in range(25):
                item_id = f"bugfix:t{thread}-{i}"
                # Rows of 6 to 60 KB: far beyond any stream buffer, so unlocked writes would interleave.
                w.write(request(item_id), response(content=f"{item_id} é " + "x" * (6000 * (1 + i % 10))))
        except BaseException as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    w.close()

    lines = (root / "bugfix.jsonl").read_bytes().split(b"\n")
    assert lines[-1] == b"" and len(lines) == 201
    assert all(json.loads(line)["item_id"].startswith("bugfix:t") for line in lines[:-1])
    recordings = read_recordings(load_cassettes(root), "bugfix")
    assert len(recordings) == 200 and load_cassettes(root).features["bugfix"].items == 200
    assert all(recordings[f"bugfix:t{t}-{i}"].response.content.startswith(f"bugfix:t{t}-{i} ") for t in range(8) for i in range(25))


def test_writers_of_two_features_close_into_one_manifest(tmp_path: Path) -> None:
    root = tmp_path / "set"
    for _ in range(5):
        writers = [writer(root, feature="bugfix"), writer(root, feature="judge", prompt="judge/v2")]
        start = threading.Barrier(2)

        def close(w: CassetteWriter) -> None:
            start.wait()
            w.close()

        threads = [threading.Thread(target=close, args=(w,)) for w in writers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(manifest_of(root)["features"]) == ["bugfix", "judge"]


# ---- strict reading: the manifest ---------------------------------------------------------------------


def _valid_manifest() -> dict[str, Any]:
    return {
        "schema": CASSETTE_SCHEMA,
        "arm": "baseline",
        "synthetic": True,
        "description": "d",
        "features": {"bugfix": {"file": "bugfix.jsonl", "prompt": "bugfix/v1", "model": "demo-model-a",
                                "settings": {"temperature": 0.0, "max_tokens": 4096, "seed": 0, "extra": {}},
                                "suite": "demo", "suite_sha256": SUITE_SHA, "items": 0}},
    }


def _write_set(root: Path, document: Any) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_FILE).write_text(document if isinstance(document, str) else json.dumps(document), encoding="utf-8")
    (root / "bugfix.jsonl").write_text("", encoding="utf-8")
    return root


def test_a_minimal_hand_written_manifest_loads(tmp_path: Path) -> None:
    document = _valid_manifest()
    del document["description"]
    tape = document["features"]["bugfix"]
    del tape["suite"], tape["suite_sha256"]
    tape["settings"] = {"temperature": 0}
    cset = load_cassettes(_write_set(tmp_path / "set", document))
    assert cset.description == "" and cset.features["bugfix"].suite is None
    # records.ModelSettings.from_json: a missing seed is no seed (none is sent), the rest default.
    assert cset.features["bugfix"].settings == ModelSettings(temperature=0.0, max_tokens=4096, seed=None, extra={})


def _tape(m: dict[str, Any]) -> dict[str, Any]:
    return m["features"]["bugfix"]


MANIFEST_DAMAGE: list[tuple[str, Callable[[dict[str, Any]], Any], str]] = [
    ("schema", lambda m: m.update(schema="llm_eval_harness.cassettes/0"), "schema"),
    ("no arm", lambda m: m.pop("arm"), "lacks 'arm'"),
    ("empty arm", lambda m: m.update(arm=""), "arm must be"),
    ("unknown key", lambda m: m.update(recorded_at="today"), "unknown key 'recorded_at'"),
    ("synthetic as text", lambda m: m.update(synthetic="yes"), "synthetic must be true or false"),
    ("description type", lambda m: m.update(description=None), "description must be a string"),
    ("features type", lambda m: m.update(features=[]), "features is a JSON array"),
    ("unknown feature", lambda m: m["features"].update(summary=_tape(m)), "unknown feature"),
    ("tape type", lambda m: m["features"].update(bugfix="bugfix.jsonl"), "is a JSON string"),
    ("no items", lambda m: _tape(m).pop("items"), "lacks 'items'"),
    ("tape unknown key", lambda m: _tape(m).update(prompt_sha256="x"), "unknown key 'prompt_sha256'"),
    ("items negative", lambda m: _tape(m).update(items=-1), "items must be"),
    ("items bool", lambda m: _tape(m).update(items=True), "items must be"),
    ("empty prompt", lambda m: _tape(m).update(prompt=""), "prompt must be"),
    ("model type", lambda m: _tape(m).update(model=3), "model must be"),
    ("suite type", lambda m: _tape(m).update(suite_sha256=5), "suite_sha256 must be"),
    ("file outside", lambda m: _tape(m).update(file="../bugfix.jsonl"), "must name a"),
    ("file below", lambda m: _tape(m).update(file="sub/bugfix.jsonl"), "must name a"),
    ("file suffix", lambda m: _tape(m).update(file="bugfix.txt"), "must name a"),
    ("file hidden", lambda m: _tape(m).update(file=".jsonl"), "must name a"),
    ("file manifest", lambda m: _tape(m).update(file=MANIFEST_FILE), "must name a"),
    ("file type", lambda m: _tape(m).update(file=7), "must name a"),
    ("shared file", lambda m: m["features"].update(judge=dict(_tape(m), prompt="judge/v1")), "share the file"),
    ("settings type", lambda m: _tape(m).update(settings=None), "settings is a JSON null"),
    ("settings key", lambda m: _tape(m)["settings"].update(top_p=0.9), "unknown key 'top_p'"),
    ("temperature text", lambda m: _tape(m)["settings"].update(temperature="0.0"), "temperature must be"),
    ("temperature negative", lambda m: _tape(m)["settings"].update(temperature=-0.5), "temperature must be"),
    ("max_tokens zero", lambda m: _tape(m)["settings"].update(max_tokens=0), "max_tokens must be"),
    ("max_tokens float", lambda m: _tape(m)["settings"].update(max_tokens=4096.0), "max_tokens must be"),
    ("seed float", lambda m: _tape(m)["settings"].update(seed=1.5), "seed must be"),
    ("extra type", lambda m: _tape(m)["settings"].update(extra=[]), "extra must be an object"),
    ("extra override", lambda m: _tape(m)["settings"].update(extra={"messages": []}), "may not override"),
]


@pytest.mark.parametrize("damage, message", [(d, msg) for _, d, msg in MANIFEST_DAMAGE], ids=[name for name, _, _ in MANIFEST_DAMAGE])
def test_a_malformed_manifest_is_refused_naming_the_file(tmp_path: Path, damage: Callable[[dict[str, Any]], Any], message: str) -> None:
    document = _valid_manifest()
    damage(document)
    root = _write_set(tmp_path / "set", document)
    with pytest.raises(CassetteError) as refused:
        load_cassettes(root)
    assert "manifest.json: " in str(refused.value) and message in str(refused.value)


@pytest.mark.parametrize(
    "text, message",
    [
        ("{not json", "manifest.json:1: not JSON"),
        ('{\n"schema": "x",\n"schema": "y"}', "duplicate key 'schema'"),
        ('{"schema": NaN}', "NaN is not JSON"),
        ("[1, 2]", "a JSON array, not an object"),
    ],
)
def test_a_manifest_that_is_not_one_json_object_is_refused(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(CassetteError, match=message):
        load_cassettes(_write_set(tmp_path / "set", text))


def test_a_directory_without_a_manifest_is_not_a_set(tmp_path: Path) -> None:
    with pytest.raises(CassetteError, match="holds no manifest.json"):
        load_cassettes(tmp_path)
    with pytest.raises(CassetteError, match="holds no manifest.json"):
        load_cassettes(tmp_path / "absent")
    (tmp_path / MANIFEST_FILE).write_bytes(b"\xff\xfe{}")
    with pytest.raises(CassetteError, match="not UTF-8"):
        load_cassettes(tmp_path)


# ---- strict reading: the recordings ---------------------------------------------------------------


def test_a_duplicate_item_id_is_refused_with_file_and_line(tmp_path: Path) -> None:
    root = recorded_set(tmp_path / "set")
    path = root / "bugfix.jsonl"
    first = path.read_text(encoding="utf-8").splitlines()[0]
    path.write_text(path.read_text(encoding="utf-8") + "\n" + first + "\n", encoding="utf-8")
    with pytest.raises(CassetteError, match=r"bugfix\.jsonl:4: item 'bugfix:a' appears twice \(first on line 1\)"):
        read_recordings(load_cassettes(root), "bugfix")


GOOD = "0" * 64
BAD_LINES: list[tuple[str, str, str]] = [
    ("not json", "{not json", "not JSON"),
    ("array", "[1, 2]", "a JSON array, not an object"),
    ("no response", json.dumps({"item_id": "bugfix:x", "request_sha256": GOOD}), "lacks 'response'"),
    ("unknown key", row("bugfix:x", GOOD, note="hand-edited"), "unknown key 'note'"),
    ("empty item", row("", GOOD), "item_id must be a non-empty string"),
    ("item type", row(5, GOOD), "item_id must be"),  # type: ignore[arg-type]
    ("short hash", row("bugfix:x", "abc"), "not 64 lowercase hex digits"),
    ("upper hash", row("bugfix:x", "A" * 64), "not 64 lowercase hex digits"),
    ("response text", row("bugfix:x", GOOD, "the answer"), "is a JSON string, not an object"),
    ("response key", row("bugfix:x", GOOD, {"content": "x", "tool_calls": []}), "unknown key 'tool_calls'"),
    ("content type", row("bugfix:x", GOOD, {"content": 5}), "content must be a string or null"),
    ("finish type", row("bugfix:x", GOOD, {"finish_reason": ["stop"]}), "finish_reason must be"),
    ("usage float", row("bugfix:x", GOOD, {"usage": {"completion_tokens": 1.5}}), "usage must be an object of integers"),
    ("usage bool", row("bugfix:x", GOOD, {"usage": {"completion_tokens": True}}), "usage must be an object of integers"),
    ("latency negative", row("bugfix:x", GOOD, {"latency_s": -1}), "latency_s must be"),
    ("latency nan", '{"item_id": "bugfix:x", "request_sha256": "%s", "response": {"latency_s": NaN}}' % GOOD, "NaN is not JSON"),
    ("duplicate key", '{"item_id": "bugfix:x", "item_id": "bugfix:y", "request_sha256": "%s", "response": {}}' % GOOD, "duplicate key 'item_id'"),
]


@pytest.mark.parametrize("line, message", [(line, msg) for _, line, msg in BAD_LINES], ids=[name for name, _, _ in BAD_LINES])
def test_a_malformed_line_is_refused_with_file_and_line(tmp_path: Path, line: str, message: str) -> None:
    root = recorded_set(tmp_path / "set", items=("bugfix:a",))
    path = root / "bugfix.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n" + line + "\n", encoding="utf-8")  # line 2 is blank
    with pytest.raises(CassetteError) as refused:
        read_recordings(load_cassettes(root), "bugfix")
    assert "bugfix.jsonl:3: " in str(refused.value) and message in str(refused.value)


def test_a_line_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    root = recorded_set(tmp_path / "set", items=("bugfix:a",))
    with (root / "bugfix.jsonl").open("ab") as handle:
        handle.write(b'{"item_id": "\xff"}\n')
    with pytest.raises(CassetteError, match=r"bugfix\.jsonl:2: not UTF-8"):
        read_recordings(load_cassettes(root), "bugfix")


def test_a_minimal_hand_written_row_reads_with_the_record_defaults(tmp_path: Path) -> None:
    root = recorded_set(tmp_path / "set", items=())
    (root / "bugfix.jsonl").write_text(row("bugfix:x", GOOD, {"content": None, "usage": None}) + "\n", encoding="utf-8")
    got = read_recordings(load_cassettes(root), "bugfix")["bugfix:x"]
    assert got.response == ChatResponse(content="")


def test_reading_a_feature_the_set_lacks_or_a_missing_file(tmp_path: Path) -> None:
    root = recorded_set(tmp_path / "set")
    cset = load_cassettes(root)
    with pytest.raises(CassetteError, match=r"no recordings for feature 'judge' \(its manifest has: 'bugfix'\)"):
        read_recordings(cset, "judge")
    (root / "bugfix.jsonl").unlink()
    with pytest.raises(CassetteError, match=r"bugfix\.jsonl: missing"):
        read_recordings(cset, "bugfix")


# ---- the fingerprint ----------------------------------------------------------------------------


def test_the_fingerprint_changes_with_any_recorded_byte(tmp_path: Path) -> None:
    root = recorded_set(tmp_path / "set", items=("bugfix:a",))
    original = fingerprint(root)
    assert load_cassettes(root).fingerprint == original
    for name in ("bugfix.jsonl", MANIFEST_FILE):
        path = root / name
        data = path.read_bytes()
        seen = {original}
        for position in range(len(data)):
            flipped = bytearray(data)
            flipped[position] ^= 0x01
            path.write_bytes(bytes(flipped))
            seen.add(fingerprint(root))
        path.write_bytes(data)
        assert len(seen) == len(data) + 1  # every single-byte edit gives a fingerprint of its own
    assert fingerprint(root) == original
    (root / "bugfix.jsonl").write_bytes((root / "bugfix.jsonl").read_bytes() + b"\n")
    assert fingerprint(root) != original  # even a trailing newline counts


def test_the_fingerprint_covers_only_the_manifest_and_tapes(tmp_path: Path) -> None:
    root = recorded_set(tmp_path / "set")
    original = fingerprint(root)
    (root / "NOTES.md").write_text("how this set was made\n", encoding="utf-8")
    (root / "bugfix.jsonl.bak").write_text("old\n", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "bugfix.jsonl").write_text("{}\n", encoding="utf-8")
    assert fingerprint(root) == original
    (root / "stray.jsonl").write_text("", encoding="utf-8")
    assert fingerprint(root) != original
    (root / "stray.jsonl").unlink()
    assert fingerprint(root) == original
    (root / "extra.jsonl.gz").write_bytes(b"gz")
    assert fingerprint(root) != original


def test_compressed_cassettes_round_trip_determinism_resume_and_guards(tmp_path: Path) -> None:
    import gzip

    dir1, dir2 = tmp_path / "c1", tmp_path / "c2"
    r1, r2 = request("bugfix:a"), request("bugfix:b")
    resp1, resp2 = response(content="answer a — 漢字"), response(content="answer b")

    w1 = writer(dir1, compress=True)
    w1.write(r1, resp1)
    w1.close()
    assert (dir1 / "bugfix.jsonl.gz").is_file() and not (dir1 / "bugfix.jsonl").exists()
    assert manifest_of(dir1)["features"]["bugfix"]["file"] == "bugfix.jsonl.gz"

    # Resume and append second item.
    w1_resumed = writer(dir1, compress=True)
    assert w1_resumed.has(r1) and not w1_resumed.has(r2)
    w1_resumed.write(r2, resp2)
    w1_resumed.close()

    # Independent identical write must produce identical gzip bytes (mtime=0).
    w2 = writer(dir2, compress=True)
    w2.write(r1, resp1)
    w2.write(r2, resp2)
    w2.close()
    assert (dir1 / "bugfix.jsonl.gz").read_bytes() == (dir2 / "bugfix.jsonl.gz").read_bytes()
    assert fingerprint(dir1) == fingerprint(dir2)

    loaded = load_cassettes(dir1)
    recs = read_recordings(loaded, "bugfix")
    assert recs["bugfix:a"].response == resp1 and recs["bugfix:b"].response == resp2

    # Refuses mixing compress modes on the same feature.
    with pytest.raises(CassetteError, match="gzipped"):
        writer(dir1, compress=False).has(r1)

    plain_dir = recorded_set(tmp_path / "plain")
    with pytest.raises(CassetteError, match="not gzipped"):
        writer(plain_dir, compress=True).has(r1)

    # Corrupt or empty gzip tape raises CassetteError.
    (dir2 / "bugfix.jsonl.gz").write_bytes(b"not-gzip")
    with pytest.raises(CassetteError, match="not valid gzip"):
        read_recordings(load_cassettes(dir2), "bugfix")
    (dir2 / "bugfix.jsonl.gz").write_bytes(b"")
    with pytest.raises(CassetteError, match="not valid gzip"):
        read_recordings(load_cassettes(dir2), "bugfix")
    (dir2 / "bugfix.jsonl.gz").write_bytes(gzip.compress(b'{"item_id":"x"}\n'))
    with pytest.raises(CassetteError, match=r"bugfix\.jsonl\.gz:1: "):
        read_recordings(load_cassettes(dir2), "bugfix")
