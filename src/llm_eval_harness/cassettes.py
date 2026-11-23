"""Cassettes: recorded model responses on disk, so that CI replays a model instead of calling one.

    load_cassettes(root)          a cassette set's manifest -> CassetteSet
    read_recordings(cset, feat)   item id -> Recording, for one feature
    fingerprint(root)             the set's identity (what the ledger and the report record)
    CassetteWriter                writes a set while a live model answers (clients.RecordingClient)

LAYOUT. A cassette set is one directory, one arm:

    <root>/manifest.json
        {"schema": "llm_eval_harness.cassettes/1",
         "arm": "baseline",
         "synthetic": true,
         "description": "...",
         "features": {
            "bugfix": {"file": "bugfix.jsonl", "prompt": "bugfix/v1", "model": "demo-model-a",
                       "settings": {"temperature": 0.0, "max_tokens": 4096, "seed": 0, "extra": {}},
                       "suite": "demo", "suite_sha256": "<sha256>", "items": 32},
            "judge": {...}}}
    <root>/<feature>.jsonl
        one recording per line, sorted by item id when written by the demo script:
        {"item_id": "bugfix:clamp-range", "request_sha256": "<sha256>",
         "response": {"content": "...", "reasoning": null, "finish_reason": "stop",
                      "usage": {"prompt_tokens": 312, "completion_tokens": 140}, "latency_s": 0.84}}

`synthetic` is true when the responses were written by hand or by a script, not by a model (the
demo sets). Every report shows it per arm, and the markdown summary says so in its first line.

READING IS STRICT, because a recording that loads wrongly is scored wrongly:
  - load_cassettes refuses a manifest that is not one JSON object with exactly the keys above
    ("description" may be left out, and so may a tape's "suite" and "suite_sha256" and any of its
    settings, which then take records.ModelSettings.from_json's defaults), a value of the wrong
    type, a feature records.FEATURES does not name, settings whose `extra` would
    override a request key, two features sharing a file, and a tape file that is not a *.jsonl
    file directly in the directory (the fingerprint covers exactly those files, so a tape
    anywhere else would escape it);
  - read_recordings refuses a line that is not one JSON object with exactly item_id,
    request_sha256 (64 lowercase hex digits) and response (some or all of the keys of
    records.ChatResponse.to_json, each of its type: strings or null, usage an object of integers,
    latency_s a number >= 0 or null), and an item id that appears twice;
  - both refuse duplicate keys and NaN or Infinity, which Python's json module accepts.
Every message names the file, and for a recording its line: "bugfix.jsonl:7: ...".

REPLAY CHECKS (clients.ReplayClient). A recording is used only for the request it answers: the
replaying side rebuilds the request with the manifest's prompt, model and settings
(prompts.build_request) and compares its sha256 with `request_sha256`. A mismatch means the task,
the template or the settings changed since recording, and the answer may not fit the question;
replay refuses (clients.CassetteMismatch) instead of scoring it.

WRITING. CassetteWriter appends each recording as soon as it arrives (flushed, so an interrupted
recording keeps what it got) and writes manifest.json last, atomically. Opening an existing set
resumes it: items already recorded with the same request hash are skipped by the caller
(CassetteWriter.has). A set whose manifest names another prompt, model or settings for the
feature is refused: record into a new directory instead. Duplicate item ids in a file are an
error when read. In detail:
  - The first call of has(), response_for(), write() or close() reads what the directory holds.
    It refuses (CassetteError, "record into a new directory") a manifest of another arm, one
    that records this feature with another prompt, model, settings or suite (the manifest
    describes every row of a tape, so a tape never mixes them), and one whose synthetic flag
    differs from the writer's: a set never mixes synthetic and recorded responses. After a
    refusal the writer never writes anything, and close() raises the refusal again.
  - The writer refuses settings that would not survive the manifest's JSON round trip (an
    integer temperature, say): their recordings could never match a replayed request.
  - write() refuses an item id the file already holds (so the reader never meets a duplicate the
    writer made) and a request whose model or settings are not the tape's (it would never
    replay). Each row is parsed back with the reader's own checks before it is appended.
  - A last line without its newline is what an interrupted append leaves. On resume it is kept
    when it parses (the next row starts on a new line), and otherwise dropped with a warning on
    stderr: its item is asked again. read_recordings never repairs anything.
  - close() writes manifest.json as json.dumps(indent=2, sort_keys=True) plus a newline, with no
    time stamp, so the same recordings always give the same bytes. It keeps the other features'
    tapes; "items" is the number of distinct item ids in the tape. Writers of different features
    may share a directory; one process writes a directory at a time.

FINGERPRINT. sha256 over the sorted list of [file name, sha256 of its bytes] for manifest.json
and every *.jsonl file directly in the directory (util.sha256_json of that list). Any edit of a
recorded response changes it, which is how the ledger notices a baseline edited outside it.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .records import FEATURES, ChatRequest, ChatResponse, ModelSettings
from .util import canonical_json, display_path, dumps_jsonl_row, log, sha256_file, sha256_json, write_text_atomic

CASSETTE_SCHEMA = "llm_eval_harness.cassettes/1"
MANIFEST_FILE = "manifest.json"


class CassetteError(ValueError):
    """A cassette set cannot be read or written as asked: no manifest, a wrong schema, a feature
    the set has no recordings for, a duplicate item id, a conflicting resume."""


@dataclass(frozen=True)
class FeatureTape:
    """One feature's entry in a manifest."""

    feature: str
    file: str
    prompt: str
    model: str
    settings: ModelSettings
    suite: str | None
    suite_sha256: str | None
    items: int


@dataclass(frozen=True)
class CassetteSet:
    """A loaded manifest. `root` is the directory as given; `source` its util.display_path."""

    root: Path
    source: str
    arm: str
    synthetic: bool
    description: str
    features: Mapping[str, FeatureTape]
    fingerprint: str


@dataclass(frozen=True)
class Recording:
    """One recorded answer: the item it belongs to, the hash of the request it answered, the response."""

    item_id: str
    request_sha256: str
    response: ChatResponse


def load_cassettes(root: str | os.PathLike[str]) -> CassetteSet:
    """Read <root>/manifest.json (CASSETTE_SCHEMA) into a CassetteSet, with fingerprint(root).
    Raises CassetteError."""
    directory = Path(root)
    manifest = _read_manifest(directory)
    return CassetteSet(
        root=directory,
        source=display_path(directory),
        arm=manifest.arm,
        synthetic=manifest.synthetic,
        description=manifest.description,
        features=manifest.features,
        fingerprint=fingerprint(directory),
    )


def read_recordings(cset: CassetteSet, feature: str) -> dict[str, Recording]:
    """The recordings of `feature`, by item id. CassetteError when the set has no tape for the
    feature, when a line is malformed, or when an item id appears twice."""
    tape = cset.features.get(feature)
    if tape is None:
        listed = _names(sorted(cset.features)) if cset.features else "none"
        raise CassetteError(f"{cset.source} has no recordings for feature {feature!r} (its manifest has: {listed})")
    path = Path(cset.root) / tape.file
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise CassetteError(f"{display_path(path)}: missing, although {MANIFEST_FILE} names it for {feature!r}") from None
    recordings = _parse_rows(data, display_path(path))
    return {item_id: recordings[item_id] for item_id in sorted(recordings)}


def fingerprint(root: str | os.PathLike[str]) -> str:
    """The set's identity: see FINGERPRINT above. CassetteError when the directory has no
    manifest.json. (Complete in the skeleton: the ledger depends on it.)"""
    directory = Path(root)
    if not (directory / MANIFEST_FILE).is_file():
        raise CassetteError(f"{display_path(directory)} holds no {MANIFEST_FILE}: not a cassette set")
    names = sorted(p.name for p in directory.iterdir() if p.is_file() and (p.name == MANIFEST_FILE or p.suffix == ".jsonl"))
    return sha256_json([[name, sha256_file(directory / name)] for name in names])


@dataclass
class CassetteWriter:
    """Records one feature of one arm into `root` (thread-safe: write() may be called from the
    recorder's worker threads).

        writer = CassetteWriter(root, arm="candidate", feature="bugfix", prompt="bugfix/v2",
                                model="my-model", settings=ModelSettings(),
                                suite_name=suite.name, suite_sha256=suite.sha256)
        if not writer.has(request): writer.write(request, response)
        writer.close()   # writes manifest.json; also on error, so a resume knows what is there

    `synthetic` is False for anything a live model answered; only the demo script sets it.
    See WRITING above for what is refused, and when."""

    root: Path
    arm: str
    feature: str
    prompt: str
    model: str
    settings: ModelSettings
    suite_name: str | None = None
    suite_sha256: str | None = None
    synthetic: bool = False
    description: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # Filled on first use (_open): the tape's file name, its rows by item id, whether the file
    # ends in an unterminated line, and a refusal to repeat.
    _file: str = field(default="", init=False, repr=False, compare=False)
    _index: dict[str, Recording] = field(default_factory=dict, init=False, repr=False, compare=False)
    _newline_due: bool = field(default=False, init=False, repr=False, compare=False)
    _opened: bool = field(default=False, init=False, repr=False, compare=False)
    _refusal: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.feature not in FEATURES:
            raise CassetteError(f"unknown feature {self.feature!r} (this harness has {_names(FEATURES)})")
        for name in ("arm", "prompt", "model"):
            if not _is_name(getattr(self, name)):
                raise CassetteError(f"the writer's {name} must be a non-empty string")
        for name in ("suite_name", "suite_sha256"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), str):
                raise CassetteError(f"the writer's {name} must be a string or None")
        if not isinstance(self.synthetic, bool) or not isinstance(self.description, str):
            raise CassetteError("the writer's synthetic must be a bool and its description a string")
        if not isinstance(self.settings, ModelSettings):
            raise CassetteError("the writer's settings must be a records.ModelSettings")
        try:
            given = _settings_json(self.settings)
            stored = _parse_settings(_strict_loads(given), "settings")
        except (TypeError, ValueError, _Invalid) as exc:
            raise CassetteError(f"these settings cannot be stored in a manifest: {exc}") from None
        if _settings_json(stored) != given:
            raise CassetteError(
                f"the settings {given} would replay as {_settings_json(stored)} after the manifest's JSON round trip "
                "(an integer temperature, say), so no recording would match its replayed request"
            )

    def has(self, request: ChatRequest) -> bool:
        """Whether a recording for request.item_id with request.sha256() is already on disk."""
        return self.response_for(request) is not None

    def response_for(self, request: ChatRequest) -> ChatResponse | None:
        """The recorded response to exactly this request (its item id and its hash), or None.
        clients.RecordingClient answers a resumed item with it."""
        digest = request.sha256()
        with self._lock:
            self._open()
            recording = self._index.get(request.item_id)
        if recording is None or recording.request_sha256 != digest:
            return None
        return recording.response

    def write(self, request: ChatRequest, response: ChatResponse) -> None:
        """Append one recording (item id, request hash, response) and flush."""
        digest = request.sha256()
        as_replayed = ChatRequest(item_id=request.item_id, model=self.model, messages=request.messages, settings=self.settings)
        if as_replayed.sha256() != digest:
            raise CassetteError(
                f"item {request.item_id!r} was asked with model {request.model!r} and settings "
                f"{_settings_json(request.settings)}, but this tape records model {self.model!r} with "
                f"{_settings_json(self.settings)}: the recording would never replay"
            )
        try:
            line = dumps_jsonl_row({"item_id": request.item_id, "request_sha256": digest, "response": response.to_json()})
            recording = _parse_row(line)
        except (TypeError, ValueError) as exc:
            raise CassetteError(f"item {request.item_id!r}: the response cannot be stored as JSON ({exc})") from None
        except _Invalid as exc:
            raise CassetteError(f"item {request.item_id!r}: the response would not read back: {exc}") from None
        with self._lock:
            self._open()
            path = self.root / self._file
            held = self._index.get(request.item_id)
            if held is not None:
                why = "for this very request" if held.request_sha256 == digest else "for another request: record into a new directory"
                raise CassetteError(f"{display_path(path)} already holds item {request.item_id!r} {why}")
            self.root.mkdir(parents=True, exist_ok=True)
            _append(path, (("\n" if self._newline_due else "") + line + "\n").encode("utf-8"))
            self._newline_due = False
            self._index[request.item_id] = recording

    def close(self) -> None:
        """Write manifest.json atomically, merging this feature's tape into any existing one."""
        with self._lock:
            self._open()
            with _manifest_lock(self.root):
                tapes: dict[str, Any] = {}
                description = self.description
                if (self.root / MANIFEST_FILE).exists():
                    manifest = _read_manifest(self.root)
                    self._check_fits(manifest, self._file)  # another writer may have closed meanwhile
                    tapes = {name: _tape_json(tape) for name, tape in manifest.features.items() if name != self.feature}
                    description = description or manifest.description
                self.root.mkdir(parents=True, exist_ok=True)
                if not (self.root / self._file).exists():
                    _append(self.root / self._file, b"")  # a tape with no rows yet still has its file
                tapes[self.feature] = {
                    "file": self._file,
                    "prompt": self.prompt,
                    "model": self.model,
                    "settings": self.settings.to_json(),
                    "suite": self.suite_name,
                    "suite_sha256": self.suite_sha256,
                    "items": len(self._index),
                }
                document = {
                    "schema": CASSETTE_SCHEMA,
                    "arm": self.arm,
                    "synthetic": self.synthetic,
                    "description": description,
                    "features": tapes,
                }
                text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
                write_text_atomic(self.root / MANIFEST_FILE, text)

    # ---- first use --------------------------------------------------------------------------

    def _open(self) -> None:
        """Read what the directory already holds, once (the caller holds self._lock)."""
        if self._opened:
            return
        if self._refusal is not None:
            raise CassetteError(self._refusal)
        file = f"{self.feature}.jsonl"
        try:
            if (self.root / MANIFEST_FILE).exists():
                manifest = _read_manifest(self.root)
                tape = manifest.features.get(self.feature)
                if tape is not None:
                    file = tape.file
                self._check_fits(manifest, file)
            index, newline_due = _read_for_resume(self.root / file)
        except CassetteError as exc:
            self._refusal = str(exc)
            raise
        self._file, self._index, self._newline_due, self._opened = file, index, newline_due, True

    def _check_fits(self, manifest: _Manifest, file: str) -> None:
        """CassetteError unless this writer may add its tape to `manifest` (see WRITING)."""
        where = display_path(self.root / MANIFEST_FILE)
        if manifest.synthetic != self.synthetic:
            raise CassetteError(
                f"{where} holds {_kind(manifest.synthetic)} responses and this writer records {_kind(self.synthetic)} "
                "ones; one set never mixes them: record into a new directory"
            )
        if manifest.arm != self.arm:
            raise CassetteError(f"{where} is the set of arm {manifest.arm!r}, not {self.arm!r}: record into a new directory")
        tape = manifest.features.get(self.feature)
        if tape is not None:
            differences = []
            if tape.prompt != self.prompt:
                differences.append(f"prompt {tape.prompt!r}, not {self.prompt!r}")
            if tape.model != self.model:
                differences.append(f"model {tape.model!r}, not {self.model!r}")
            if _settings_json(tape.settings) != _settings_json(self.settings):
                differences.append(f"settings {_settings_json(tape.settings)}, not {_settings_json(self.settings)}")
            if (tape.suite, tape.suite_sha256) != (self.suite_name, self.suite_sha256):
                differences.append(
                    f"suite {tape.suite!r} {_short(tape.suite_sha256)}, not {self.suite_name!r} {_short(self.suite_sha256)}"
                )
            if differences:
                raise CassetteError(
                    f"{where} records {self.feature!r} with {'; '.join(differences)}: record into a new directory"
                )
        for other in manifest.features.values():
            if other.feature != self.feature and other.file == file:
                raise CassetteError(f"{where} keeps feature {other.feature!r} in {file!r}: record into a new directory")


# ---- strict JSON ------------------------------------------------------------------------------


class _Invalid(Exception):
    """Why a value is refused, without the file and line the caller puts in front."""


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _strict_loads(text: str) -> Any:
    """json.loads, refusing duplicate keys and NaN / Infinity (json.loads accepts both)."""
    return json.loads(text, object_pairs_hook=_object_without_duplicates, parse_constant=_refuse_constant)


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return "array" if isinstance(value, list) else "object"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_name(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _names(keys: Iterable[str]) -> str:
    return ", ".join(repr(key) for key in keys)


def _require_keys(obj: Mapping[str, Any], required: frozenset[str], allowed: frozenset[str], what: str) -> None:
    missing = sorted(required - obj.keys())
    if missing:
        raise _Invalid(f"{what} lacks {_names(missing)}")
    unknown = sorted(obj.keys() - allowed)
    if unknown:
        raise _Invalid(f"{what} has unknown key{'s' if len(unknown) > 1 else ''} {_names(unknown)}")


# ---- the manifest -----------------------------------------------------------------------------

_MANIFEST_REQUIRED = frozenset({"schema", "arm", "synthetic", "features"})
_MANIFEST_KEYS = _MANIFEST_REQUIRED | {"description"}
_TAPE_REQUIRED = frozenset({"file", "prompt", "model", "settings", "items"})
_TAPE_KEYS = _TAPE_REQUIRED | {"suite", "suite_sha256"}
_SETTINGS_KEYS = frozenset({"temperature", "max_tokens", "seed", "extra"})


@dataclass(frozen=True)
class _Manifest:
    arm: str
    synthetic: bool
    description: str
    features: dict[str, FeatureTape]


def _read_manifest(directory: Path) -> _Manifest:
    path = directory / MANIFEST_FILE
    if not path.is_file():
        raise CassetteError(f"{display_path(directory)} holds no {MANIFEST_FILE}: not a cassette set")
    where = display_path(path)
    try:
        obj = _strict_loads(path.read_bytes().decode("utf-8"))
    except UnicodeDecodeError:
        raise CassetteError(f"{where}: not UTF-8 text") from None
    except json.JSONDecodeError as exc:
        raise CassetteError(f"{where}:{exc.lineno}: not JSON ({exc.msg})") from None
    except ValueError as exc:
        raise CassetteError(f"{where}: not JSON ({exc})") from None
    try:
        return _parse_manifest(obj)
    except _Invalid as exc:
        raise CassetteError(f"{where}: {exc}") from None


def _parse_manifest(obj: Any) -> _Manifest:
    if not isinstance(obj, dict):
        raise _Invalid(f"a JSON {_json_type(obj)}, not an object")
    _require_keys(obj, _MANIFEST_REQUIRED, _MANIFEST_KEYS, "the manifest")
    if obj["schema"] != CASSETTE_SCHEMA:
        raise _Invalid(f"schema {obj['schema']!r} is not {CASSETTE_SCHEMA!r}")
    if not _is_name(obj["arm"]):
        raise _Invalid("arm must be a non-empty string")
    if not isinstance(obj["synthetic"], bool):
        raise _Invalid("synthetic must be true or false")
    description = obj.get("description", "")
    if not isinstance(description, str):
        raise _Invalid("description must be a string")
    if not isinstance(obj["features"], dict):
        raise _Invalid(f"features is a JSON {_json_type(obj['features'])}, not an object")
    tapes: dict[str, FeatureTape] = {}
    owners: dict[str, str] = {}
    for name in sorted(obj["features"]):
        tape = _parse_tape(name, obj["features"][name])
        if tape.file in owners:
            raise _Invalid(f"features {owners[tape.file]!r} and {name!r} share the file {tape.file!r}")
        owners[tape.file] = name
        tapes[name] = tape
    return _Manifest(arm=obj["arm"], synthetic=obj["synthetic"], description=description, features=tapes)


def _parse_tape(name: str, obj: Any) -> FeatureTape:
    what = f"features.{name}"
    if name not in FEATURES:
        raise _Invalid(f"{what}: unknown feature (this harness has {_names(FEATURES)})")
    if not isinstance(obj, dict):
        raise _Invalid(f"{what} is a JSON {_json_type(obj)}, not an object")
    _require_keys(obj, _TAPE_REQUIRED, _TAPE_KEYS, what)
    if not _is_tape_file(obj["file"]):
        raise _Invalid(f"{what}.file must name a *.jsonl file directly in the set's directory, not {obj['file']!r}")
    for key in ("prompt", "model"):
        if not _is_name(obj[key]):
            raise _Invalid(f"{what}.{key} must be a non-empty string")
    for key in ("suite", "suite_sha256"):
        if obj.get(key) is not None and not isinstance(obj[key], str):
            raise _Invalid(f"{what}.{key} must be a string or null")
    if not _is_int(obj["items"]) or obj["items"] < 0:
        raise _Invalid(f"{what}.items must be a non-negative integer")
    return FeatureTape(
        feature=name,
        file=obj["file"],
        prompt=obj["prompt"],
        model=obj["model"],
        settings=_parse_settings(obj["settings"], f"{what}.settings"),
        suite=obj.get("suite"),
        suite_sha256=obj.get("suite_sha256"),
        items=obj["items"],
    )


def _parse_settings(obj: Any, what: str) -> ModelSettings:
    if not isinstance(obj, dict):
        raise _Invalid(f"{what} is a JSON {_json_type(obj)}, not an object")
    _require_keys(obj, frozenset(), _SETTINGS_KEYS, what)
    if "temperature" in obj and not (_is_number(obj["temperature"]) and obj["temperature"] >= 0):
        raise _Invalid(f"{what}.temperature must be a number >= 0")
    if "max_tokens" in obj and not (_is_int(obj["max_tokens"]) and obj["max_tokens"] > 0):
        raise _Invalid(f"{what}.max_tokens must be a positive integer")
    if obj.get("seed") is not None and not _is_int(obj["seed"]):
        raise _Invalid(f"{what}.seed must be an integer or null")
    if "extra" in obj and not isinstance(obj["extra"], dict):
        raise _Invalid(f"{what}.extra must be an object")
    settings = ModelSettings.from_json(obj)
    try:
        ChatRequest(item_id="", model="", messages=(), settings=settings).payload()
    except ValueError as exc:
        raise _Invalid(f"{what}.extra: {exc}") from None
    return settings


def _is_tape_file(name: Any) -> bool:
    """A name fingerprint() covers: a *.jsonl file directly in the set's directory."""
    return (
        isinstance(name, str)
        and name != MANIFEST_FILE
        and not any(ch in name for ch in "/\\\x00")
        and Path(name).name == name
        and Path(name).suffix == ".jsonl"
    )


def _settings_json(settings: ModelSettings) -> str:
    return canonical_json(settings.to_json())


def _tape_json(tape: FeatureTape) -> dict[str, Any]:
    return {
        "file": tape.file,
        "prompt": tape.prompt,
        "model": tape.model,
        "settings": tape.settings.to_json(),
        "suite": tape.suite,
        "suite_sha256": tape.suite_sha256,
        "items": tape.items,
    }


def _kind(synthetic: bool) -> str:
    return "synthetic" if synthetic else "recorded"


def _short(digest: str | None) -> str:
    return f"(sha256 {digest[:12]})" if digest else "(no sha256)"


# ---- recordings -------------------------------------------------------------------------------

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_ROW_KEYS = frozenset({"item_id", "request_sha256", "response"})
_RESPONSE_KEYS = frozenset({"content", "reasoning", "finish_reason", "usage", "latency_s"})


def _parse_row(text: str) -> Recording:
    try:
        obj = _strict_loads(text)
    except ValueError as exc:
        raise _Invalid(f"not JSON ({exc.msg if isinstance(exc, json.JSONDecodeError) else exc})") from None
    if not isinstance(obj, dict):
        raise _Invalid(f"a JSON {_json_type(obj)}, not an object")
    _require_keys(obj, _ROW_KEYS, _ROW_KEYS, "the recording")
    item_id, digest, response = obj["item_id"], obj["request_sha256"], obj["response"]
    if not _is_name(item_id):
        raise _Invalid("item_id must be a non-empty string")
    if not isinstance(digest, str) or not _SHA256_HEX.fullmatch(digest):
        raise _Invalid(f"the request_sha256 of {item_id!r} is not 64 lowercase hex digits")
    if not isinstance(response, dict):
        raise _Invalid(f"the response of {item_id!r} is a JSON {_json_type(response)}, not an object")
    _require_keys(response, frozenset(), _RESPONSE_KEYS, f"the response of {item_id!r}")
    for key in ("content", "reasoning", "finish_reason"):
        if response.get(key) is not None and not isinstance(response[key], str):
            raise _Invalid(f"the response of {item_id!r}: {key} must be a string or null")
    usage = response.get("usage")
    if usage is not None and not (isinstance(usage, dict) and all(_is_int(count) for count in usage.values())):
        raise _Invalid(f"the response of {item_id!r}: usage must be an object of integers")
    latency = response.get("latency_s")
    if latency is not None and not (_is_number(latency) and latency >= 0):
        raise _Invalid(f"the response of {item_id!r}: latency_s must be a number >= 0 or null")
    return Recording(item_id=item_id, request_sha256=digest, response=ChatResponse.from_json(response))


def _parse_rows(data: bytes, where: str) -> dict[str, Recording]:
    """Every recording in a tape's bytes, by item id. Lines are split on b"\\n" only, so a
    response holding U+2028 or a form feed stays on its line."""
    recordings: dict[str, Recording] = {}
    first_seen: dict[str, int] = {}
    for number, raw in enumerate(data.split(b"\n"), start=1):
        if not raw.strip():
            continue
        try:
            recording = _parse_row(raw.decode("utf-8"))
        except UnicodeDecodeError:
            raise CassetteError(f"{where}:{number}: not UTF-8 text") from None
        except _Invalid as exc:
            raise CassetteError(f"{where}:{number}: {exc}") from None
        if recording.item_id in recordings:
            raise CassetteError(
                f"{where}:{number}: item {recording.item_id!r} appears twice (first on line {first_seen[recording.item_id]})"
            )
        recordings[recording.item_id] = recording
        first_seen[recording.item_id] = number
    return recordings


def _read_for_resume(path: Path) -> tuple[dict[str, Recording], bool]:
    """A tape's rows for a resuming writer, and whether its last line lacks a newline (see
    WRITING: an incomplete last line is dropped, a complete one is kept)."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return {}, False
    where = display_path(path)
    newline_due = False
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        try:
            _parse_rows(data[cut:], where)
            newline_due = True
        except CassetteError:
            log(f"warning: {where}: dropping an incomplete last line (an interrupted recording); its item will be asked again")
            os.truncate(path, cut)
            data = data[:cut]
    return _parse_rows(data, where), newline_due


def _append(path: Path, data: bytes) -> None:
    """Append unbuffered to an O_APPEND descriptor: a row normally reaches the file in a single
    write call, so an interruption leaves at most an incomplete last line."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
    finally:
        os.close(fd)


_manifest_locks: dict[str, threading.Lock] = {}
_manifest_locks_guard = threading.Lock()


def _manifest_lock(directory: Path) -> threading.Lock:
    """One lock per directory: writers of different features in one process then merge their
    tapes into manifest.json one at a time (close() reads, merges and replaces it)."""
    key = os.path.realpath(directory)
    with _manifest_locks_guard:
        return _manifest_locks.setdefault(key, threading.Lock())
