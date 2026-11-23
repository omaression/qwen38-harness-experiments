"""Model clients: one interface, three implementations.

    ModelClient          the interface: complete(request) -> ChatResponse
    OpenAICompatClient   a live OpenAI-compatible endpoint over urllib (no SDK)
    ReplayClient         answers from a cassette set; refuses a request it was not recorded for
    RecordingClient      wraps a live client and writes every answer into a cassette set
    client_from_env      the live client configured from environment variables

THE LIVE CLIENT. POSTs request.payload() as JSON to <base_url>/chat/completions (base_url
usually ends in /v1, e.g. "http://localhost:8000/v1"; a trailing slash is ignored). Only http and
https URLs are accepted (urllib would also open file:// URLs). Sends "Authorization: Bearer
<key>" when a key is set, and a User-Agent of "llm-eval-harness/<version>". From the reply it
takes choices[0].message.content (null becomes ""), the separate thinking in
message.reasoning_content or message.reasoning, choices[0].finish_reason and `usage` (with
completion_tokens_details.reasoning_tokens flattened to usage["reasoning_tokens"]). It measures
latency with time.monotonic().

Details of the live client:
  - The body is util.canonical_json(request.payload()) in UTF-8, so the sha256 of the bytes on
    the wire is request.sha256(), the hash the cassette records.
  - Thinking: the first non-empty string of reasoning_content and reasoning, else None. A
    content, reasoning or finish_reason that is neither a string nor null is a ClientError: the
    reply is not a chat completion this client understands.
  - usage keeps the integer counts at its top level (nested objects such as
    prompt_tokens_details are left out) plus the flattened reasoning_tokens.
  - latency_s is the wall time of the attempt that answered, not of the retries before it.
  - Redirects are never followed: a 3xx is a ClientError, because following one would carry the
    Authorization header to wherever the redirect points.
  - Proxies come from the usual environment variables (http_proxy, https_proxy, no_proxy), as
    everywhere in urllib.

Retries: on a connection error, a timeout, or HTTP 408, 409, 425, 429, 500, 502, 503 or 504, up
to max_retries more attempts, waiting backoff_s * 2**attempt seconds (at most 30), or the
server's Retry-After seconds when it sends them (at most 60). Any other HTTP status, a reply that
is not JSON, or one without choices[0].message raises ClientError at once. (Retry-After may also
be an HTTP date; the wait is then the time until it. The wait goes through the instance's _sleep
attribute, time.sleep by default, so that tests can record it instead of waiting.)

FAILURES ARE NOT ANSWERS. A request that fails after its retries raises ClientError; it is never
turned into an empty reply. An empty reply would be scored as a wrong answer, and a flaky
endpoint would then look like a regression of the model. The runner stops the evaluation instead
(exit status 1), and the recorder leaves the item unrecorded for the next resume.

SECRETS. The API key comes from an environment variable (client_from_env), never from a command
line flag, a file in the tree or a cassette. It appears in no exception message, log line or
repr; error text the server sends back is passed through redact() before it is shown. redact()
replaces the key as sent and in its JSON-escaped and URL-encoded spellings, before an error text
is shortened, so that a cut can never leave half a key behind. A base URL with credentials in it
(user:password@host) or with a query string is refused: neither is a place for a secret.
"""

from __future__ import annotations

import email.utils
import http.client
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from . import __version__
from .cassettes import CassetteWriter, Recording
from .records import ChatRequest, ChatResponse
from .util import canonical_json, round6

#: Environment variables client_from_env reads by default (the CI live job sets them from secrets).
BASE_URL_ENV = "LLM_EVAL_BASE_URL"
API_KEY_ENV = "LLM_EVAL_API_KEY"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_S = 2.0
#: HTTP statuses worth another attempt.
RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
#: The longest wait before another attempt: the client's own backoff, and a server's Retry-After.
MAX_BACKOFF_S = 30.0
MAX_RETRY_AFTER_S = 60.0
#: The User-Agent header of every request.
USER_AGENT = f"llm-eval-harness/{__version__}"

_EXPECTED_URL = "expected http(s)://host[:port][/path], e.g. http://localhost:8000/v1"
#: How much of an error reply is read, and how much of it a message quotes.
_ERROR_BODY_BYTES = 8192
_DETAIL_CHARS = 300


class ClientError(RuntimeError):
    """A request could not be answered: transport failure after retries, an HTTP error, a reply
    that is not a chat completion, or (replay) a request with no fitting recording."""


class MissingRecording(ClientError):
    """Replay: the cassette has no recording for this item id."""


class CassetteMismatch(ClientError):
    """Replay: the recording for this item answered a different request (its request_sha256
    differs from the rebuilt request's). The task, template, model or settings changed since
    recording: record again."""


@runtime_checkable
class ModelClient(Protocol):
    """Anything that answers a chat request. Implementations must be safe to call from several
    threads at once (the runner calls them from a thread pool)."""

    def complete(self, request: ChatRequest) -> ChatResponse:
        ...


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Turns every redirect into an HTTPError: a chat completion is never redirected, and
    urllib would resend the Authorization header to the new location."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102 (urllib's signature)
        return None


class OpenAICompatClient:
    """A live client for an OpenAI-compatible chat-completions endpoint (see THE LIVE CLIENT)."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_s: float = DEFAULT_BACKOFF_S,
    ) -> None:
        """ValueError for a base_url that is not http(s)://host[:port][/path]."""
        self.base_url = _check_base_url(base_url)
        if not (_is_number(timeout_s) and timeout_s > 0):
            raise ValueError("timeout_s must be a positive number of seconds")
        if not (_is_int(max_retries) and max_retries >= 0):
            raise ValueError("max_retries must be a non-negative integer")
        if not (_is_number(backoff_s) and backoff_s >= 0):
            raise ValueError("backoff_s must be a number of seconds >= 0")
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError("the API key must be a string or None")
        key = (api_key or "").strip() or None
        if key is not None and not (key.isascii() and key.isprintable() and not any(ch.isspace() for ch in key)):
            raise ValueError("the API key must be printable ASCII without spaces or line breaks")
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.backoff_s = float(backoff_s)
        self._url = self.base_url + "/chat/completions"
        self._api_key = key
        self._secrets = _spellings(key)
        self._headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT}
        if key is not None:
            self._headers["Authorization"] = f"Bearer {key}"
        self._opener = urllib.request.build_opener(_RefuseRedirects())
        self._sleep: Callable[[float], Any] = time.sleep

    def complete(self, request: ChatRequest) -> ChatResponse:
        """One chat completion, with the retries above. Raises ClientError."""
        body = canonical_json(request.payload()).encode("utf-8")
        for attempt in range(self.max_retries + 1):
            final = attempt == self.max_retries
            started = time.monotonic()
            try:
                post = urllib.request.Request(self._url, data=body, headers=self._headers, method="POST")
                with self._opener.open(post, timeout=self.timeout_s) as reply:
                    status, raw = reply.status, reply.read()
            except urllib.error.HTTPError as exc:
                retry_after = exc.headers.get("Retry-After") if exc.headers is not None else None
                detail = self._error_detail(exc)
                if exc.code in RETRY_STATUSES and not final:
                    self._sleep(self._wait(attempt, retry_after))
                    continue
                if 300 <= exc.code < 400:
                    detail = f"the endpoint redirects, and redirects are not followed: check the base URL ({detail})"
                raise ClientError(self._failure(f"HTTP {exc.code}", attempt, detail)) from None
            except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
                if not final:
                    self._sleep(self._backoff(attempt))
                    continue
                raise ClientError(self._failure("no answer", attempt, _reason(exc))) from None
            return self._parse(raw, status, time.monotonic() - started)
        raise AssertionError("unreachable: the last attempt returns or raises")

    def redact(self, text: str) -> str:
        """`text` with the API key, if any, replaced by "***"."""
        for secret in self._secrets:
            text = text.replace(secret, "***")
        return text

    def __repr__(self) -> str:
        """Shows the base URL, never the key."""
        key = "'***'" if self._api_key else "None"
        return (
            f"{type(self).__name__}(base_url={self.redact(self.base_url)!r}, api_key={key}, "
            f"timeout_s={self.timeout_s:g}, max_retries={self.max_retries}, backoff_s={self.backoff_s:g})"
        )

    # ---- helpers ------------------------------------------------------------------------------

    def _backoff(self, attempt: int) -> float:
        return min(self.backoff_s * 2**attempt, MAX_BACKOFF_S)

    def _wait(self, attempt: int, retry_after: str | None) -> float:
        seconds = _retry_after_seconds(retry_after)
        return self._backoff(attempt) if seconds is None else min(seconds, MAX_RETRY_AFTER_S)

    def _failure(self, what: str, attempt: int, detail: str) -> str:
        tries = f" after {attempt + 1} attempts" if attempt else ""
        return self.redact(f"POST {self._url}: {what}{tries}: {detail}")

    def _describe(self, raw: bytes) -> str:
        """A short, one-line, redacted account of a reply body for an error message: the
        error message of an OpenAI-style error object when there is one, else the text."""
        text = raw.decode("utf-8", errors="replace")
        try:
            obj = json.loads(text)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            error = obj.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                text = error["message"]
            elif isinstance(error, str):
                text = error
            elif isinstance(obj.get("message"), str):
                text = obj["message"]
            elif isinstance(obj.get("detail"), str):
                text = obj["detail"]
        text = " ".join(self.redact(text).split())  # redact before shortening: a cut never halves the key
        if not text:
            return "(empty body)"
        return text if len(text) <= _DETAIL_CHARS else text[: _DETAIL_CHARS - 3] + "..."

    def _error_detail(self, exc: urllib.error.HTTPError) -> str:
        try:
            raw = exc.read(_ERROR_BODY_BYTES) or b""
        except Exception:  # the body is a courtesy for the message; failing to read it changes nothing
            raw = b""
        finally:
            try:
                exc.close()
            except Exception:
                pass
        if len(raw) >= _ERROR_BODY_BYTES and self._secrets:
            # The read cut the body, possibly inside an echoed key, which redact() would then miss:
            # drop as many bytes as the longest spelling of the key (all are ASCII).
            raw = raw[: -len(self._secrets[0])]
        return self._describe(raw)

    def _parse(self, raw: bytes, status: int, latency_s: float) -> ChatResponse:
        def bad(problem: str) -> ClientError:
            return ClientError(self.redact(f"POST {self._url}: HTTP {status}, but {problem}: {self._describe(raw)}"))

        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:  # includes UnicodeDecodeError and JSONDecodeError
            raise bad("the reply is not JSON") from None
        if not isinstance(data, dict):
            raise bad("the reply is not a JSON object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise bad("the reply has no choices[0]")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise bad("the reply has no choices[0].message")
        content = message.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            raise bad("choices[0].message.content is not a string")
        reasoning: str | None = None
        for key in ("reasoning_content", "reasoning"):
            value = message.get(key)
            if value is None:
                continue
            if not isinstance(value, str):
                raise bad(f"choices[0].message.{key} is not a string")
            if value and reasoning is None:
                reasoning = value
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise bad("choices[0].finish_reason is not a string")
        return ChatResponse(
            content=content,
            reasoning=reasoning,
            finish_reason=finish_reason,
            usage=_usage(data.get("usage")),
            latency_s=round6(max(latency_s, 0.0)),
        )


class ReplayClient:
    """Answers from recordings (cassettes.read_recordings): the CI path, free and deterministic.

    complete() looks the request up by item id (MissingRecording when absent) and, unless
    verify_hash is False, compares the recording's request_sha256 with request.sha256()
    (CassetteMismatch on a difference, naming the item and both hash prefixes). It returns the
    recorded ChatResponse unchanged, latency included."""

    def __init__(self, recordings: Mapping[str, Recording], *, verify_hash: bool = True) -> None:
        self._recordings = dict(recordings)
        self._verify_hash = verify_hash

    def complete(self, request: ChatRequest) -> ChatResponse:
        recording = self._recordings.get(request.item_id)
        if recording is None:
            raise MissingRecording(f"no recording for item {request.item_id!r}")
        if self._verify_hash:
            digest = request.sha256()
            if recording.request_sha256 != digest:
                raise CassetteMismatch(
                    f"item {request.item_id!r}: the recording answered request {recording.request_sha256[:12]}, but "
                    f"the rebuilt request is {digest[:12]}: the task, template, model or settings changed since "
                    "recording; record again"
                )
        return recording.response

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self._recordings)} recordings, verify_hash={self._verify_hash})"


class RecordingClient:
    """Wraps a live client; every successful answer is written to `writer` before it is returned.
    A request the writer already has (same item id and request hash) is not sent again: its
    recorded answer is returned, so an interrupted recording resumes where it stopped.

    A ClientError of the inner client propagates and nothing is written. The writer's
    CassetteError (a conflicting resume, an item recorded for another request) propagates too."""

    def __init__(self, inner: ModelClient, writer: CassetteWriter) -> None:
        self._inner = inner
        self._writer = writer

    def complete(self, request: ChatRequest) -> ChatResponse:
        recorded = self._writer.response_for(request)
        if recorded is not None:
            return recorded
        response = self._inner.complete(request)
        self._writer.write(request, response)
        return response

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._inner!r} -> {self._writer.root.name}/{self._writer.feature})"


def client_from_env(
    base_url: str | None = None,
    *,
    base_url_env: str = BASE_URL_ENV,
    api_key_env: str = API_KEY_ENV,
    environ: Mapping[str, str] | None = None,
    **options: float | int,
) -> OpenAICompatClient:
    """An OpenAICompatClient for `base_url`, or the URL in $<base_url_env>, with the key from
    $<api_key_env> (none when unset or empty). ClientError when there is no URL at all.
    `options` are passed on (timeout_s, max_retries, backoff_s); `environ` defaults to os.environ.
    A URL that is there but malformed raises the constructor's ValueError, which never quotes it."""
    env = os.environ if environ is None else environ
    url = (base_url or "").strip() or (env.get(base_url_env) or "").strip()
    if not url:
        raise ClientError(f"no endpoint: pass a base URL or set ${base_url_env}")
    key = (env.get(api_key_env) or "").strip() or None
    return OpenAICompatClient(url, key, **options)


# ---- helpers ------------------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_base_url(base_url: Any) -> str:
    """The base URL without trailing slashes, or ValueError. The messages never quote the URL:
    a misconfigured one may hold a secret."""
    if not isinstance(base_url, str):
        raise ValueError(f"the base URL must be a string: {_EXPECTED_URL}")
    text = base_url.strip()
    if any(ord(ch) <= 0x20 or ord(ch) == 0x7F for ch in text):
        raise ValueError(f"the base URL holds a space or a control character: {_EXPECTED_URL}")
    if not text.isascii():  # http.client would fail on it at the first request, outside ClientError
        raise ValueError("the base URL must be ASCII: write an international host name in its xn-- form and percent-encode the path")
    try:
        parts = urllib.parse.urlsplit(text)
        _ = parts.port  # ValueError unless the port is a number in 0..65535
    except ValueError:
        raise ValueError(f"the base URL is malformed: {_EXPECTED_URL}") from None
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"the base URL must be an http:// or https:// URL: {_EXPECTED_URL}")
    if "@" in parts.netloc:
        raise ValueError("the base URL must not carry credentials (user:password@host): the key comes from the environment")
    if not parts.hostname:
        raise ValueError(f"the base URL names no host: {_EXPECTED_URL}")
    if parts.query or parts.fragment or "?" in text or "#" in text:
        raise ValueError(f"the base URL must not have a query or a fragment: {_EXPECTED_URL}")
    return text.rstrip("/")


def _spellings(key: str | None) -> tuple[str, ...]:
    """The ways a server may echo the key back: as sent, JSON-escaped, URL-encoded. Longest
    first, so that no replacement leaves part of a longer spelling behind."""
    if not key:
        return ()
    forms = {key, json.dumps(key)[1:-1], key.replace("/", "\\/"), urllib.parse.quote(key, safe=""), urllib.parse.quote_plus(key)}
    return tuple(sorted((form for form in forms if form), key=lambda form: (-len(form), form)))


def _retry_after_seconds(value: str | None) -> float | None:
    """Retry-After as seconds: a number of seconds, or an HTTP date (the time until it, 0 when
    past). None when absent or unreadable."""
    if value is None or not value.strip():
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value.strip())
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if when is None:  # older Pythons returned None for some unreadable dates
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _reason(exc: BaseException) -> str:
    """What went wrong with a transport, in one line."""
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        text = str(reason)
        if isinstance(reason, BaseException):
            return f"{type(reason).__name__}: {text}" if text else type(reason).__name__
        return text or type(exc).__name__
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _usage(value: Any) -> dict[str, int]:
    """The integer counts of a reply's usage object, with completion_tokens_details.reasoning_tokens
    flattened to reasoning_tokens."""
    if not isinstance(value, dict):
        return {}
    usage = {str(key): count for key, count in value.items() if _is_int(count)}
    details = value.get("completion_tokens_details")
    if isinstance(details, dict) and _is_int(details.get("reasoning_tokens")):
        usage["reasoning_tokens"] = details["reasoning_tokens"]
    return usage
