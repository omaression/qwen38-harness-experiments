"""The model clients (clients.py). The live client is tested against a stub server on 127.0.0.1
in a daemon thread: no test uses the external network, and the injected sleep means no test
waits for a backoff."""

from __future__ import annotations

import email.utils
import http.server
import json
import socket
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pytest

from llm_eval_harness import __version__
from llm_eval_harness.cassettes import CassetteError, CassetteWriter, Recording, load_cassettes, read_recordings
from llm_eval_harness.clients import (
    API_KEY_ENV,
    BASE_URL_ENV,
    MAX_BACKOFF_S,
    MAX_RETRY_AFTER_S,
    RETRY_STATUSES,
    CassetteMismatch,
    ClientError,
    MissingRecording,
    ModelClient,
    OpenAICompatClient,
    RecordingClient,
    ReplayClient,
    client_from_env,
)
from llm_eval_harness.records import ChatRequest, ChatResponse, Message, ModelSettings
from llm_eval_harness.util import canonical_json, sha256_bytes

KEY = "test-key-123"  # obviously fake

REQUEST = ChatRequest(
    item_id="bugfix:t-clamp",
    model="demo-model",
    messages=(Message("system", "You fix bugs."), Message("user", "Fix clamp — é")),
    settings=ModelSettings(extra={"chat_template_kwargs": {"enable_thinking": True}}),
)


# ---- the stub server --------------------------------------------------------------------------------


@dataclass
class Received:
    method: str
    path: str
    headers: dict[str, str]  # lower-cased names
    body: bytes


@dataclass
class Reply:
    """One scripted answer. `body`: a dict or list (sent as JSON), str or bytes (as is), or a
    callable of the Received request returning one of those."""

    status: int = 200
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    delay_s: float = 0.0
    drop: bool = False  # close the connection without answering


class _QuietServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass  # a client that timed out and left is what some tests want


class StubServer:
    """An OpenAI-compatible endpoint that answers from a script of Replies, and records every
    request it receives. With the script exhausted it answers HTTP 500."""

    def __init__(self) -> None:
        self.replies: list[Reply] = []
        self.received: list[Received] = []
        self._lock = threading.Lock()
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                stub._serve(self)

            do_GET = do_POST

            def log_message(self, *args: Any) -> None:
                pass

        self._httpd = _QuietServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def queue(self, *replies: Reply) -> None:
        with self._lock:
            self.replies.extend(replies)

    def _serve(self, handler: http.server.BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length") or 0)
        got = Received(handler.command, handler.path, {k.lower(): v for k, v in handler.headers.items()}, handler.rfile.read(length))
        with self._lock:
            self.received.append(got)
            reply = self.replies.pop(0) if self.replies else Reply(500, {"error": {"message": "the stub has no reply queued"}})
        if reply.delay_s:
            time.sleep(reply.delay_s)
        if reply.drop:
            handler.close_connection = True
            return
        body = reply.body(got) if callable(reply.body) else reply.body
        if isinstance(body, (dict, list)):
            data, content_type = json.dumps(body).encode("utf-8"), "application/json"
        elif isinstance(body, str):
            data, content_type = body.encode("utf-8"), "text/plain; charset=utf-8"
        else:
            data, content_type = body or b"", "application/octet-stream"
        handler.send_response(reply.status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(data)))
        for name, value in reply.headers.items():
            handler.send_header(name, value)
        handler.end_headers()
        handler.wfile.write(data)

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture(autouse=True)
def _no_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    """urllib honours proxy variables; a proxy set on the test machine must not see the stub."""
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture
def server() -> Iterator[StubServer]:
    stub = StubServer()
    yield stub
    stub.close()


def completion(content: Any = '{"explanation": "e", "patched_code": "c"}', *, finish_reason: Any = "stop",
               usage: Any = None, **message: Any) -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content, **message}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10} if usage is None else usage,
    }


def client_for(server: StubServer, key: str | None = KEY, sleeps: list[float] | None = None, **options: Any) -> OpenAICompatClient:
    options = {"timeout_s": 5.0, "max_retries": 3, "backoff_s": 0.5, **options}
    client = OpenAICompatClient(server.base_url + "/", key, **options)  # the trailing slash is ignored
    client._sleep = sleeps.append if sleeps is not None else (lambda seconds: None)
    return client


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# ---- a successful call --------------------------------------------------------------------------------


def test_a_completion_posts_the_canonical_payload_and_parses_the_reply(server: StubServer) -> None:
    server.queue(Reply(body=completion()))
    got = client_for(server).complete(REQUEST)
    assert got.content == '{"explanation": "e", "patched_code": "c"}'
    assert (got.reasoning, got.finish_reason) == (None, "stop")
    assert got.usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    assert isinstance(got.latency_s, float) and 0.0 <= got.latency_s < 5.0

    [sent] = server.received
    assert (sent.method, sent.path) == ("POST", "/v1/chat/completions")
    assert sent.headers["content-type"] == "application/json"
    assert sent.headers["user-agent"] == f"llm-eval-harness/{__version__}"
    assert sent.headers["authorization"] == f"Bearer {KEY}"
    assert json.loads(sent.body) == REQUEST.payload() and "item_id" not in json.loads(sent.body)
    assert sent.body == canonical_json(REQUEST.payload()).encode("utf-8")
    assert sha256_bytes(sent.body) == REQUEST.sha256()  # the cassette's hash is the hash of the bytes sent


def test_null_content_becomes_an_empty_string(server: StubServer) -> None:
    server.queue(Reply(body=completion(None, finish_reason="length")))
    got = client_for(server).complete(REQUEST)
    assert (got.content, got.finish_reason) == ("", "length")


def test_a_missing_finish_reason_is_none(server: StubServer) -> None:
    body = completion()
    del body["choices"][0]["finish_reason"]
    server.queue(Reply(body=body))
    assert client_for(server).complete(REQUEST).finish_reason is None


@pytest.mark.parametrize(
    "message, thinking",
    [
        ({"reasoning_content": "lo was used twice"}, "lo was used twice"),
        ({"reasoning": "the upper bound is hi"}, "the upper bound is hi"),
        ({"reasoning_content": "first", "reasoning": "second"}, "first"),
        ({"reasoning_content": "", "reasoning": "second"}, "second"),
        ({"reasoning_content": None, "reasoning": None}, None),
        ({}, None),
    ],
)
def test_separately_returned_thinking(server: StubServer, message: dict[str, Any], thinking: str | None) -> None:
    server.queue(Reply(body=completion("<answer>", **message)))
    got = client_for(server).complete(REQUEST)
    assert (got.content, got.reasoning) == ("<answer>", thinking)


def test_thinking_inside_the_content_stays_there(server: StubServer) -> None:
    server.queue(Reply(body=completion("<think>hmm</think>{}")))
    got = client_for(server).complete(REQUEST)
    assert (got.content, got.reasoning) == ("<think>hmm</think>{}", None)


@pytest.mark.parametrize(
    "usage, flat",
    [
        ({"prompt_tokens": 7, "completion_tokens": 30, "total_tokens": 37,
          "completion_tokens_details": {"reasoning_tokens": 21, "audio_tokens": 0},
          "prompt_tokens_details": {"cached_tokens": 4}, "cost": 0.5, "cached": True},
         {"prompt_tokens": 7, "completion_tokens": 30, "total_tokens": 37, "reasoning_tokens": 21}),
        ({"completion_tokens": 3, "completion_tokens_details": None}, {"completion_tokens": 3}),
        ({"completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 1.0}}, {"completion_tokens": 3}),
        ([], {}),
        ("none", {}),
    ],
)
def test_usage_is_flattened_to_integer_counts(server: StubServer, usage: Any, flat: dict[str, int]) -> None:
    server.queue(Reply(body=completion(usage=usage)))
    assert client_for(server).complete(REQUEST).usage == flat


def test_a_reply_without_usage(server: StubServer) -> None:
    body = completion()
    del body["usage"]
    server.queue(Reply(body=body))
    assert client_for(server).complete(REQUEST).usage == {}


@pytest.mark.parametrize("key, header", [(KEY, f"Bearer {KEY}"), (f"  {KEY}\n", f"Bearer {KEY}"), (None, None), ("", None), ("   ", None)])
def test_the_authorization_header_is_sent_only_with_a_key(server: StubServer, key: str | None, header: str | None) -> None:
    server.queue(Reply(body=completion()))
    client_for(server, key=key).complete(REQUEST)
    assert server.received[0].headers.get("authorization") == header


def test_one_client_serves_many_threads(server: StubServer) -> None:
    server.queue(*[Reply(body=completion(f"answer {i}")) for i in range(8)])
    client = client_for(server)
    answers: list[str] = []
    lock = threading.Lock()

    def ask() -> None:
        content = client.complete(REQUEST).content
        with lock:
            answers.append(content)

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(answers) == sorted(f"answer {i}" for i in range(8)) and len(server.received) == 8


# ---- retries ----------------------------------------------------------------------------------------


def test_a_503_is_retried_after_the_backoff(server: StubServer) -> None:
    server.queue(Reply(503, {"error": {"message": "loading"}}), Reply(body=completion("finally")))
    sleeps: list[float] = []
    assert client_for(server, sleeps=sleeps).complete(REQUEST).content == "finally"
    assert sleeps == [0.5]
    assert [r.body for r in server.received] == [server.received[0].body] * 2  # the same request again


@pytest.mark.parametrize("status", sorted(RETRY_STATUSES))
def test_every_retry_status_is_retried(server: StubServer, status: int) -> None:
    server.queue(Reply(status, "busy"), Reply(body=completion()))
    sleeps: list[float] = []
    client_for(server, sleeps=sleeps).complete(REQUEST)
    assert len(sleeps) == 1 and len(server.received) == 2


def test_the_backoff_doubles_up_to_its_cap(server: StubServer) -> None:
    server.queue(*[Reply(502, "bad gateway")] * 4, Reply(body=completion()))
    sleeps: list[float] = []
    client_for(server, sleeps=sleeps, backoff_s=12.0, max_retries=4).complete(REQUEST)
    assert sleeps == [12.0, 24.0, MAX_BACKOFF_S, MAX_BACKOFF_S]


@pytest.mark.parametrize(
    "retry_after, wait",
    [
        ("7", 7.0),
        (" 3 ", 3.0),
        ("0", 0.0),
        ("1.5", 1.5),
        ("120", MAX_RETRY_AFTER_S),
        (email.utils.formatdate(time.time() + 3600, usegmt=True), MAX_RETRY_AFTER_S),
        (email.utils.formatdate(time.time() - 3600, usegmt=True), 0.0),
        ("soon", 0.5),  # unreadable: the backoff
        ("-5", 0.5),
        ("nan", 0.5),
    ],
)
def test_retry_after_is_honoured(server: StubServer, retry_after: str, wait: float) -> None:
    server.queue(Reply(429, {"error": {"message": "slow down"}}, headers={"Retry-After": retry_after}), Reply(body=completion()))
    sleeps: list[float] = []
    client_for(server, sleeps=sleeps).complete(REQUEST)
    assert sleeps == [wait]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_other_statuses_are_not_retried(server: StubServer, status: int) -> None:
    server.queue(Reply(status, {"error": {"message": f"refused with {status}", "type": "invalid_request_error"}}), Reply(body=completion()))
    sleeps: list[float] = []
    with pytest.raises(ClientError) as failed:
        client_for(server, sleeps=sleeps).complete(REQUEST)
    assert f"HTTP {status}" in str(failed.value) and f"refused with {status}" in str(failed.value)
    assert "attempts" not in str(failed.value)
    assert sleeps == [] and len(server.received) == 1


def test_retries_end_in_a_client_error(server: StubServer) -> None:
    server.queue(*[Reply(500, "internal error")] * 4)
    sleeps: list[float] = []
    with pytest.raises(ClientError, match="HTTP 500 after 4 attempts: internal error"):
        client_for(server, sleeps=sleeps).complete(REQUEST)
    assert sleeps == [0.5, 1.0, 2.0] and len(server.received) == 4


def test_no_retries_when_max_retries_is_zero(server: StubServer) -> None:
    server.queue(Reply(503, "busy"), Reply(body=completion()))
    sleeps: list[float] = []
    with pytest.raises(ClientError, match="HTTP 503"):
        client_for(server, sleeps=sleeps, max_retries=0).complete(REQUEST)
    assert sleeps == [] and len(server.received) == 1


def test_connection_refused_raises_after_the_retries() -> None:
    client = OpenAICompatClient(f"http://127.0.0.1:{free_port()}/v1", KEY, max_retries=2, backoff_s=0.25, timeout_s=5.0)
    sleeps: list[float] = []
    client._sleep = sleeps.append
    with pytest.raises(ClientError, match="no answer after 3 attempts") as failed:
        client.complete(REQUEST)
    assert "refused" in str(failed.value).lower()
    assert sleeps == [0.25, 0.5]


def test_a_dropped_connection_is_retried(server: StubServer) -> None:
    server.queue(Reply(drop=True), Reply(body=completion("second time")))
    sleeps: list[float] = []
    assert client_for(server, sleeps=sleeps).complete(REQUEST).content == "second time"
    assert sleeps == [0.5]


def test_a_timeout_is_retried_and_then_raises(server: StubServer) -> None:
    server.queue(Reply(body=completion(), delay_s=1.0), Reply(body=completion(), delay_s=1.0))
    sleeps: list[float] = []
    with pytest.raises(ClientError, match="no answer after 2 attempts") as failed:
        client_for(server, sleeps=sleeps, timeout_s=0.2, max_retries=1).complete(REQUEST)
    assert "timed out" in str(failed.value).lower()
    assert sleeps == [0.5]


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirects_are_never_followed(server: StubServer, status: int) -> None:
    elsewhere = StubServer()
    try:
        elsewhere.queue(Reply(body=completion("from elsewhere")))
        server.queue(Reply(status, "moved", headers={"Location": elsewhere.base_url + "/chat/completions"}))
        sleeps: list[float] = []
        with pytest.raises(ClientError, match=f"HTTP {status}.*redirects are not followed"):
            client_for(server, sleeps=sleeps).complete(REQUEST)
        assert elsewhere.received == [] and sleeps == []  # the key never went anywhere else
    finally:
        elsewhere.close()


# ---- replies that are not chat completions ----------------------------------------------------------


@pytest.mark.parametrize(
    "body, problem",
    [
        ("<html>gateway</html>", "not JSON"),
        (b"\xff\xfe\x00", "not JSON"),
        ("", "not JSON"),
        ([1, 2], "not a JSON object"),
        ({"object": "error", "error": {"message": "model not loaded"}}, "no choices[0]"),
        ({"choices": []}, "no choices[0]"),
        ({"choices": ["text"]}, "no choices[0]"),
        ({"choices": [{"finish_reason": "stop"}]}, "no choices[0].message"),
        ({"choices": [{"message": "text"}]}, "no choices[0].message"),
        (completion(["part 1", "part 2"]), "content is not a string"),
        (completion("x", reasoning_content={"text": "thinking"}), "reasoning_content is not a string"),
        (completion("x", reasoning=7), "reasoning is not a string"),
        (completion("x", finish_reason=1), "finish_reason is not a string"),
    ],
)
def test_a_reply_that_is_not_a_chat_completion_raises_at_once(server: StubServer, body: Any, problem: str) -> None:
    server.queue(Reply(200, body), Reply(body=completion()))
    sleeps: list[float] = []
    with pytest.raises(ClientError) as failed:
        client_for(server, sleeps=sleeps).complete(REQUEST)
    assert problem in str(failed.value) and "HTTP 200" in str(failed.value)
    assert sleeps == [] and len(server.received) == 1


def test_a_failure_is_never_an_empty_reply(server: StubServer) -> None:
    server.queue(*[Reply(503, "down")] * 4)
    with pytest.raises(ClientError):
        client_for(server).complete(REQUEST)  # raises; nothing is returned that could be scored


def test_the_client_errors_form_one_family() -> None:
    assert issubclass(MissingRecording, ClientError) and issubclass(CassetteMismatch, ClientError)
    assert issubclass(ClientError, RuntimeError)


# ---- base URLs --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://127.0.0.1/v1",
        "localhost:8000/v1",
        "127.0.0.1:8000",
        "//localhost:8000/v1",
        "http://",
        "http:///v1",
        "https://:8000/v1",
        "http://user:pw@localhost:8000/v1",
        "http://localhost:8000/v1?api-key=x",
        "http://localhost:8000/v1?",
        "http://localhost:8000/v1#part",
        "http://localhost:99999/v1",
        "http://localhost:port/v1",
        "http://[::1/v1",
        "http://local host/v1",
        "http://local\nhost/v1",
        "http://localhost:8000/v\u00e9/",
        "",
        "   ",
    ],
)
def test_a_base_url_that_is_not_http_host_path_is_refused(url: str) -> None:
    with pytest.raises(ValueError) as refused:
        OpenAICompatClient(url)
    if "pw" in url or "api-key" in url:
        assert "pw" not in str(refused.value) and "api-key=x" not in str(refused.value)


@pytest.mark.parametrize(
    "url, base",
    [
        ("http://localhost:8000/v1", "http://localhost:8000/v1"),
        ("http://localhost:8000/v1///", "http://localhost:8000/v1"),
        ("  http://127.0.0.1:8000/v1  ", "http://127.0.0.1:8000/v1"),
        ("HTTPS://inference.example.test", "HTTPS://inference.example.test"),
        ("http://[::1]:8000/v1", "http://[::1]:8000/v1"),
    ],
)
def test_accepted_base_urls(url: str, base: str) -> None:
    assert OpenAICompatClient(url).base_url == base


@pytest.mark.parametrize(
    "options",
    [{"timeout_s": 0}, {"timeout_s": float("inf")}, {"max_retries": -1}, {"max_retries": 1.5}, {"max_retries": True}, {"backoff_s": -1.0}],
)
def test_bad_options_are_refused(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        OpenAICompatClient("http://localhost:8000/v1", **options)


@pytest.mark.parametrize("key", [f"{KEY}\r\nX-Injected: 1", "test key", "clé-123", 123])
def test_a_key_that_cannot_be_a_header_is_refused_without_quoting_it(key: Any) -> None:
    with pytest.raises(ValueError) as refused:
        OpenAICompatClient("http://localhost:8000/v1", key)
    assert str(key) not in str(refused.value)


# ---- the key never shows ----------------------------------------------------------------------------


SLASHED_KEY = "test/key+12"  # a key with characters that JSON and URLs escape


def _echoes(received: Received) -> dict[str, Any]:
    token = received.headers.get("authorization", "")
    key = token.removeprefix("Bearer ")
    return {"error": {"message": f"invalid token {token}; key={key}; json={json.dumps(key)}; escaped={key.replace('/', chr(92) + '/')}; "
                                 f"quoted={key.replace('/', '%2F').replace('+', '%2B')}"}}


def _assert_hidden(text: str, key: str) -> None:
    assert key not in text
    assert key.replace("/", "\\/") not in text and key.replace("/", "%2F").replace("+", "%2B") not in text
    assert key[: len(key) // 2] not in text.replace("***", "")  # not even half of it


@pytest.mark.parametrize("key", [KEY, SLASHED_KEY])
def test_the_key_never_appears_in_errors_or_repr(server: StubServer, key: str) -> None:
    server.queue(Reply(401, _echoes), Reply(200, lambda rx: f"<html>{rx.headers['authorization']}</html>"),
                 *[Reply(503, lambda rx: f"overloaded for {rx.headers['authorization']}")] * 2)
    client = client_for(server, key=key, max_retries=1)
    for _ in range(3):
        with pytest.raises(ClientError) as failed:
            client.complete(REQUEST)
        message = str(failed.value)
        assert "***" in message
        _assert_hidden(message, key)
        _assert_hidden(repr(failed.value), key)
        _assert_hidden("".join(traceback.format_exception(failed.type, failed.value, failed.tb)), key)
        assert failed.value.__cause__ is None and failed.value.__suppress_context__
    for shown in (repr(client), str(client)):
        _assert_hidden(shown, key)
        assert server.base_url in shown and "***" in shown


def test_a_long_echo_is_shortened_after_redaction(server: StubServer) -> None:
    padding = "x" * 284  # unredacted, the cut at 300 characters would fall inside the key
    server.queue(Reply(400, lambda rx: {"error": {"message": padding + rx.headers["authorization"] + "y" * 400}}))
    with pytest.raises(ClientError) as failed:
        client_for(server).complete(REQUEST)
    _assert_hidden(str(failed.value), KEY)
    assert str(failed.value).endswith("...")


def test_an_error_body_cut_inside_the_key_shows_none_of_it(server: StubServer) -> None:
    # The client reads 8 KiB of an error body; here the cut falls inside the echoed key, and
    # collapsing the whitespace would bring what is left of it into the message.
    body = "x" + " " * (8192 - 1 - 6) + KEY + " and more"
    server.queue(Reply(400, body))
    with pytest.raises(ClientError) as failed:
        client_for(server).complete(REQUEST)
    _assert_hidden(str(failed.value), KEY)


def test_redact() -> None:
    client = OpenAICompatClient("http://localhost:8000/v1", SLASHED_KEY)
    assert client.redact(f"a {SLASHED_KEY} b test\\/key+12 c test%2Fkey%2B12") == "a *** b *** c ***"
    keyless = OpenAICompatClient("http://localhost:8000/v1")
    assert keyless.redact("nothing to hide") == "nothing to hide"
    assert "api_key=None" in repr(keyless)


# ---- client_from_env ----------------------------------------------------------------------------


def test_client_from_env_reads_both_variables(server: StubServer) -> None:
    client = client_from_env(environ={BASE_URL_ENV: server.base_url, API_KEY_ENV: KEY}, timeout_s=3.0, max_retries=0, backoff_s=0.0)
    assert (client.base_url, client.timeout_s, client.max_retries, client.backoff_s) == (server.base_url, 3.0, 0, 0.0)
    server.queue(Reply(body=completion()))
    client.complete(REQUEST)
    assert server.received[0].headers["authorization"] == f"Bearer {KEY}"


def test_client_from_env_prefers_the_given_url_and_takes_other_names(server: StubServer) -> None:
    environ = {"MY_URL": "http://localhost:1/v1", "MY_KEY": "", BASE_URL_ENV: "http://localhost:2/v1"}
    client = client_from_env(server.base_url, base_url_env="MY_URL", api_key_env="MY_KEY", environ=environ)
    assert client.base_url == server.base_url
    server.queue(Reply(body=completion()))
    client.complete(REQUEST)
    assert "authorization" not in server.received[0].headers  # an empty variable is no key
    assert client_from_env(environ=environ, base_url_env="MY_URL").base_url == "http://localhost:1/v1"


def test_client_from_env_defaults_to_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BASE_URL_ENV, "http://localhost:8000/v1")
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    assert client_from_env().base_url == "http://localhost:8000/v1"


@pytest.mark.parametrize("environ", [{}, {BASE_URL_ENV: ""}, {BASE_URL_ENV: "  "}, {API_KEY_ENV: KEY}])
def test_client_from_env_without_a_url_raises(environ: dict[str, str]) -> None:
    with pytest.raises(ClientError, match=f"no endpoint.*{BASE_URL_ENV}") as failed:
        client_from_env(environ=environ)
    assert KEY not in str(failed.value)


def test_client_from_env_with_a_malformed_url_raises_the_constructors_error() -> None:
    with pytest.raises(ValueError, match="http:// or https://") as refused:
        client_from_env(environ={BASE_URL_ENV: KEY})  # the key put into the wrong variable
    assert KEY not in str(refused.value)


# ---- replay -------------------------------------------------------------------------------------


def _recorded(request: ChatRequest, content: str = "recorded answer", digest: str | None = None) -> Recording:
    answer = ChatResponse(content=content, reasoning="thought", finish_reason="stop", usage={"completion_tokens": 9}, latency_s=0.84)
    return Recording(item_id=request.item_id, request_sha256=digest or request.sha256(), response=answer)


def test_replay_returns_the_recorded_response_unchanged() -> None:
    recording = _recorded(REQUEST)
    replay = ReplayClient({REQUEST.item_id: recording})
    assert replay.complete(REQUEST) is recording.response
    assert replay.complete(REQUEST).latency_s == 0.84


def test_replay_refuses_an_item_it_has_no_recording_for() -> None:
    with pytest.raises(MissingRecording, match="bugfix:t-clamp"):
        ReplayClient({}).complete(REQUEST)


def test_replay_refuses_a_recording_made_for_another_request() -> None:
    other = ChatRequest(item_id=REQUEST.item_id, model="demo-model", messages=(Message("user", "an older prompt"),))
    recording = _recorded(other)
    with pytest.raises(CassetteMismatch) as refused:
        ReplayClient({REQUEST.item_id: recording}).complete(REQUEST)
    text = str(refused.value)
    assert "bugfix:t-clamp" in text and other.sha256()[:12] in text and REQUEST.sha256()[:12] in text
    assert "record again" in text


def test_replay_without_the_hash_check() -> None:
    recording = _recorded(REQUEST, digest="0" * 64)
    assert ReplayClient({REQUEST.item_id: recording}, verify_hash=False).complete(REQUEST) is recording.response
    with pytest.raises(CassetteMismatch):
        ReplayClient({REQUEST.item_id: recording}).complete(REQUEST)


def test_replay_keeps_its_own_copy_of_the_recordings() -> None:
    recordings = {REQUEST.item_id: _recorded(REQUEST)}
    replay = ReplayClient(recordings)
    recordings.clear()
    assert replay.complete(REQUEST).content == "recorded answer"


def test_every_client_is_a_model_client(tmp_path: Path) -> None:
    writer = CassetteWriter(tmp_path, arm="a", feature="bugfix", prompt="bugfix/v2", model="demo-model", settings=REQUEST.settings)
    for client in (OpenAICompatClient("http://localhost:8000/v1"), ReplayClient({}), RecordingClient(ReplayClient({}), writer)):
        assert isinstance(client, ModelClient)


# ---- recording ------------------------------------------------------------------------------------


class FakeModel:
    """A stand-in live client that counts its calls."""

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail_on = fail_on or set()
        self._lock = threading.Lock()

    def complete(self, request: ChatRequest) -> ChatResponse:
        with self._lock:
            self.calls.append(request.item_id)
        if request.item_id in self.fail_on:
            raise ClientError("the endpoint is down")
        return ChatResponse(content=f"answer to {request.item_id}", finish_reason="stop", usage={"completion_tokens": 4}, latency_s=0.5)


def _item(item_id: str) -> ChatRequest:
    return ChatRequest(item_id=item_id, model="demo-model", messages=(Message("user", f"Fix {item_id}"),), settings=REQUEST.settings)


def _writer(root: Path) -> CassetteWriter:
    return CassetteWriter(root, arm="candidate", feature="bugfix", prompt="bugfix/v2", model="demo-model", settings=REQUEST.settings,
                          suite_name="tiny", suite_sha256="ab" * 32)


def test_recording_writes_every_answer_and_resumes_without_asking_again(tmp_path: Path) -> None:
    root = tmp_path / "live"
    model = FakeModel()
    writer = _writer(root)
    recorder = RecordingClient(model, writer)
    first = [recorder.complete(_item(f"bugfix:t{i}")) for i in range(3)]
    writer.close()
    assert model.calls == ["bugfix:t0", "bugfix:t1", "bugfix:t2"]
    assert [json.loads(line)["item_id"] for line in (root / "bugfix.jsonl").read_text(encoding="utf-8").splitlines()] == model.calls

    again = FakeModel()
    writer = _writer(root)
    recorder = RecordingClient(again, writer)
    resumed = [recorder.complete(_item(f"bugfix:t{i}")) for i in range(4)]
    writer.close()
    assert again.calls == ["bugfix:t3"]  # only the item not yet recorded was asked
    assert resumed[:3] == first and resumed[3].content == "answer to bugfix:t3"
    recordings = read_recordings(load_cassettes(root), "bugfix")
    assert sorted(recordings) == [f"bugfix:t{i}" for i in range(4)]
    assert all(recordings[f"bugfix:t{i}"].request_sha256 == _item(f"bugfix:t{i}").sha256() for i in range(4))


def test_a_failed_request_is_left_unrecorded(tmp_path: Path) -> None:
    root = tmp_path / "live"
    model = FakeModel(fail_on={"bugfix:t1"})
    writer = _writer(root)
    recorder = RecordingClient(model, writer)
    recorder.complete(_item("bugfix:t0"))
    with pytest.raises(ClientError, match="the endpoint is down"):
        recorder.complete(_item("bugfix:t1"))
    writer.close()
    assert not writer.has(_item("bugfix:t1"))
    assert sorted(read_recordings(load_cassettes(root), "bugfix")) == ["bugfix:t0"]
    assert load_cassettes(root).features["bugfix"].items == 1


def test_recording_refuses_to_mix_requests_into_a_conflicting_set(tmp_path: Path) -> None:
    root = tmp_path / "live"
    writer = _writer(root)
    RecordingClient(FakeModel(), writer).complete(_item("bugfix:t0"))
    writer.close()
    model = FakeModel()
    other = CassetteWriter(root, arm="candidate", feature="bugfix", prompt="bugfix/v1", model="demo-model", settings=REQUEST.settings,
                           suite_name="tiny", suite_sha256="ab" * 32)
    with pytest.raises(CassetteError, match="record into a new directory"):
        RecordingClient(model, other).complete(_item("bugfix:t1"))
    assert model.calls == []  # refused before the model was asked


def test_recording_a_live_endpoint_end_to_end(server: StubServer, tmp_path: Path) -> None:
    root = tmp_path / "live"
    server.queue(Reply(503, "warming up"), Reply(body=completion("fixed", reasoning_content="thinking")))
    sleeps: list[float] = []
    live = client_for(server, sleeps=sleeps)
    writer = _writer(root)
    answer = RecordingClient(live, writer).complete(REQUEST)
    writer.close()
    assert (answer.content, answer.reasoning, sleeps) == ("fixed", "thinking", [0.5])

    cset = load_cassettes(root)
    [recording] = read_recordings(cset, "bugfix").values()
    assert recording == Recording(REQUEST.item_id, REQUEST.sha256(), answer)
    assert KEY not in (root / "bugfix.jsonl").read_text(encoding="utf-8")
    assert KEY not in (root / "manifest.json").read_text(encoding="utf-8")
    assert ReplayClient(read_recordings(cset, "bugfix")).complete(REQUEST) == answer

    contacted = len(server.received)
    writer = _writer(root)
    assert RecordingClient(live, writer).complete(REQUEST) == answer  # resumed from disk
    writer.close()
    assert len(server.received) == contacted
