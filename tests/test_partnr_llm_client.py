"""LLMClient transport tests against a local fake server (stdlib only)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from harness_vla.envs.partnr.llm_client import LLMClient, ollama_api_base


class _Handler(BaseHTTPRequestHandler):
    requests = []
    reply = {}

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        _Handler.requests.append((self.path, payload))
        body = json.dumps(_Handler.reply).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence
        pass


@pytest.fixture
def server():
    _Handler.requests = []
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def test_openai_compat_post_and_seed(server, monkeypatch):
    monkeypatch.setenv("PARTNR_LLM_SEED", "1004")
    _Handler.reply = {"choices": [{"message": {"content": '{"agent_0": "Wait[]"}'}}],
                      "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
    client = LLMClient(base_url=server, model="m", max_tokens=32)
    client.start_episode(agent_uid=1)
    content = client.post([{"role": "user", "content": "hi"}])
    assert content == '{"agent_0": "Wait[]"}'
    assert client.last_meta["backend"] == "openai_compat"
    assert client.last_meta["total_tokens"] == 10
    path, payload = _Handler.requests[-1]
    assert path == "/v1/chat/completions"
    assert payload["seed"] == 1004 + 100_000
    client.post([{"role": "user", "content": "again"}])
    assert _Handler.requests[-1][1]["seed"] == 1004 + 100_000 + 1


def test_native_ollama_post_thinking_off(server, monkeypatch):
    monkeypatch.delenv("PARTNR_LLM_SEED", raising=False)
    _Handler.reply = {"model": "m", "message": {"content": "Wait[]", "thinking": ""}, "done": True,
                      "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 2,
                      "eval_duration": 2_000_000_000}
    client = LLMClient(base_url=server, model="m", disable_thinking=True, num_ctx=8192)
    client.start_episode()
    content = client.post([{"role": "user", "content": "hi"}])
    assert content == "Wait[]"
    path, payload = _Handler.requests[-1]
    assert path == "/api/chat" and payload["think"] is False
    assert payload["options"]["num_ctx"] == 8192 and "seed" not in payload["options"]
    assert client.last_meta["backend"] == "ollama_native"
    assert client.last_meta["completion_tokens"] == 2 and client.last_meta["tok_per_s"] == 1.0


def test_empty_content_and_transport_failure_are_llm_failures(server):
    _Handler.reply = {"choices": [{"message": {"content": ""}}]}
    client = LLMClient(base_url=server, model="m")
    assert client.post([{"role": "user", "content": "hi"}]) == ""
    assert client.last_meta["llm_failure"] and client.last_meta["llm_error"] == "empty_content"
    dead = LLMClient(base_url="http://127.0.0.1:9/v1", model="m", timeout_s=0.5, max_retries=1)
    assert dead.post([{"role": "user", "content": "hi"}]) == ""
    assert dead.last_meta["llm_failure"]


def test_flags_and_api_base():
    with pytest.raises(ValueError):
        LLMClient(enable_thinking=True, disable_thinking=True)
    assert ollama_api_base("http://h:11435/v1") == "http://h:11435"
    assert ollama_api_base("http://h:11435") == "http://h:11435"
