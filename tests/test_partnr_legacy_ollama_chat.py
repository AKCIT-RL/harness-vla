import json
from unittest.mock import patch

import pytest

from harness_vla.envs.partnr.legacy.ollama_chat import chat_ollama, ollama_api_base


def test_native_chat_separates_thinking_and_reports_usage():
    body = json.dumps(
        {
            "model": "gemma4:12b",
            "message": {
                "thinking": "Navigate before picking.",
                "content": '{"action":"Navigate","args":"table_1"}',
            },
            "prompt_eval_count": 100,
            "eval_count": 20,
            "total_duration": 2_500_000_000,
            "eval_duration": 1_000_000_000,
            "done_reason": "stop",
        }
    ).encode()
    response = type(
        "Response",
        (),
        {
            "__enter__": lambda self: self,
            "__exit__": lambda self, *args: None,
            "read": lambda self: body,
        },
    )()

    with patch("harness_vla.envs.partnr.legacy.ollama_chat.urllib.request.urlopen", return_value=response) as urlopen:
        content, meta = chat_ollama(
            base_url="http://localhost:11435/v1",
            model="gemma4:12b",
            messages=[{"role": "user", "content": "task"}],
            think=True,
            temperature=0,
            max_tokens=4096,
            timeout_s=600,
            num_ctx=8192,
            stop=["\n"],
        )

    assert content == '{"action":"Navigate","args":"table_1"}'
    assert meta["thinking"] == "Navigate before picking."
    assert meta["total_tokens"] == 120
    assert meta["total_duration_s"] == 2.5
    request = urlopen.call_args.args[0]
    sent = json.loads(request.data)
    assert request.full_url == "http://localhost:11435/api/chat"
    assert sent["think"] is True
    assert sent["options"]["num_ctx"] == 8192
    assert sent["options"]["stop"] == ["\n"]


def test_native_chat_rejects_non_ollama_path():
    with pytest.raises(ValueError, match="Ollama"):
        ollama_api_base("http://localhost:11435/openai/v1")