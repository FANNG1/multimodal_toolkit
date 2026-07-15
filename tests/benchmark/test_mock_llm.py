from __future__ import annotations

import json
import urllib.request

from benchmark.audio.mock_llm import response_plan, start_in_thread


def test_response_plan_is_deterministic():
    assert response_plan("doc-1", 1, "standard") == response_plan("doc-1", 1, "standard")
    delay, status = response_plan("doc-1", 2, "fast")
    assert delay >= 0
    assert status == 200


def test_mock_has_health_and_openai_response():
    server, thread = start_in_thread(port=0, profile="fast")
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as response:
            assert json.load(response)["status"] == "ok"
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": "hello"}]}).encode(),
            headers={"Content-Type": "application/json", "X-Benchmark-Doc-Id": "doc-1"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
        content = payload["choices"][0]["message"]["content"]
        assert json.loads(content)["category"] == "benchmark"
        assert len(content) >= 3800
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
