from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


PROFILES = {
    "fast": {"median": 0.02, "p95": 0.05, "failure_rate": 0.0},
    "standard": {"median": 2.0, "p95": 8.0, "failure_rate": 0.01},
}


def _rng(doc_id: str, attempt: int) -> random.Random:
    digest = hashlib.sha256(f"{doc_id}:{attempt}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def response_plan(doc_id: str, attempt: int, profile: str) -> tuple[float, int]:
    cfg = PROFILES[profile]
    rng = _rng(doc_id, attempt)
    sigma = math.log(cfg["p95"] / cfg["median"]) / 1.6448536269514722
    delay = rng.lognormvariate(math.log(cfg["median"]), sigma)
    # Only the first attempt is faulted, so the benchmark exercises recovery
    # without making final row counts depend on random retry exhaustion.
    status = 200
    if attempt == 1 and rng.random() < cfg["failure_rate"]:
        status = 429 if rng.random() < 0.5 else 500
    return delay, status


class MockHandler(BaseHTTPRequestHandler):
    server_version = "BenchmarkMockLLM/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(200, {"status": "ok", "profile": self.server.profile})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        doc_id = self.headers.get("X-Benchmark-Doc-Id", "unknown")
        attempt = int(self.headers.get("X-Benchmark-Attempt", "1"))
        delay, status = response_plan(doc_id, attempt, self.server.profile)
        time.sleep(delay)
        if status != 200:
            self._json(status, {"error": {"message": "injected benchmark failure"}})
            return
        prompt = json.dumps(body, ensure_ascii=False)
        content = json.dumps(
            {
                "summary": f"mock summary for {doc_id}",
                "category": "benchmark",
                "confidence": 0.99,
                "request_chars": len(prompt),
                "padding": "x" * 3800,
            },
            ensure_ascii=False,
        )
        self._json(
            200,
            {
                "id": f"mock-{doc_id}",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": max(1, len(prompt) // 4), "completion_tokens": len(content) // 4},
            },
        )

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except BrokenPipeError:
            pass


class MockServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], profile: str):
        super().__init__(address, MockHandler)
        self.profile = profile


def start_in_thread(host: str = "127.0.0.1", port: int = 8010, profile: str = "fast"):
    server = MockServer((host, port), profile)
    thread = threading.Thread(target=server.serve_forever, name="benchmark-mock-llm", daemon=True)
    thread.start()
    return server, thread


def serve(host: str, port: int, profile: str) -> None:
    server = MockServer((host, port), profile)
    try:
        print(f"Mock LLM listening on http://{host}:{port} ({profile})")
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="fast")
    args = parser.parse_args()
    serve(args.host, args.port, args.profile)
