from __future__ import annotations

import multiprocessing
import os
import time

import httpx
import pytest
import uvicorn

os.environ["FRONTEND_DEV_MODE"] = "1"

from main import app


def run_server():
    ""'Run the FastAPI server in a separate process.""'
    uvicorn.run(app, host="0.0.0.0", port=8000)


@pytest.fixture(scope="module")
def server():
    ""'Start the server in a background process and tear it down after the tests.""'
    proc = multiprocessing.Process(target=run_server, args=())
    proc.start()
    time.sleep(2)  # Wait for the server to start
    yield
    proc.terminate()


def test_chat_e2e(server):
    ""'Test the chat endpoint with a real HTTP request.""'
    with httpx.stream("POST", "http://localhost:8000/api/chat", json={"message": "Hello"}, timeout=5) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(line[6:])

        assert events
        assert events[-1] == "[DONE]"
