from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ["FRONTEND_DEV_MODE"] = "1"

from main import app


@pytest.fixture
def client():
    ""'Create a test client for the FastAPI app.""'
    return TestClient(app)


def test_index_route(client: TestClient):
    ""'Test that the index route returns a 200 status code and HTML content.""'
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_chat_route_streaming(client: TestClient):
    ""'Test that the chat route returns a streaming response.""'
    response = client.post("/api/chat", json={"message": "Hello"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert response.text.startswith("data: ")
    assert response.text.endswith("data: [DONE]\n\n")
