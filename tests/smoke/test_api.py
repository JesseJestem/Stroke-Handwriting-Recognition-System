from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)


def test_root_response_contains_request_id():
    response = client.get("/")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"]


def test_each_request_gets_unique_request_id():
    first_response = client.get("/")
    second_response = client.get("/")

    first_response_id = first_response.headers["X-Request-ID"]
    second_response_id = second_response.headers["X-Request-ID"]

    assert first_response_id != second_response_id