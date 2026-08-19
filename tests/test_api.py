from fastapi.testclient import TestClient

from api.main import app


client = TestClient(
    app
)


def test_root():

    response = client.get(
        "/"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "online"


def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }


def test_invalid_file_type():

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"invalid file",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400