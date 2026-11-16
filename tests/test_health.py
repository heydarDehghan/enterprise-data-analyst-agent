def test_health_endpoint_returns_ok_json() -> None:
    """Phase 0 contract: the service exposes a minimal liveness endpoint."""
    from app.main import app
    from fastapi.testclient import TestClient

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"status": "ok"}
