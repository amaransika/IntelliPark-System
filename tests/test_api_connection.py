from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_api_root_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "Active", "message": "IntelliPark API is running."}

def test_anpr_status_endpoint():
    response = client.get("/api/anpr/status")
    assert response.status_code == 200
    data = response.json()
    assert "feed" in data
    assert "history" in data