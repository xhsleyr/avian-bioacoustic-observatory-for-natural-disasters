"""Unit tests for headless REST API endpoints."""

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health():
    """Verify health endpoint returns operational status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OPERATIONAL"
    assert data["active_canopy_nodes"] == 4


def test_get_stations():
    """Verify canopy station roster."""
    response = client.get("/api/v1/stations")
    assert response.status_code == 200
    stations = response.json()
    assert len(stations) >= 4
    assert stations[0]["station_id"] == "OBS-01-SHENANDOAH"


def test_analyze_audio_endpoint():
    """Verify audio analysis endpoint with real sample."""
    with open("data/samples/calm_robin_song.wav", "rb") as f:
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("calm_robin_song.wav", f, "audio/wav")}
        )
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "SUCCESS"
    assert "distress_analysis" in res
    assert "formats" in res
    assert "geojson" in res["formats"]
