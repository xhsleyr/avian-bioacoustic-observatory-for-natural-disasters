"""Project INTERA — Headless REST Telemetry API (FastAPI).

Provides headless high-throughput endpoints for remote IoT bioacoustic canopy nodes,
Raspberry Pi sensor hubs, and civil disaster warning feeds.
"""

import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.audio_processor import extract_acoustic_features, load_audio
from src.distress_engine import (
    compute_distress_index,
    compute_distress_timeline,
    evaluate_flock_consensus,
    generate_telemetry_csv,
    generate_telemetry_geojson,
    generate_telemetry_json,
)
from src.species_classifier import identify_species
from src.spoof_detector import evaluate_avian_authenticity

app = FastAPI(
    title="Project INTERA — Avian Bioacoustic Intelligence API",
    description="Planetary early-warning bioacoustic analysis and anti-spoofing telemetry engine.",
    version="2.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CANOPY_STATIONS = [
    {
        "station_id": "OBS-01-SHENANDOAH",
        "name": "Shenandoah Ridge Primary Array",
        "latitude": 38.5312,
        "longitude": -78.4321,
        "elevation_m": 850.0,
        "status": "ACTIVE",
        "channels": 2,
        "p_wave_lead_time_min": 65
    },
    {
        "station_id": "OBS-02-ALLEGHENY",
        "name": "Allegheny High Canopy Node",
        "latitude": 38.8921,
        "longitude": -79.2145,
        "elevation_m": 1120.0,
        "status": "ACTIVE",
        "channels": 1,
        "p_wave_lead_time_min": 80
    },
    {
        "station_id": "OBS-03-BLUERIDGE",
        "name": "Blue Ridge Infrasound Array",
        "latitude": 37.9542,
        "longitude": -79.0512,
        "elevation_m": 760.0,
        "status": "ACTIVE",
        "channels": 2,
        "p_wave_lead_time_min": 50
    },
    {
        "station_id": "OBS-04-PIEDMONT",
        "name": "Piedmont Foothills Sentinel",
        "latitude": 38.0293,
        "longitude": -78.4767,
        "elevation_m": 240.0,
        "status": "STANDBY",
        "channels": 1,
        "p_wave_lead_time_min": 40
    }
]


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp_utc: str
    active_canopy_nodes: int


@app.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """System health check and canopy node readiness."""
    return HealthResponse(
        status="OPERATIONAL",
        version="2.4.0",
        timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        active_canopy_nodes=len(CANOPY_STATIONS)
    )


@app.get("/api/v1/stations")
def get_canopy_stations() -> list[dict[str, Any]]:
    """Get list of active planetary canopy observatory nodes and coordinates."""
    return CANOPY_STATIONS


@app.post("/api/v1/analyze")
async def analyze_audio_stream(
    file: UploadFile
) -> dict[str, Any]:
    """Analyze raw bioacoustic audio stream for avian distress, taxonomy, and anti-spoofing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing audio file.")

    audio_bytes = await file.read()
    if len(audio_bytes) < 64:
        raise HTTPException(status_code=400, detail="Audio file empty or corrupted.")
    if len(audio_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File size exceeds 25 MB limit.")

    try:
        y, sr, duration = load_audio(audio_bytes, apply_denoising=True)
        features = extract_acoustic_features(y, sr)
        species_info = identify_species(audio_bytes, features, top_k=5)
        results = compute_distress_index(features, species_info=species_info)
        spoof_info = evaluate_avian_authenticity(y, sr, species_info, features)
        timeline_info = compute_distress_timeline(y, sr, species_info=species_info)
        consensus_info = evaluate_flock_consensus(species_info, features, results["adi"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Acoustic inference error: {e}") from e

    return {
        "status": "SUCCESS",
        "stream_name": file.filename,
        "duration_sec": round(duration, 2),
        "sampling_rate_hz": sr,
        "distress_analysis": results,
        "species_classification": species_info,
        "biometric_shield": spoof_info,
        "timeline_escalation": timeline_info,
        "flock_consensus": consensus_info,
        "formats": {
            "json_stream": generate_telemetry_json(
                file.filename, duration, sr, features, species_info, spoof_info, results, timeline_info, consensus_info
            ),
            "geojson": generate_telemetry_geojson(
                file.filename, results, species_info, spoof_info, timeline_info
            ),
            "csv_matrix": generate_telemetry_csv(
                file.filename, duration, sr, features, species_info, spoof_info, results, timeline_info
            )
        }
    }
