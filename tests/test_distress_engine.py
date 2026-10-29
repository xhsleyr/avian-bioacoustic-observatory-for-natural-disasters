"""Unit tests for distress engine, early warning algorithms, and telemetry serialization."""

import json

from src.distress_engine import (
    compute_distress_index,
    compute_distress_timeline,
    evaluate_flock_consensus,
    generate_forensic_report,
    generate_telemetry_csv,
    generate_telemetry_geojson,
    generate_telemetry_json,
    get_species_baseline,
)


def test_species_baselines():
    """Verify taxonomic baseline classification mapping."""
    base_robin = get_species_baseline("European Robin", "Erithacus rubecula")
    assert "Passerine" in base_robin["family_label"]

    base_peafowl = get_species_baseline("Indian Peafowl", "Pavo cristatus")
    assert "Galliform" in base_peafowl["family_label"]

    base_owl = get_species_baseline("Great Horned Owl", "Bubo virginianus")
    assert "Raptor" in base_owl["family_label"]


def test_compute_distress_index():
    """Verify ADI computation on nominal vs severe input."""
    normal_features = {
        "spectral_centroid": {"mean": 2400.0, "std": 120.0, "max": 2800.0, "trajectory": [2400.0]},
        "call_rate": 0.8,
        "spectral_flatness": {"mean": 0.0025, "trajectory": [0.0025]},
        "rms_energy": {"mean": 0.12, "std": 0.02, "max": 0.25, "trajectory": [0.12]},
        "duration": 5.0
    }
    species_info = {"primary_species": {"common_name": "European Robin", "scientific_name": "Erithacus rubecula"}}
    res = compute_distress_index(normal_features, species_info)
    assert res["adi"] < 38.0
    assert res["threat_level"] == "NORMAL"

    panic_features = {
        "spectral_centroid": {"mean": 6800.0, "std": 900.0, "max": 8500.0, "trajectory": [6800.0]},
        "call_rate": 3.8,
        "spectral_flatness": {"mean": 0.052, "trajectory": [0.052]},
        "rms_energy": {"mean": 0.65, "std": 0.18, "max": 0.95, "trajectory": [0.65]},
        "duration": 5.0
    }
    res_panic = compute_distress_index(panic_features, species_info)
    assert res_panic["adi"] >= 70.0
    assert res_panic["threat_level"] == "CRITICAL"


def test_distress_timeline():
    """Verify temporal distress escalation calculation."""
    import numpy as np
    sr = 22050
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    y = np.sin(2 * np.pi * 3000 * t)
    timeline = compute_distress_timeline(y, sr)
    assert "times" in timeline
    assert "adi_series" in timeline
    assert "escalation_rate" in timeline
    assert len(timeline["adi_series"]) == len(timeline["times"])


def test_flock_consensus():
    """Verify multi-species consensus logic."""
    species_info = {
        "is_valid_bird": True,
        "primary_species": {"common_name": "Northern Cardinal"},
        "top_candidates": [
            {"common_name": "Northern Cardinal", "confidence": 92.0},
            {"common_name": "Song Sparrow", "confidence": 84.0},
            {"common_name": "American Robin", "confidence": 78.0}
        ]
    }
    features = {"call_rate": 3.4}
    consensus = evaluate_flock_consensus(species_info, features, adi_score=78.0)
    assert consensus["consensus_type"] == "macro_ecological"


def test_telemetry_serializers():
    """Verify multi-format serialization into valid JSON, GeoJSON, and CSV."""
    features = {
        "duration": 5.0,
        "call_rate": 2.5,
        "spectral_centroid": {"mean": 4200.0},
        "spectral_flatness": {"mean": 0.02}
    }
    species_info = {"primary_species": {"common_name": "Northern Cardinal", "scientific_name": "Cardinalis cardinalis", "confidence": 94.0}}
    spoof_info = {"is_authentic": True, "spoof_score": 12.0, "sub800_ratio": 0.015, "suppress_alarm": False}
    distress_info = {"adi": 75.0, "threat_level": "CRITICAL", "baseline_profile": {"family_label": "Passerine"}}
    timeline_info = {"escalation_rate": 3.1, "shockwave_detected": False, "peak_adi": 75.0, "peak_time": 3.2}
    consensus_info = {"status": "MACRO ECOLOGICAL PANIC", "active_species": ["Northern Cardinal"]}

    # JSON
    s_json = generate_telemetry_json("test_stream.wav", 5.0, 22050, features, species_info, spoof_info, distress_info, timeline_info, consensus_info)
    parsed_json = json.loads(s_json)
    assert parsed_json["early_warning_distress"]["adi_score"] == 75.0

    # GeoJSON
    s_geo = generate_telemetry_geojson("test_stream.wav", distress_info, species_info, spoof_info, timeline_info)
    parsed_geo = json.loads(s_geo)
    assert parsed_geo["type"] == "FeatureCollection"
    assert parsed_geo["features"][0]["geometry"]["type"] == "Point"

    # CSV
    s_csv = generate_telemetry_csv("test_stream.wav", 5.0, 22050, features, species_info, spoof_info, distress_info, timeline_info)
    assert "parameter,measured_value" in s_csv
    assert "adi_distress_index,75.0" in s_csv

    # Forensic Markdown Report
    s_rep = generate_forensic_report("test_stream.wav", 5.0, 22050, features, species_info, spoof_info, distress_info, timeline_info, consensus_info)
    assert "AVIAN BIOACOUSTIC OBSERVATORY" in s_rep
