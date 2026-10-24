"""Distress analysis engine and natural disaster early-warning scoring.

Computes the Acoustic Distress Index (ADI) by evaluating biological markers
such as pitch elevation, call burst density, noise harshness, and energy volatility.
"""

from typing import Dict, Any, List
import numpy as np


def compute_distress_index(features: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the Acoustic Distress Index (ADI) and Early Warning Threat Tier.
    
    Formula weights:
    - Pitch Elevation (Spectral Centroid): 35%
    - Burst Density (Call Rate / Onsets): 30%
    - Spectral Flatness / Harsh Chaos: 20%
    - Energy Volatility (RMS Variation): 15%
    
    Returns:
        Dictionary with overall ADI score (0-100), threat classification,
        sub-scores, and diagnostic insights.
    """
    centroid_mean = features["spectral_centroid"]["mean"]
    call_rate = features["call_rate"]
    flatness_mean = features["spectral_flatness"]["mean"]
    rms_std = features["rms_energy"]["std"]
    duration = features["duration"]

    # 1. Pitch Score (0 to 100)
    # Calm song: ~1500 - 2800 Hz
    # Elevated alarm: 3500 - 4500 Hz
    # Extreme panic: > 5500 - 7500 Hz
    norm_pitch = (centroid_mean - 2000.0) / 3800.0
    score_pitch = float(np.clip(norm_pitch, 0.0, 1.0) * 100.0)

    # 2. Call Rate Score (0 to 100)
    # Calm song: 0.3 - 1.2 calls/sec
    # Elevated calls: 1.8 - 2.8 calls/sec
    # Frantic panic bursts: > 3.5 calls/sec
    norm_rate = (call_rate - 0.6) / 3.0
    score_rate = float(np.clip(norm_rate, 0.0, 1.0) * 100.0)

    # 3. Harshness / Flatness Score (0 to 100)
    # Melodic pure tone: ~0.0001 - 0.01
    # Harsh panic noise / chaotic screech: > 0.05 - 0.12
    norm_flatness = (flatness_mean - 0.005) / 0.06
    score_flatness = float(np.clip(norm_flatness, 0.0, 1.0) * 100.0)

    # 4. Energy Volatility Score (0 to 100)
    norm_energy = (rms_std - 0.04) / 0.16
    score_energy = float(np.clip(norm_energy, 0.0, 1.0) * 100.0)

    # Composite Acoustic Distress Index (ADI)
    adi = (
        0.35 * score_pitch +
        0.30 * score_rate +
        0.20 * score_flatness +
        0.15 * score_energy
    )
    adi = round(float(np.clip(adi, 0.0, 100.0)), 1)

    # Classify Early Warning Status
    if adi >= 70.0:
        threat_level = "CRITICAL"
        threat_color = "#E63946"  # Red
        alert_title = "CRITICAL EARLY WARNING: Severe Collective Panic Detected"
        alert_summary = (
            "Acoustic signatures show extreme high-frequency shrieking, frantic burst intervals, "
            "and chaotic noise distortion. These bioacoustic anomalies strongly correlate with "
            "imminent natural threats (e.g., pre-seismic P-wave ground vibrations, approaching cyclone infrasound)."
        )
    elif adi >= 38.0:
        threat_level = "ELEVATED"
        threat_color = "#F4A261"  # Amber/Orange
        alert_title = "ELEVATED ALERT: Abnormal Agitation Detected"
        alert_summary = (
            "Higher-than-normal pitch and accelerated vocalization frequency detected. "
            "Indicates heightened flock agitation, localized predator activity, or sudden barometric pressure drop."
        )
    else:
        threat_level = "NORMAL"
        threat_color = "#2A9D8F"  # Green
        alert_title = "NORMAL STATUS: Baseline Avian Communication"
        alert_summary = (
            "Vocalizations exhibit stable melodic contours, regular rhythmic call pacing, "
            "and clear harmonic balance. No environmental distress markers detected."
        )

    # Diagnostic breakdown for presentation & reports
    diagnostics: List[Dict[str, Any]] = [
        {
            "name": "Pitch Elevation Index",
            "score": round(score_pitch, 1),
            "value": f"{centroid_mean:.0f} Hz",
            "normal_range": "1,500 – 2,800 Hz",
            "status": "High / Screech" if score_pitch > 60 else "Normal Harmonic"
        },
        {
            "name": "Vocal Burst Frequency (Call Rate)",
            "score": round(score_rate, 1),
            "value": f"{call_rate:.2f} calls/sec",
            "normal_range": "0.3 – 1.2 calls/sec",
            "status": "Accelerated" if score_rate > 50 else "Calm Interval"
        },
        {
            "name": "Acoustic Chaos & Harshness",
            "score": round(score_flatness, 1),
            "value": f"{flatness_mean:.4f}",
            "normal_range": "< 0.0100",
            "status": "Harsh Chaos" if score_flatness > 50 else "Harmonic Song"
        },
        {
            "name": "Energy Volatility",
            "score": round(score_energy, 1),
            "value": f"{rms_std:.3f} RMS-std",
            "normal_range": "0.02 – 0.08",
            "status": "Erratic Volatility" if score_energy > 55 else "Controlled Energy"
        }
    ]

    return {
        "adi": adi,
        "threat_level": threat_level,
        "threat_color": threat_color,
        "alert_title": alert_title,
        "alert_summary": alert_summary,
        "sub_scores": {
            "pitch": round(score_pitch, 1),
            "call_rate": round(score_rate, 1),
            "flatness": round(score_flatness, 1),
            "energy_volatility": round(score_energy, 1)
        },
        "diagnostics": diagnostics,
        "duration_analyzed": round(duration, 2)
    }
