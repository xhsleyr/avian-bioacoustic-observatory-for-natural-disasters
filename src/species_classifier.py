"""Species identification module for avian vocalizations.

Identifies the bird species from audio characteristics and correlates
the detected vocalization against the species' known baseline.
"""

from typing import Dict, Any, List
import numpy as np


# Bioacoustic database of common bird species and their acoustic baselines
SPECIES_DATABASE = [
    {
        "id": "amerob",
        "common_name": "American Robin",
        "scientific_name": "Turdus migratorius",
        "family": "Turdidae (Thrushes)",
        "base_pitch_hz": 2400,
        "pitch_range": (1800, 3200),
        "call_rate_range": (0.4, 1.4),
        "call_type": "Melodic Whistle & Flute Song",
        "disaster_sensitivity": "High sensitivity to seismic P-wave tremors; exhibits abrupt singing cessation followed by frantic high-pitch warning calls."
    },
    {
        "id": "houspa",
        "common_name": "House Sparrow",
        "scientific_name": "Passer domesticus",
        "family": "Passeridae",
        "base_pitch_hz": 4100,
        "pitch_range": (3400, 4800),
        "call_rate_range": (1.2, 2.8),
        "call_type": "Rapid Chirps & Chatter",
        "disaster_sensitivity": "Sensitive to micro-barometric air pressure drops prior to severe storms; forms tight agitated ground clusters."
    },
    {
        "id": "eurchi",
        "common_name": "Common Chaffinch",
        "scientific_name": "Fringilla coelebs",
        "family": "Fringillidae",
        "base_pitch_hz": 3300,
        "pitch_range": (2600, 4200),
        "call_rate_range": (0.8, 2.0),
        "call_type": "Cascading Musical Trill",
        "disaster_sensitivity": "Historically observed exhibiting prolonged alarm calls prior to storm fronts and localized earth tremors."
    },
    {
        "id": "blujay",
        "common_name": "Blue Jay",
        "scientific_name": "Cyanocitta cristata",
        "family": "Corvidae",
        "base_pitch_hz": 2900,
        "pitch_range": (2000, 3800),
        "call_rate_range": (0.5, 1.8),
        "call_type": "Harsh Jay Call & Whistle Mimic",
        "disaster_sensitivity": "Highly vocal sentinel species; mobbing calls escalate rapidly during early infrasound disturbance."
    },
    {
        "id": "barowl",
        "common_name": "Barn Owl",
        "scientific_name": "Tyto alba",
        "family": "Tytonidae",
        "base_pitch_hz": 1600,
        "pitch_range": (1000, 2200),
        "call_rate_range": (0.2, 0.8),
        "call_type": "Rasping Low Hiss & Screech",
        "disaster_sensitivity": "Nocturnal hearing acutely tuned to low-frequency ground vibrations and seismic P-waves."
    },
    {
        "id": "eursta",
        "common_name": "European Starling",
        "scientific_name": "Sturnus vulgaris",
        "family": "Sturnidae",
        "base_pitch_hz": 5800,
        "pitch_range": (4500, 7500),
        "call_rate_range": (2.0, 5.0),
        "call_type": "Whistling, Chattering & Shrieking",
        "disaster_sensitivity": "Exhibits collective murmuration flight panic when detecting infrasonic storm shockwaves."
    },
    {
        "id": "compig",
        "common_name": "Rock Pigeon / Dove",
        "scientific_name": "Columba livia",
        "family": "Columbidae",
        "base_pitch_hz": 650,
        "pitch_range": (300, 1100),
        "call_rate_range": (0.3, 1.0),
        "call_type": "Low Guttural Coo",
        "disaster_sensitivity": "Demonstrated navigational and infrasound sensitivity (<10 Hz); leaves perches en masse hours before cyclone landfall."
    },
    {
        "id": "corcro",
        "common_name": "Common Crow / Raven",
        "scientific_name": "Corvus corax",
        "family": "Corvidae",
        "base_pitch_hz": 1500,
        "pitch_range": (900, 2200),
        "call_rate_range": (0.4, 1.5),
        "call_type": "Guttural Caw & Croak",
        "disaster_sensitivity": "Extremely intelligent sentinel bird; circles high in thermal updrafts prior to earthquakes and severe tornadic events."
    }
]


def identify_species(features: Dict[str, Any]) -> Dict[str, Any]:
    """Identify the most likely bird species from bioacoustic parameters.
    
    Computes a multi-dimensional acoustic similarity score comparing
    detected pitch, call rate, and noise profile against species baselines.
    """
    detected_pitch = features["spectral_centroid"]["mean"]
    detected_rate = features["call_rate"]
    detected_flatness = features["spectral_flatness"]["mean"]

    scored_species: List[Dict[str, Any]] = []

    for sp in SPECIES_DATABASE:
        base_pitch = sp["base_pitch_hz"]
        p_min, p_max = sp["pitch_range"]
        r_min, r_max = sp["call_rate_range"]

        # Pitch distance (Gaussian penalty)
        pitch_dist = abs(detected_pitch - base_pitch) / 1800.0
        pitch_sim = np.exp(-0.5 * (pitch_dist ** 2))

        # Call rate distance
        mid_rate = (r_min + r_max) / 2.0
        rate_dist = abs(detected_rate - mid_rate) / 2.5
        rate_sim = np.exp(-0.5 * (rate_dist ** 2))

        # Combined similarity score
        score = 0.65 * pitch_sim + 0.35 * rate_sim
        scored_species.append({
            "species": sp,
            "raw_score": float(score)
        })

    # Sort by descending similarity score
    scored_species.sort(key=lambda x: x["raw_score"], reverse=True)

    # Softmax normalization over top 3 for realistic confidence percentages
    top_3 = scored_species[:3]
    top_scores = np.array([x["raw_score"] for x in top_3])
    # Temperature scaling
    exp_scores = np.exp(top_scores * 3.5)
    probs = exp_scores / np.sum(exp_scores)

    primary = top_3[0]["species"]
    primary_confidence = float(probs[0] * 100.0)

    # Build candidates list
    candidates = []
    for i, item in enumerate(top_3):
        candidates.append({
            "common_name": item["species"]["common_name"],
            "scientific_name": item["species"]["scientific_name"],
            "family": item["species"]["family"],
            "confidence": round(float(probs[i] * 100.0), 1),
            "base_pitch_hz": item["species"]["base_pitch_hz"],
            "disaster_sensitivity": item["species"]["disaster_sensitivity"]
        })

    # Species-specific deviation analysis
    base_freq = primary["base_pitch_hz"]
    delta_hz = detected_pitch - base_freq
    pct_deviation = (delta_hz / base_freq) * 100.0

    if pct_deviation > 40.0:
        vocal_deviation_state = f"High Pitch Elevation (+{pct_deviation:.1f}% above species baseline)"
    elif pct_deviation < -30.0:
        vocal_deviation_state = f"Sub-harmonic / Low Frequency Shift ({pct_deviation:.1f}% below species baseline)"
    else:
        vocal_deviation_state = f"Normal Species Pitch Band ({pct_deviation:+.1f}% of baseline)"

    return {
        "primary_species": {
            "common_name": primary["common_name"],
            "scientific_name": primary["scientific_name"],
            "family": primary["family"],
            "confidence": round(primary_confidence, 1),
            "call_type": primary["call_type"],
            "base_pitch_hz": base_freq,
            "disaster_sensitivity": primary["disaster_sensitivity"],
            "vocal_deviation_state": vocal_deviation_state,
            "pitch_deviation_pct": round(pct_deviation, 1)
        },
        "top_candidates": candidates
    }
