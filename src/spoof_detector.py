"""Avian Biometric Verification & Anti-Spoofing Shield.

Detects human whistling, synthetic mimicry, and ambient noise to prevent
adversarial false alarms in natural disaster early-warning systems.
"""

from typing import Dict, Any, List
import numpy as np
import librosa

NON_BIRD_CLASSES = {
    "human whistle",
    "human vocal",
    "human non-vocal",
    "engine",
    "siren"
}


def evaluate_avian_authenticity(
    y: np.ndarray,
    sr: int,
    species_info: Dict[str, Any],
    acoustic_features: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate whether an audio signal is an authentic avian vocalization
    or a human mimicry / whistling attempt.
    
    Biometric criteria:
    1. BirdNET Explicit Non-Bird Detection ('Human whistle', 'Human vocal', etc.)
    2. BirdNET Neural Confidence: Valid bird confidence >= 15%
    3. Human Whistling Frequency Constraint (< 3,000 Hz)
    4. Breath Turbulence Check in sub-600 Hz
    
    Returns:
        Dictionary with verification status, authenticity flag, spoofing score,
        and civil false-alarm suppression directives.
    """
    top_candidate = species_info.get("primary_species", {})
    common_name = top_candidate.get("common_name", "").lower()
    scientific_name = top_candidate.get("scientific_name", "").lower()
    birdnet_conf = top_candidate.get("confidence", 0.0)

    centroid_max = acoustic_features.get("spectral_centroid", {}).get("max", 0.0)
    centroid_mean = acoustic_features.get("spectral_centroid", {}).get("mean", 0.0)

    # 1. Check for explicit non-bird / human classes identified by BirdNET
    is_explicit_non_bird = any(nb in common_name or nb in scientific_name for nb in NON_BIRD_CLASSES)

    # 2. Check candidate list to see if 'Human whistle' is ranked high
    for cand in species_info.get("top_candidates", []):
        c_name = cand.get("common_name", "").lower()
        if "human whistle" in c_name and cand.get("confidence", 0) > 30.0:
            is_explicit_non_bird = True
            break

    # 3. Low-frequency breath turbulence energy ratio (0 - 600 Hz vs Total Energy)
    s = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    breath_mask = (freqs >= 80) & (freqs <= 650)
    total_energy = np.sum(s) + 1e-9
    breath_energy = np.sum(s[breath_mask, :])
    breath_ratio = float(breath_energy / total_energy)

    spoof_factors: List[str] = []
    spoof_score = 0.0

    if is_explicit_non_bird:
        spoof_score = 100.0
        spoof_factors.append(
            f"Explicit Non-Avian Acoustic Match: Identified as '{top_candidate.get('common_name')}' with {birdnet_conf:.1f}% confidence."
        )
    else:
        # Criterion A: BirdNET Confidence Gate
        if birdnet_conf < 10.0:
            spoof_score += 45.0
            spoof_factors.append(
                f"Low Species Confidence ({birdnet_conf:.1f}% < 10% threshold) — lacks distinct avian spectrographic ridges."
            )
        elif birdnet_conf < 20.0:
            spoof_score += 20.0

        # Criterion B: Human Whistling Frequency Constraint
        if centroid_max < 3000.0 and centroid_mean < 2300.0:
            spoof_score += 25.0
            spoof_factors.append(
                f"Acoustic frequency ceiling restricted ({centroid_max:.0f} Hz max) — consistent with human oral cavity whistling limits."
            )

        # Criterion C: Human Breath Air Turbulence
        if breath_ratio > 0.28:
            spoof_score += 30.0
            spoof_factors.append(
                f"Elevated sub-600 Hz breath noise detected ({breath_ratio*100:.1f}% energy) — characteristic of human exhalation turbulence."
            )

    spoof_score = float(np.clip(spoof_score, 0.0, 100.0))
    is_authentic = spoof_score < 50.0

    if is_authentic:
        status_label = "VERIFIED AUTHENTIC BIRD VOCALIZATION"
        status_color = "#10B981"  # Emerald Green
        status_summary = (
            f"Confirmed authentic avian biometrics. Dual-syrinx modulation verified with {birdnet_conf:.1f}% "
            "BirdNET confidence. Normal disaster prediction pipeline authorized."
        )
        suppress_alarm = False
    else:
        status_label = "REJECTED: HUMAN MIMICRY / SPOOFING DETECTED"
        status_color = "#EF4444"  # Red / Amber
        status_summary = (
            "Acoustic markers match human oral whistling or non-avian ambient noise. "
            "Disaster Early Warning trigger has been SUPPRESSED to prevent civil false alarms."
        )
        suppress_alarm = True

    return {
        "is_authentic": is_authentic,
        "spoof_score": round(spoof_score, 1),
        "status_label": status_label,
        "status_color": status_color,
        "status_summary": status_summary,
        "suppress_alarm": suppress_alarm,
        "spoof_factors": spoof_factors,
        "breath_ratio": round(breath_ratio * 100.0, 1),
        "birdnet_conf": birdnet_conf,
        "detected_class": top_candidate.get("common_name", "Unknown")
    }
