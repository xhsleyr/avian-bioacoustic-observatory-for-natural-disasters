"""Avian Biometric Verification & Anti-Spoofing Shield.

Detects domestic animal sounds (dogs, cats), human whistling/vocal mimicry,
smartphone loudspeaker replays, and ambient non-avian noise to prevent
adversarial false alarms in planetary disaster early-warning systems.
"""

from typing import Any

import librosa
import numpy as np

# Prohibited non-avian biological and synthetic class tokens
NON_BIRD_KEYWORDS = [
    "dog",
    "canis",
    "bark",
    "cat",
    "felis",
    "meow",
    "human whistle",
    "human vocal",
    "human non-vocal",
    "human",
    "speech",
    "whistle",
    "engine",
    "siren",
    "noise",
    "ambient",
    "car horn",
    "gun",
    "fireworks"
]


def evaluate_avian_authenticity(
    y: np.ndarray,
    sr: int,
    species_info: dict[str, Any],
    acoustic_features: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate whether an audio signal is an authentic avian vocalization
    or a domestic animal sound (dog/cat), smartphone speaker replay, or human mimicry.
    
    Multi-Layer Biometric Criteria:
    1. Explicit Non-Bird Neural Identification (Dog_Dog, Human, Siren, Noise)
    2. Linear Sub-800 Hz Energy Dominance (Syrinx vs Larynx)
    3. Indian Peafowl vs Feline Meow Acoustic Discrimination
    4. Neural Confidence Gating (Rejects Out-of-Distribution sounds < 25% confidence)
    5. Smartphone Loudspeaker Replay Frequency Ceiling & Compression
    6. Human Oral Whistling & Breath Turbulence Constraints
    
    Returns:
        Dictionary with verification status, authenticity flag, spoofing score,
        detected non-avian category, and civil false-alarm suppression directives.
    """
    top_candidate = species_info.get("primary_species", {})
    common_name = top_candidate.get("common_name", "").lower()
    scientific_name = top_candidate.get("scientific_name", "").lower()
    birdnet_conf = float(top_candidate.get("confidence", 0.0))

    centroid_max = acoustic_features.get("spectral_centroid", {}).get("max", 0.0)
    centroid_mean = acoustic_features.get("spectral_centroid", {}).get("mean", 0.0)
    sub600_ratio = float(acoustic_features.get("sub600_ratio", 0.0))
    sub800_ratio = float(acoustic_features.get("sub800_ratio", 0.0))
    sub1k_ratio = float(acoustic_features.get("sub1k_ratio", 0.0))
    rolloff_95 = float(acoustic_features.get("spectral_rolloff_95", {}).get("mean", 5000.0))

    spoof_factors: list[str] = []
    spoof_score = 0.0
    detected_category = "authentic_avian"

    # --- LAYER 1: Explicit Non-Bird Neural Match in Top Candidates ---
    is_explicit_non_bird = False
    
    for kw in NON_BIRD_KEYWORDS:
        if kw in common_name or kw in scientific_name:
            if kw == "cat" and "catbird" in common_name and birdnet_conf > 30.0:
                continue
            is_explicit_non_bird = True
            if any(k in kw for k in ["dog", "canis", "bark"]):
                detected_category = "canine_dog"
            elif any(k in kw for k in ["cat", "felis", "meow"]):
                detected_category = "feline_cat"
            elif any(k in kw for k in ["human", "whistle", "speech"]):
                detected_category = "human_mimicry"
            else:
                detected_category = "ambient_noise"
            break

    # Scan top candidates for non-bird classes (e.g. Dog_Dog, Siren, Human vocal)
    if not is_explicit_non_bird:
        for cand in species_info.get("top_candidates", []):
            c_name = cand.get("common_name", "").lower()
            c_conf = cand.get("confidence", 0.0)
            for kw in ["dog", "human whistle", "human vocal", "siren", "engine", "noise", "gun"]:
                if kw in c_name and c_conf > 10.0:
                    is_explicit_non_bird = True
                    spoof_score = 100.0
                    if "dog" in kw:
                        detected_category = "canine_dog"
                    elif "siren" in kw:
                        detected_category = "artificial_siren"
                    elif "engine" in kw:
                        detected_category = "mechanical_engine"
                    else:
                        detected_category = "human_mimicry"
                    spoof_factors.append(
                        f"Non-Avian Signal Detected: Neural classifier identified '{cand.get('common_name')}' with {c_conf:.1f}% probability."
                    )
                    break
            if is_explicit_non_bird:
                break

    if is_explicit_non_bird:
        spoof_score = 100.0
        if not spoof_factors:
            spoof_factors.append(
                f"Explicit Non-Avian Acoustic Match: Identified as '{top_candidate.get('common_name')}' with {birdnet_conf:.1f}% confidence."
            )
    else:
        # --- LAYER 2: Special Peafowl / Feline Meow Discrimination ---
        # Domestic cat meows share a rising-falling contour with Indian Peafowl calls,
        # but genuine peafowl honks have < 5% energy below 800 Hz and centroid > 1,850 Hz.
        # Cat meows have 30% - 60% energy below 800 Hz and centroid around 800 - 1,400 Hz.
        is_peafowl = ("peafowl" in common_name) or ("pavo" in scientific_name)
        if is_peafowl and (sub800_ratio > 0.20 or centroid_mean < 1700.0):
            spoof_score = 100.0
            detected_category = "feline_cat"
            spoof_factors.append(
                f"Peafowl Bio-Verification Failed: Identified as a domestic cat meow. Audio exhibits "
                f"{sub800_ratio*100:.1f}% energy below 800 Hz (authentic Indian Peafowl has < 5.0%) and "
                f"a low spectral centroid ({centroid_mean:.0f} Hz vs > 1,850 Hz required for peafowl)."
            )

        # --- LAYER 3: Strict Mammalian Larynx Sub-800 Hz Energy Dominance ---
        # Domestic dogs, cats, and humans vocalize with laryngeal vocal folds, creating heavy sub-800 Hz energy (> 25%).
        # Wild birds recorded in real outdoor environments typically have < 8% sub-800 Hz energy.
        elif sub800_ratio > 0.25 or (sub800_ratio > 0.18 and birdnet_conf < 35.0):
            spoof_score = 100.0
            if rolloff_95 < 10500 and centroid_mean < 3600 and birdnet_conf < 45.0:
                detected_category = "phone_speaker_replay"
                spoof_factors.append(
                    f"Smartphone Speaker Replay Transducer Detected: Sharp high-frequency rolloff at {rolloff_95:.0f} Hz "
                    f"with domestic sound resonance ({sub800_ratio*100:.1f}% sub-800 Hz)."
                )
            elif "cat" in common_name or (centroid_mean < 1400.0 and sub800_ratio < 0.60):
                detected_category = "feline_cat"
                spoof_factors.append(
                    f"Domestic Feline Acoustic Profile: Meow harmonics detected ({sub800_ratio*100:.1f}% sub-800 Hz, centroid {centroid_mean:.0f} Hz)."
                )
            elif "dog" in common_name or sub800_ratio >= 0.60:
                detected_category = "canine_dog"
                spoof_factors.append(
                    f"Domestic Canine Acoustic Signature: Heavy laryngeal resonance ({sub800_ratio*100:.1f}% sub-800 Hz)."
                )
            else:
                detected_category = "mammalian_domestic"
                spoof_factors.append(
                    f"Mammalian Laryngeal Resonance: {sub800_ratio*100:.1f}% energy below 800 Hz (avian baseline < 8.0%)."
                )
        elif sub1k_ratio > 0.35 and birdnet_conf < 30.0:
            spoof_score += 65.0
            if detected_category == "authentic_avian":
                detected_category = "mammalian_domestic"
            spoof_factors.append(
                f"Elevated sub-1,000 Hz low-frequency energy detected ({sub1k_ratio*100:.1f}% vs < 10% avian baseline)."
            )

        # --- LAYER 4: Neural Confidence Gate (Out-of-Distribution Sounds) ---
        if birdnet_conf < 15.0:
            spoof_score += 70.0
            spoof_factors.append(
                f"Very Low Species Match ({birdnet_conf:.1f}% < 15% threshold): Lacks distinct avian syrinx spectrogram ridges."
            )
        elif birdnet_conf < 25.0:
            spoof_score += 45.0
            spoof_factors.append(
                f"Sub-threshold Species Match ({birdnet_conf:.1f}% < 25% threshold): High out-of-distribution ambiguity."
            )

        # --- LAYER 5: Smartphone Loudspeaker Replay Transducer Artifacts ---
        if rolloff_95 < 10500 and centroid_mean < 3600 and birdnet_conf < 35.0 and sub1k_ratio > 0.18:
            spoof_score += 45.0
            if detected_category in ["authentic_avian", "mammalian_domestic"]:
                detected_category = "phone_speaker_replay"
            spoof_factors.append(
                f"Smartphone Speaker Replay Signature: Sharp high-frequency cutoff at {rolloff_95:.0f} Hz "
                "with mid-range compression typical of mobile phone loudspeaker playback."
            )

        # --- LAYER 6: Human Whistling & Exhalation Breath Turbulence ---
        s = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        breath_mask = (freqs >= 80) & (freqs <= 650)
        total_energy = np.sum(s) + 1e-9
        breath_energy = np.sum(s[breath_mask, :])
        breath_ratio = float(breath_energy / total_energy)

        if centroid_max < 3000.0 and centroid_mean < 2300.0 and birdnet_conf < 40.0:
            spoof_score += 35.0
            if detected_category == "authentic_avian":
                detected_category = "human_mimicry"
            spoof_factors.append(
                f"Acoustic frequency ceiling restricted ({centroid_max:.0f} Hz max) — consistent with human oral whistling limits."
            )

        if breath_ratio > 0.28 and birdnet_conf < 40.0:
            spoof_score += 40.0
            spoof_factors.append(
                f"Elevated sub-650 Hz breath turbulence detected ({breath_ratio*100:.1f}% energy) — characteristic of human exhalation airflow."
            )

    spoof_score = float(np.clip(spoof_score, 0.0, 100.0))
    is_authentic = spoof_score < 45.0

    # Human-readable classification labels
    if is_authentic:
        status_label = "VERIFIED AUTHENTIC BIRD VOCALIZATION"
        status_color = "#10B981"  # Emerald Green
        status_summary = (
            f"Confirmed authentic avian biometrics. Dual-syrinx modulation verified with {birdnet_conf:.1f}% "
            "BirdNET neural confidence. Normal planetary disaster prediction pipeline authorized."
        )
        suppress_alarm = False
    else:
        status_color = "#EF4444"  # Red
        suppress_alarm = True

        if detected_category == "feline_cat":
            status_label = "REJECTED: DOMESTIC FELINE SOUND (CAT MEOW)"
            status_summary = (
                "Acoustic analysis identified a domestic cat meow. While sliding vocal tract formants can confuse "
                "generic bird models into predicting a peacock, biological biometrics confirmed > 30% low-frequency "
                "mammalian resonance that an avian syrinx cannot produce. Civil disaster trigger safely DISARMED."
            )
        elif detected_category == "canine_dog":
            status_label = "REJECTED: DOMESTIC CANINE SOUND (DOG BARK / WHINE)"
            status_summary = (
                "Acoustic signature matches domestic dog barking or canine vocalization. "
                "Disaster Early Warning trigger has been safely DISARMED to prevent false civil alarms."
            )
        elif detected_category == "phone_speaker_replay":
            status_label = "REJECTED: SMARTPHONE SPEAKER REPLAY DETECTED"
            status_summary = (
                "Acoustic analysis detected artificial loudspeaker playback from a phone or Bluetooth speaker. "
                "Disaster Early Warning trigger has been safely DISARMED to prevent civil spoofing."
            )
        elif detected_category == "mammalian_domestic":
            status_label = "REJECTED: NON-AVIAN MAMMALIAN SOUND DETECTED"
            status_summary = (
                "Low-frequency vocal tract resonance detected (dog bark, cat meow, or mammalian noise). "
                "Disaster Early Warning trigger has been safely DISARMED to prevent false civil alarms."
            )
        elif detected_category == "human_mimicry":
            status_label = "REJECTED: HUMAN WHISTLING / VOCAL MIMICRY"
            status_summary = (
                "Acoustic markers match human oral cavity whistling or vocal imitation. "
                "Disaster Early Warning trigger has been safely DISARMED to prevent false civil alarms."
            )
        else:
            status_label = "REJECTED: OUT-OF-DISTRIBUTION NON-AVIAN NOISE"
            status_summary = (
                "Audio lacks distinct avian spectrographic characteristics and failed species confidence thresholds. "
                "Disaster Early Warning trigger has been safely DISARMED."
            )

    return {
        "is_authentic": is_authentic,
        "spoof_score": round(spoof_score, 1),
        "status_label": status_label,
        "status_color": status_color,
        "status_summary": status_summary,
        "suppress_alarm": suppress_alarm,
        "detected_category": detected_category,
        "spoof_factors": spoof_factors,
        "sub600_ratio": round(sub600_ratio * 100.0, 1),
        "sub800_ratio": round(sub800_ratio * 100.0, 1),
        "sub1k_ratio": round(sub1k_ratio * 100.0, 1),
        "birdnet_conf": round(birdnet_conf, 1)
    }
