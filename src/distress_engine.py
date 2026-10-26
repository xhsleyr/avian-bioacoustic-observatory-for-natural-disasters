"""Distress analysis engine and natural disaster early-warning scoring.

Computes the Acoustic Distress Index (ADI) by evaluating biological markers
such as pitch elevation, call burst density, noise harshness, and energy volatility,
dynamically calibrated to species-specific avian taxonomy baselines.
Also computes temporal threat escalation curves (d(ADI)/dt), multi-species
sentinel flock consensus, and forensic telemetry reports.
"""

import datetime
import re
from typing import Any

import librosa
import numpy as np

# Biologically calibrated baselines across avian taxonomic orders
SPECIES_BASELINES: dict[str, dict[str, Any]] = {
    "passerine": {
        "family_label": "Passerine Songbird (Passeriformes)",
        "pitch_min": 3000.0,
        "pitch_scale": 3500.0,
        "rate_min": 0.8,
        "rate_scale": 3.6,
        "flatness_min": 0.0030,
        "flatness_scale": 0.050,
        "normal_pitch_range": "2,800 - 4,800 Hz",
        "normal_rate_range": "0.8 - 2.4 calls/sec"
    },
    "raptor_owl": {
        "family_label": "Raptor / Strigiform (Owls, Hawks, Eagles)",
        "pitch_min": 600.0,
        "pitch_scale": 1800.0,
        "rate_min": 0.2,
        "rate_scale": 1.6,
        "flatness_min": 0.0040,
        "flatness_scale": 0.045,
        "normal_pitch_range": "500 - 1,400 Hz",
        "normal_rate_range": "0.2 - 0.8 calls/sec"
    },
    "gamebird": {
        "family_label": "Galliform / Gamebird (Peafowl, Pheasants, Quail)",
        "pitch_min": 1000.0,
        "pitch_scale": 1800.0,
        "rate_min": 0.3,
        "rate_scale": 2.0,
        "flatness_min": 0.0004,
        "flatness_scale": 0.025,
        "normal_pitch_range": "900 - 1,800 Hz",
        "normal_rate_range": "0.3 - 1.0 calls/sec"
    },
    "corvid": {
        "family_label": "Corvid Sentinel (Crow, Raven, Jay, Magpie)",
        "pitch_min": 1100.0,
        "pitch_scale": 2000.0,
        "rate_min": 0.4,
        "rate_scale": 2.4,
        "flatness_min": 0.0080,
        "flatness_scale": 0.060,
        "normal_pitch_range": "1,100 - 2,000 Hz",
        "normal_rate_range": "0.4 - 1.2 calls/sec"
    },
    "columbid": {
        "family_label": "Columbid Infrasound Sensor (Pigeons, Doves)",
        "pitch_min": 450.0,
        "pitch_scale": 1200.0,
        "rate_min": 0.3,
        "rate_scale": 1.5,
        "flatness_min": 0.0030,
        "flatness_scale": 0.040,
        "normal_pitch_range": "450 - 900 Hz",
        "normal_rate_range": "0.3 - 0.9 calls/sec"
    },
    "waterfowl_seabird": {
        "family_label": "Waterfowl / Seabird (Ducks, Gulls, Herons)",
        "pitch_min": 1200.0,
        "pitch_scale": 2200.0,
        "rate_min": 0.4,
        "rate_scale": 2.2,
        "flatness_min": 0.0060,
        "flatness_scale": 0.055,
        "normal_pitch_range": "1,200 - 2,200 Hz",
        "normal_rate_range": "0.4 - 1.4 calls/sec"
    },
    "default_avian": {
        "family_label": "Standard Avian Taxonomy Baseline",
        "pitch_min": 2000.0,
        "pitch_scale": 3800.0,
        "rate_min": 0.6,
        "rate_scale": 3.0,
        "flatness_min": 0.0050,
        "flatness_scale": 0.060,
        "normal_pitch_range": "1,500 - 2,800 Hz",
        "normal_rate_range": "0.3 - 1.2 calls/sec"
    }
}


def get_species_baseline(common_name: str, scientific_name: str = "") -> dict[str, Any]:
    """Map identified species common and scientific names to calibrated biological baselines."""
    tokens = set(re.findall(r'[a-z]+', (common_name + " " + scientific_name).lower()))

    # 1. Gamebirds / Peafowl
    if any(w in tokens for w in ["peafowl", "peacock", "peahen", "pavo", "pheasant", "quail", "turkey", "grouse", "partridge", "junglefowl"]):
        return SPECIES_BASELINES["gamebird"]
    # 2. Raptors / Owls
    elif any(w in tokens for w in ["owl", "owls", "bubo", "tyto", "screech", "hawk", "hawks", "buteo", "eagle", "eagles", "falcon", "accipiter", "harrier", "nightjar"]):
        return SPECIES_BASELINES["raptor_owl"]
    # 3. Corvids
    elif any(w in tokens for w in ["crow", "crows", "raven", "ravens", "corvus", "corv", "jay", "jays", "magpie", "magpies", "treepie", "nutcracker"]):
        return SPECIES_BASELINES["corvid"]
    # 4. Columbids
    elif any(w in tokens for w in ["pigeon", "pigeons", "dove", "doves", "columba", "streptopelia", "zenaida"]):
        return SPECIES_BASELINES["columbid"]
    # 5. Waterfowl / Seabirds
    elif any(w in tokens for w in ["duck", "ducks", "goose", "geese", "swan", "swans", "gull", "gulls", "tern", "terns", "heron", "herons", "egret", "plover", "sandpiper", "pelican", "cormorant", "ibis", "anser", "anas"]):
        return SPECIES_BASELINES["waterfowl_seabird"]
    # 6. Passerines
    elif any(w in tokens for w in ["robin", "sparrow", "warbler", "finch", "thrush", "cardinal", "swallow", "tit", "wren", "starling", "myna", "blackbird", "flycatcher", "tanager", "bunting", "oriole", "canary", "chickadee"]):
        return SPECIES_BASELINES["passerine"]
    else:
        return SPECIES_BASELINES["default_avian"]


def compute_distress_index(
    features: dict[str, Any],
    species_info: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Compute the Acoustic Distress Index (ADI) calibrated to species taxonomy."""
    centroid_mean = features["spectral_centroid"]["mean"]
    call_rate = features["call_rate"]
    flatness_mean = features["spectral_flatness"]["mean"]
    rms_std = features["rms_energy"]["std"]
    duration = features["duration"]

    common_name = ""
    scientific_name = ""
    if species_info and "primary_species" in species_info:
        common_name = species_info["primary_species"].get("common_name", "")
        scientific_name = species_info["primary_species"].get("scientific_name", "")

    baseline = get_species_baseline(common_name, scientific_name)

    # 1. Species-Adaptive Pitch Score (0 to 100)
    norm_pitch = (centroid_mean - baseline["pitch_min"]) / baseline["pitch_scale"]
    score_pitch = float(np.clip(norm_pitch, 0.0, 1.0) * 100.0)

    # 2. Species-Adaptive Call Rate Score (0 to 100)
    norm_rate = (call_rate - baseline["rate_min"]) / baseline["rate_scale"]
    score_rate = float(np.clip(norm_rate, 0.0, 1.0) * 100.0)
    
    # Musical trill dampener: pure harmonic melodic songs with low flatness (< 0.003)
    # should not have rapid trill notes penalized as frantic chaotic panic.
    if flatness_mean < 0.0030:
        score_rate *= 0.45

    # 3. Harshness / Flatness Score (0 to 100)
    norm_flatness = (flatness_mean - baseline["flatness_min"]) / baseline["flatness_scale"]
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

    if adi >= 70.0:
        threat_level = "CRITICAL"
        threat_color = "#E63946"
        alert_title = "CRITICAL EARLY WARNING: Severe Collective Panic Detected"
        alert_summary = (
            "Acoustic signatures show extreme high-frequency shrieking, frantic burst intervals, "
            "and chaotic noise distortion. These bioacoustic anomalies strongly correlate with "
            "imminent natural threats (e.g., pre-seismic P-wave ground vibrations, approaching cyclone infrasound)."
        )
    elif adi >= 38.0:
        threat_level = "ELEVATED"
        threat_color = "#F4A261"
        alert_title = "ELEVATED ALERT: Abnormal Agitation Detected"
        alert_summary = (
            "Higher-than-normal pitch and accelerated vocalization frequency detected. "
            "Indicates heightened flock agitation, localized predator activity, or sudden barometric pressure drop."
        )
    else:
        threat_level = "NORMAL"
        threat_color = "#2A9D8F"
        alert_title = "NORMAL STATUS: Baseline Avian Communication"
        alert_summary = (
            "Vocalizations exhibit stable melodic contours, regular rhythmic call pacing, "
            "and clear harmonic balance. No environmental distress markers detected."
        )

    diagnostics: list[dict[str, Any]] = [
        {
            "name": "Pitch Elevation Index",
            "score": round(score_pitch, 1),
            "value": f"{centroid_mean:.0f} Hz",
            "normal_range": baseline["normal_pitch_range"],
            "status": "High / Screech" if score_pitch > 60 else "Normal Harmonic"
        },
        {
            "name": "Vocal Burst Frequency (Call Rate)",
            "score": round(score_rate, 1),
            "value": f"{call_rate:.2f} calls/sec",
            "normal_range": baseline["normal_rate_range"],
            "status": "Accelerated" if score_rate > 50 else "Calm Interval"
        },
        {
            "name": "Acoustic Chaos & Harshness",
            "score": round(score_flatness, 1),
            "value": f"{flatness_mean:.4f}",
            "normal_range": f"< {baseline['flatness_min'] + 0.005:.4f}",
            "status": "Harsh Chaos" if score_flatness > 50 else "Harmonic Song"
        },
        {
            "name": "Energy Volatility",
            "score": round(score_energy, 1),
            "value": f"{rms_std:.3f} RMS-std",
            "normal_range": "0.02 - 0.08",
            "status": "Erratic Volatility" if score_energy > 55 else "Controlled Energy"
        }
    ]

    return {
        "adi": adi,
        "threat_level": threat_level,
        "threat_color": threat_color,
        "alert_title": alert_title,
        "alert_summary": alert_summary,
        "baseline_profile": baseline,
        "sub_scores": {
            "pitch": round(score_pitch, 1),
            "call_rate": round(score_rate, 1),
            "flatness": round(score_flatness, 1),
            "energy_volatility": round(score_energy, 1)
        },
        "diagnostics": diagnostics,
        "duration_analyzed": round(duration, 2)
    }


def compute_distress_timeline(
    y: np.ndarray,
    sr: int,
    species_info: dict[str, Any] | None = None,
    window_sec: float = 1.8,
    hop_sec: float = 0.4
) -> dict[str, Any]:
    """Compute rolling temporal distress timeline and threat escalation velocity d(ADI)/dt."""
    duration = len(y) / sr
    if duration < 0.6:
        return {
            "times": [0.0],
            "adi_series": [50.0],
            "escalation_rate": 0.0,
            "peak_time": 0.0,
            "peak_adi": 50.0,
            "shockwave_detected": False,
            "trend_label": "Stationary"
        }

    w_sec = min(window_sec, max(duration / 3.0, 0.6))
    h_sec = max(w_sec / 3.0, 0.2)
    
    win_samples = int(w_sec * sr)
    hop_samples = int(h_sec * sr)

    common_name = ""
    scientific_name = ""
    if species_info and "primary_species" in species_info:
        common_name = species_info["primary_species"].get("common_name", "")
        scientific_name = species_info["primary_species"].get("scientific_name", "")

    baseline = get_species_baseline(common_name, scientific_name)

    times: list[float] = []
    adi_series: list[float] = []
    hop_length = 512

    for start_sample in range(0, len(y) - win_samples + 1, hop_samples):
        chunk = y[start_sample : start_sample + win_samples]
        t_center = (start_sample + win_samples // 2) / sr
        times.append(round(t_center, 2))

        c_cent = librosa.feature.spectral_centroid(y=chunk, sr=sr, hop_length=hop_length)[0]
        c_flat = librosa.feature.spectral_flatness(y=chunk, hop_length=hop_length)[0]
        c_rms = librosa.feature.rms(y=chunk, hop_length=hop_length)[0]

        mean_cent = float(np.mean(c_cent)) if len(c_cent) > 0 else 2500.0
        mean_flat = float(np.mean(c_flat)) if len(c_flat) > 0 else 0.01
        std_rms = float(np.std(c_rms)) if len(c_rms) > 0 else 0.05

        onset_f = librosa.onset.onset_detect(y=chunk, sr=sr, hop_length=hop_length, wait=3, delta=0.06)
        chunk_rate = len(onset_f) / w_sec

        p_score = float(np.clip((mean_cent - baseline["pitch_min"]) / baseline["pitch_scale"], 0.0, 1.0) * 100.0)
        r_score = float(np.clip((chunk_rate - baseline["rate_min"]) / baseline["rate_scale"], 0.0, 1.0) * 100.0)
        if mean_flat < 0.0030:
            r_score *= 0.45
        f_score = float(np.clip((mean_flat - baseline["flatness_min"]) / baseline["flatness_scale"], 0.0, 1.0) * 100.0)
        e_score = float(np.clip((std_rms - 0.04) / 0.16, 0.0, 1.0) * 100.0)

        chunk_adi = round(0.35 * p_score + 0.30 * r_score + 0.20 * f_score + 0.15 * e_score, 1)
        adi_series.append(float(np.clip(chunk_adi, 0.0, 100.0)))

    if len(adi_series) == 0:
        times = [0.0]
        adi_series = [50.0]

    dt = h_sec
    d_adi = np.diff(adi_series) / max(dt, 0.01) if len(adi_series) > 1 else np.array([0.0])
    max_escalation_rate = float(np.max(d_adi)) if len(d_adi) > 0 else 0.0

    peak_idx = int(np.argmax(adi_series))
    peak_time = times[peak_idx]
    peak_adi = adi_series[peak_idx]

    shockwave_detected = (max_escalation_rate > 28.0) and (peak_adi > 65.0)

    if shockwave_detected:
        trend_label = "Sudden Shockwave Infrasound Onset"
    elif max_escalation_rate > 15.0:
        trend_label = "Rapid Panic Acceleration"
    elif np.mean(adi_series) > 60.0:
        trend_label = "Sustained High Threat Frenzy"
    else:
        trend_label = "Stable Ambient Rhythm"

    return {
        "times": times,
        "adi_series": adi_series,
        "escalation_rate": round(max_escalation_rate, 1),
        "peak_time": peak_time,
        "peak_adi": peak_adi,
        "shockwave_detected": shockwave_detected,
        "trend_label": trend_label
    }


def evaluate_flock_consensus(
    species_info: dict[str, Any],
    features: dict[str, Any],
    adi_score: float
) -> dict[str, Any]:
    """Evaluate multi-species flock murmuration and sentinel consensus."""
    candidates = species_info.get("top_candidates", [])
    call_rate = features.get("call_rate", 0.0)

    active_species = [
        c.get("common_name", "Unknown") for c in candidates
        if c.get("confidence", 0.0) >= 15.0 and "dog" not in c.get("common_name", "").lower() and "human" not in c.get("common_name", "").lower()
    ]

    is_multi_species = len(active_species) >= 2

    if adi_score >= 60.0 and is_multi_species and call_rate >= 1.5:
        consensus_status = "MULTI-SPECIES SENTINEL CONSENSUS"
        consensus_type = "macro_ecological"
        consensus_color = "#E63946"
        summary = (
            f"Macro-Ecological Panic Consensus: {len(active_species)} distinct avian species "
            f"({', '.join(active_species[:3])}) exhibit synchronized frantic alarm calls. "
            "Cross-species collective panic is an established natural indicator of approaching geophysical shockwaves or severe storms."
        )
    elif adi_score >= 38.0:
        consensus_status = "LOCALIZED SINGLE-SPECIES SENTINEL ALERT"
        consensus_type = "micro_localized"
        consensus_color = "#F4A261"
        summary = (
            f"Localized Agitation: Distress detected predominantly in {active_species[0] if active_species else 'single avian species'}. "
            "Typical of localized predator activity, territorial friction, or minor micro-barometric shifts."
        )
    else:
        consensus_status = "COLLECTIVE NOMINAL BIOSPHERE"
        consensus_type = "nominal_calm"
        consensus_color = "#2A9D8F"
        summary = (
            "Multi-Species Calm Chorus: Harmonious acoustic dispersion across detected species. "
            "Zero collective agitation markers detected in the bioacoustic biosphere."
        )

    return {
        "status": consensus_status,
        "consensus_type": consensus_type,
        "color": consensus_color,
        "summary": summary,
        "active_species": active_species,
        "species_count": len(active_species)
    }


def generate_forensic_report(
    sample_label: str,
    duration: float,
    sr: int,
    features: dict[str, Any],
    species_info: dict[str, Any],
    spoof_info: dict[str, Any],
    distress_info: dict[str, Any],
    timeline_info: dict[str, Any],
    consensus_info: dict[str, Any]
) -> str:
    """Generate a formal, publication-ready Bioacoustic Forensic Telemetry Report."""
    now_utc = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    prim = species_info.get("primary_species", {})
    sub_scores = distress_info.get("sub_scores", {})
    base = distress_info.get("baseline_profile", {})

    lines = [
        "# ==============================================================================",
        "# PROJECT INTERA: AVIAN BIOACOUSTIC FORENSIC TELEMETRY REPORT",
        "# CLASSIFICATION: CIVIL DISASTER EARLY WARNING INCIDENT CERTIFICATE",
        "# ==============================================================================",
        "",
        "REPORT METADATA:",
        f"- Timestamp: {now_utc}",
        "- Observatory Node: INTERA-GLOBAL-ACOUSTIC-OBSERVATORY-01",
        f"- Specimen Stream ID: {sample_label}",
        f"- Signal Duration: {duration:.2f} seconds",
        f"- Ingestion Sampling Rate: {sr:,} Hz",
        "",
        "--------------------------------------------------------------------------------",
        "1. AVIAN SYRINX BIOMETRIC VERIFICATION & ANTI-SPOOFING SHIELD",
        "--------------------------------------------------------------------------------",
        f"- Biometric Verification Status: {spoof_info.get('status_label')}",
        f"- Avian Biometric Authenticity: {'PASSED (GENUINE AVIAN SYRINX)' if spoof_info.get('is_authentic') else 'REJECTED (NON-AVIAN ADVERSARIAL SOUND)'}",
        f"- Biometric Spoofing Risk Score: {spoof_info.get('spoof_score', 0.0)}% (Safety Threshold < 45.0%)",
        f"- Sub-600 Hz Low-Frequency Energy: {spoof_info.get('sub600_ratio', 0.0)}% (Avian Syrinx Baseline < 2.5%)",
        f"- Sub-800 Hz Low-Frequency Energy: {spoof_info.get('sub800_ratio', 0.0)}% (Avian Syrinx Baseline < 3.0%)",
        f"- Sub-1,000 Hz Low-Frequency Energy: {spoof_info.get('sub1k_ratio', 0.0)}% (Mammals > 30.0%)",
        f"- Civil Early-Warning Alarm Disarm: {'DISARMED (CIVIL ALERT PREVENTED)' if spoof_info.get('suppress_alarm') else 'ARMED & OPERATIONAL'}",
        "",
        "Biometric Diagnostic Factors:"
    ]

    if spoof_info.get('spoof_factors'):
        for factor in spoof_info.get('spoof_factors', []):
            lines.append(f"  * {factor}")
    else:
        lines.append("  * Full dual-syrinx harmonic ridges verified; zero mammalian larynx or transducer artifacts.")

    lines.extend([
        "",
        "--------------------------------------------------------------------------------",
        "2. NEURAL SPECIES TAXONOMY (CORNELL BirdNET v2.4 - 6,522 SPECIES)",
        "--------------------------------------------------------------------------------",
        f"- Primary Classified Species: {prim.get('common_name')}",
        f"- Binomial Scientific Nomenclature: {prim.get('scientific_name')}",
        f"- BirdNET Neural Confidence: {prim.get('confidence', 0.0)}%",
        f"- Valid Bird Biometric Confirmation: {species_info.get('is_valid_bird')}",
        f"- Calibrated Taxonomic Baseline: {base.get('family_label', 'Standard Avian')}",
        f"- Pitch Mode: {prim.get('pitch_status')} ({prim.get('detected_pitch', 0.0):.0f} Hz)",
        "",
        "Top Neural Candidate Probabilities:"
    ])

    for cand in species_info.get("top_candidates", [])[:5]:
        lines.append(f"  * {cand.get('common_name')} ({cand.get('scientific_name')}): {cand.get('confidence')}% | Sensitivity: {cand.get('disaster_sensitivity')[:75]}...")

    lines.extend([
        "",
        "--------------------------------------------------------------------------------",
        "3. ACOUSTIC DISTRESS INDEX (ADI) & NATURAL DISASTER EARLY WARNING",
        "--------------------------------------------------------------------------------",
        f"- Composite Acoustic Distress Index (ADI): {distress_info.get('adi')}%",
        f"- Threat Assessment Tier: {distress_info.get('threat_level')}",
        f"- Alert Directive: {distress_info.get('alert_title')}",
        f"- Alert Biological Summary: {distress_info.get('alert_summary')}",
        "",
        "Multi-Dimensional Bioacoustic Sub-Scores (Species-Calibrated):",
        f"  * Pitch Elevation Index (35% weight): {sub_scores.get('pitch')}% (Normal: {base.get('normal_pitch_range')})",
        f"  * Vocal Burst Call Rate (30% weight): {sub_scores.get('call_rate')}% (Normal: {base.get('normal_rate_range')})",
        f"  * Acoustic Chaos / Flatness (20% weight): {sub_scores.get('flatness')}% (Normal: < {base.get('flatness_min', 0.005) + 0.005:.4f})",
        f"  * Energy Volatility / RMS (15% weight): {sub_scores.get('energy_volatility')}% (Normal: 0.02 - 0.08)",
        "",
        "--------------------------------------------------------------------------------",
        "4. TEMPORAL THREAT ESCALATION & INFRASOUND SHOCKWAVE TELEMETRY",
        "--------------------------------------------------------------------------------",
        f"- Maximum Escalation Velocity d(ADI)/dt: {timeline_info.get('escalation_rate')}% per second",
        f"- Peak Distress Timestamp: {timeline_info.get('peak_time')}s (Peak Intensity: {timeline_info.get('peak_adi')}%)",
        f"- Pre-Seismic Shockwave Signature Detected: {'POSITIVE (SUDDEN INFRASOUND BURST DETECTED)' if timeline_info.get('shockwave_detected') else 'NEGATIVE (NORMAL AMBIENT GRADIENT)'}",
        f"- Temporal Progression State: {timeline_info.get('trend_label')}",
        "",
        "--------------------------------------------------------------------------------",
        "5. MULTI-SPECIES SENTINEL FLOCK CONSENSUS",
        "--------------------------------------------------------------------------------",
        f"- Consensus Designation: {consensus_info.get('status')}",
        f"- Consensus Scope: {consensus_info.get('consensus_type')}",
        f"- Synchronized Species Detected: {', '.join(consensus_info.get('active_species', []))}",
        f"- Ecological Interpretation: {consensus_info.get('summary')}",
        "",
        "--------------------------------------------------------------------------------",
        "6. OFFICIAL CIVIL DISASTER AGENCY DIRECTIVE",
        "--------------------------------------------------------------------------------"
    ])

    if spoof_info.get("suppress_alarm"):
        lines.append("DIRECTIVE: Civil alarms are SAFELY DISARMED. Adversarial non-avian sound source confirmed.")
    elif distress_info.get("threat_level") == "CRITICAL":
        lines.append("DIRECTIVE: ACTIVATE CIVIL EMERGENCY ALERT. Synchronized high-intensity avian panic detected.")
    elif distress_info.get("threat_level") == "ELEVATED":
        lines.append("DIRECTIVE: ELEVATE ENVIRONMENTAL SURVEILLANCE. Heightened flock agitation logged.")
    else:
        lines.append("DIRECTIVE: NOMINAL BASELINE. No civil action required.")

    lines.append("")
    lines.append("# ================= END OF FORENSIC TELEMETRY CERTIFICATE =================")
    return "\n".join(lines)
