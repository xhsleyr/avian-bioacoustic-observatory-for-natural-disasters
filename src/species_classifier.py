"""Global Bird Species Classifier powered by Cornell BirdNET v2.4 (6,522 Species).

Performs audio chunking and inference using ONNX Runtime to identify bird species
across 6,500+ global species, combined with disaster-sensitivity heuristics.
"""

from typing import Dict, Any, List, Tuple
import os
import io
import numpy as np
import librosa
import onnxruntime as ort

MODEL_PATH = "models/birdnet_model.onnx"
LABELS_PATH = "models/BirdNET_GLOBAL_6K_V2.4_Labels.txt"
BIRDNET_SR = 48000
WINDOW_SIZE = 144000  # 3 seconds @ 48kHz

_SESSION = None
_LABELS = None


def get_birdnet_session() -> Tuple[ort.InferenceSession, List[str]]:
    """Lazy-load the ONNX session and 6,522 species labels."""
    global _SESSION, _LABELS
    if _SESSION is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(
                f"BirdNET model or labels missing at {MODEL_PATH} / {LABELS_PATH}"
            )
        # Fast & quiet CPU execution on Apple Silicon M1
        _SESSION = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            _LABELS = [line.strip() for line in f if line.strip()]

    return _SESSION, _LABELS


def infer_disaster_sensitivity(common_name: str, scientific_name: str) -> str:
    """Infer known disaster-sensing behavioral traits from avian family taxonomy."""
    name_lower = (common_name + " " + scientific_name).lower()

    if any(k in name_lower for k in ["owl", "tyto", "strig"]):
        return "Nocturnal hearing acutely tuned to low-frequency ground vibrations and seismic P-waves."
    elif any(k in name_lower for k in ["pigeon", "dove", "columb"]):
        return "Proven infrasonic sensitivity (<10 Hz); leaves nesting sites en masse hours prior to cyclone and tsunami arrivals."
    elif any(k in name_lower for k in ["crow", "raven", "corv", "jay", "magpie"]):
        return "Highly intelligent sentinel flock species; forms loud mobbing aggregations and circles in high updrafts before earthquakes."
    elif any(k in name_lower for k in ["starling", "sturn", "myna"]):
        return "Demonstrates collective flock murmuration panic upon sensing shockwaves and sudden atmospheric barometric drops."
    elif any(k in name_lower for k in ["thrush", "robin", "turd"]):
        return "Historically documented displaying sudden singing silence followed by high-frequency distress chatter prior to seismic tremors."
    elif any(k in name_lower for k in ["sparrow", "passer", "finch"]):
        return "Acutely sensitive to micro-barometric air pressure fluctuations preceding flash floods and severe tornadic storm fronts."
    elif any(k in name_lower for k in ["heron", "egret", "crane", "stork", "waterfowl", "duck", "goose"]):
        return "Wetland bio-indicator; exhibits abrupt flight away from shorelines prior to storm surges and coastal tsunamis."
    elif any(k in name_lower for k in ["gull", "tern", "petrel", "shearwater"]):
        return "Pelagic oceanic bird; senses barometric pressure gradients and avoids maritime cyclone centers."
    elif any(k in name_lower for k in ["peafowl", "pavo", "pheasant", "quail"]):
        return "Ground-dwelling gallinaceous bird; famous for loud, frantic pre-tremor screeching calls triggered by sub-audible earth vibrations."
    else:
        return "Avian environmental indicator; collective agitation correlates with approaching weather fronts and environmental shockwaves."


def identify_species(
    file_source: Any,
    acoustic_features: Dict[str, Any] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """Identify bird species using the global Cornell BirdNET 6,522 model.
    
    Args:
        file_source: Audio file path or byte buffer.
        acoustic_features: Extracted features from audio_processor (optional).
        top_k: Number of top candidate species to return.
        
    Returns:
        Dictionary containing primary species, confidence, candidate list,
        and disaster sensitivity intelligence.
    """
    session, labels = get_birdnet_session()

    if isinstance(file_source, (bytes, bytearray)):
        file_source = io.BytesIO(file_source)

    # Load audio at 48kHz for BirdNET
    audio, sr = librosa.load(file_source, sr=BIRDNET_SR, mono=True)
    if len(audio) == 0:
        raise ValueError("Audio buffer is empty.")

    # Create 3-second windows with 50% overlap (1.5s step)
    step_size = WINDOW_SIZE // 2
    windows = []
    
    if len(audio) < WINDOW_SIZE:
        padded = np.pad(audio, (0, WINDOW_SIZE - len(audio)))
        windows.append(padded)
    else:
        for start in range(0, len(audio) - WINDOW_SIZE + 1, step_size):
            windows.append(audio[start:start + WINDOW_SIZE])
        if len(windows) == 0:
            windows.append(audio[:WINDOW_SIZE])

    windows_arr = np.array(windows, dtype=np.float32)

    # Run inference through ONNX session
    logits = session.run(['Identity_0'], {'INPUT': windows_arr})[0]
    
    # Sigmoid activation to convert logits to confidence probabilities
    probs = 1.0 / (1.0 + np.exp(-logits))

    # Aggregate window predictions: Max pooling across time windows
    max_probs = np.max(probs, axis=0)

    # Top-K candidate indices
    top_indices = np.argsort(max_probs)[::-1][:top_k]

    candidates = []
    for idx in top_indices:
        raw_label = labels[idx]
        if "_" in raw_label:
            sci_name, com_name = raw_label.split("_", 1)
        else:
            sci_name = raw_label
            com_name = raw_label

        conf = float(max_probs[idx] * 100.0)
        candidates.append({
            "common_name": com_name.strip(),
            "scientific_name": sci_name.strip(),
            "confidence": round(conf, 1),
            "disaster_sensitivity": infer_disaster_sensitivity(com_name, sci_name)
        })

    primary = candidates[0]

    # Baseline deviation check if features available
    detected_pitch = acoustic_features["spectral_centroid"]["mean"] if acoustic_features else 3000.0
    if detected_pitch > 5000:
        pitch_status = "High Pitch Elevation / Screech Mode"
    elif detected_pitch > 3500:
        pitch_status = "Elevated Alert Pitch"
    else:
        pitch_status = "Normal Harmonic Song Band"

    return {
        "model_name": "Cornell BirdNET Global v2.4 (6,522 Species)",
        "total_species_supported": 6522,
        "primary_species": {
            "common_name": primary["common_name"],
            "scientific_name": primary["scientific_name"],
            "confidence": primary["confidence"],
            "disaster_sensitivity": primary["disaster_sensitivity"],
            "pitch_status": pitch_status,
            "detected_pitch": round(detected_pitch, 0)
        },
        "top_candidates": candidates
    }
