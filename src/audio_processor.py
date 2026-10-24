"""Audio processing and bioacoustic feature extraction module.

Standardizes input signals and extracts acoustic descriptors used by the
distress analysis engine, using voice activity masking to ignore silence.
"""

from typing import Dict, Any, Tuple
import io
import numpy as np
import librosa
import soundfile as sf


TARGET_SR = 22050


def load_audio(
    file_source: Any,
    target_sr: int = TARGET_SR,
    max_duration_sec: float = 60.0
) -> Tuple[np.ndarray, int, float]:
    """Load an audio file or bytes buffer into a standardized mono numpy array.
    
    Args:
        file_source: File path (str) or file-like bytes buffer.
        target_sr: Target sample rate for analysis (default 22050 Hz).
        max_duration_sec: Max audio duration to analyze to prevent memory spikes.
        
    Returns:
        Tuple of (y_mono, sr, duration_seconds)
    """
    if isinstance(file_source, (bytes, bytearray)):
        file_source = io.BytesIO(file_source)

    y, sr = librosa.load(file_source, sr=target_sr, mono=True, duration=max_duration_sec)
    
    # Peak normalization to prevent volume bias
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
        
    duration = len(y) / sr
    return y, sr, duration


def extract_acoustic_features(y: np.ndarray, sr: int = TARGET_SR) -> Dict[str, Any]:
    """Extract bioacoustic features correlated with bird vocalization distress.
    
    Uses vocal activity masking so silent intervals between calls do not skew
    spectral flatness or centroid calculations.
    """
    if len(y) == 0:
        raise ValueError("Audio buffer is empty.")

    hop_length = 512
    n_fft = 2048

    # 1. Mel-Spectrogram (128 mel bins)
    s_mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128, fmin=200, fmax=10000
    )
    s_mel_db = librosa.power_to_db(s_mel, ref=np.max)

    # 2. RMS Energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    max_rms = float(np.max(rms)) if len(rms) > 0 else 1.0

    # Vocal Activity Mask: Focus spectral analysis on frames with actual bird audio (>5% peak RMS)
    active_mask = rms > (0.05 * max_rms)
    if not np.any(active_mask):
        active_mask = np.ones_like(rms, dtype=bool)

    # 3. Spectral Centroid (Hz)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    active_centroid = centroid[active_mask]
    
    # 4. Spectral Rolloff (85% energy threshold)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)[0]
    active_rolloff = rolloff[active_mask]

    # 5. Spectral Flatness (measure of noise/screech harshness vs harmonic purity)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
    active_flatness = flatness[active_mask]

    # 6. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
    active_zcr = zcr[active_mask]

    # 7. Onset Detection (Call Bursts / Chirp Events)
    # Energy onset detection tailored for bird calls
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, backtrack=True, wait=4, delta=0.07
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    duration = len(y) / sr
    call_rate = len(onset_frames) / max(duration, 0.1)

    # Time axes for plotting
    times = librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=hop_length)

    return {
        "duration": duration,
        "sample_rate": sr,
        "times": times,
        "mel_db": s_mel_db,
        "spectral_centroid": {
            "mean": float(np.mean(active_centroid)),
            "std": float(np.std(active_centroid)),
            "max": float(np.max(active_centroid)),
            "trajectory": centroid
        },
        "spectral_rolloff": {
            "mean": float(np.mean(active_rolloff)),
            "trajectory": rolloff
        },
        "spectral_flatness": {
            "mean": float(np.mean(active_flatness)),
            "trajectory": flatness
        },
        "rms_energy": {
            "mean": float(np.mean(rms)),
            "std": float(np.std(rms)),
            "max": max_rms,
            "trajectory": rms
        },
        "zcr": {
            "mean": float(np.mean(active_zcr)),
            "trajectory": zcr
        },
        "call_rate": float(call_rate),
        "total_onsets": int(len(onset_frames)),
        "onset_times": onset_times
    }
