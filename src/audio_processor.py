"""Audio processing and bioacoustic feature extraction module.

Standardizes input signals and extracts acoustic descriptors used by the
distress analysis engine and biometric anti-spoofing shield, using voice
activity masking to ignore ambient silence and adaptive environmental denoising.
"""

import io
from typing import Any

import librosa
import numpy as np
import scipy.signal

TARGET_SR = 22050


def filter_environmental_noise(y: np.ndarray, sr: int, cutoff_hz: float = 90.0) -> np.ndarray:
    """Filter out sub-audible wind rumble, microphone handling noise, and AC line hum (50/60 Hz).
    
    Uses a gentle 2nd-order Butterworth high-pass filter at 90 Hz.
    This preserves the full mammalian larynx (energy from 200 Hz upwards)
    and avian syrinx (800 Hz to 10 kHz), while eliminating DC offset and low-frequency rumble.
    """
    if len(y) == 0:
        return y
    sos = scipy.signal.butter(2, cutoff_hz, btype='highpass', fs=sr, output='sos')
    filtered_arr = np.asarray(scipy.signal.sosfilt(sos, y), dtype=y.dtype)
    orig_peak = float(np.max(np.abs(y)))
    new_peak = float(np.max(np.abs(filtered_arr)))
    if new_peak > 0 and orig_peak > 0:
        filtered_arr = filtered_arr * (orig_peak / new_peak)
    return filtered_arr


def adaptive_spectral_denoise(
    y: np.ndarray,
    sr: int = TARGET_SR,
    noise_reduce_factor: float = 0.65,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """Apply adaptive spectral noise gating to suppress stationary ambient environmental noise
    (e.g., wind shear, light rain, leaf rustle) while preserving avian syringeal harmonics.
    """
    if len(y) < n_fft:
        return y
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    
    # Estimate noise floor using the quietest 15% energy frames
    frame_energies = np.sum(mag, axis=0)
    noise_threshold = float(np.percentile(frame_energies, 15))
    noise_mask = frame_energies <= max(noise_threshold, 1e-6)
    
    if np.any(noise_mask):
        noise_profile = np.mean(mag[:, noise_mask], axis=1, keepdims=True)
    else:
        noise_profile = np.min(mag, axis=1, keepdims=True)
        
    cleaned_mag = np.maximum(mag - (noise_reduce_factor * noise_profile), 0.15 * mag)
    cleaned_stft = cleaned_mag * np.exp(1j * phase)
    y_cleaned = librosa.istft(cleaned_stft, hop_length=hop_length, length=len(y))
    return np.asarray(y_cleaned, dtype=y.dtype)


def estimate_stereo_bearing(
    stereo_audio: np.ndarray,
    sr: int = TARGET_SR,
    mic_distance_meters: float = 0.20
) -> float | None:
    """Calculate the spatial bearing angle (azimuth in degrees: -90° to +90°)
    of an incoming acoustic wavefront using Generalized Cross-Correlation with Phase Transform (GCC-PHAT).
    """
    if stereo_audio.ndim != 2:
        return None
    if stereo_audio.shape[0] == 2:
        ch1, ch2 = stereo_audio[0], stereo_audio[1]
    elif stereo_audio.shape[1] == 2:
        ch1, ch2 = stereo_audio[:, 0], stereo_audio[:, 1]
    else:
        return None
        
    n = len(ch1)
    if n < 512:
        return None
        
    x1 = np.fft.rfft(ch1, n=n)
    x2 = np.fft.rfft(ch2, n=n)
    cc = x1 * np.conj(x2)
    denom = np.abs(cc)
    denom[denom < 1e-12] = 1e-12
    cc_norm = cc / denom
    r = np.fft.irfft(cc_norm, n=n)
    
    max_delay = int(np.ceil((mic_distance_meters / 343.0) * sr))
    r_window = np.concatenate((r[-max_delay:], r[:max_delay + 1]))
    delay_idx = int(np.argmax(r_window)) - max_delay
    
    tau = delay_idx / float(sr)
    sin_theta = (tau * 343.0) / max(mic_distance_meters, 1e-6)
    sin_theta = float(np.clip(sin_theta, -1.0, 1.0))
    azimuth_deg = float(np.degrees(np.arcsin(sin_theta)))
    return round(azimuth_deg, 1)


def load_audio(
    file_source: Any,
    target_sr: int = TARGET_SR,
    max_duration_sec: float = 60.0,
    apply_denoising: bool = True
) -> tuple[np.ndarray, int, float]:
    """Load an audio file or bytes buffer into a standardized mono numpy array.
    
    Args:
        file_source: File path (str) or file-like bytes buffer.
        target_sr: Target sample rate for analysis (default 22050 Hz).
        max_duration_sec: Max audio duration to analyze to prevent memory spikes.
        apply_denoising: Whether to apply environmental 90 Hz high-pass rumble removal.
        
    Returns:
        Tuple of (y_mono, sr, duration_seconds)
    """
    if isinstance(file_source, (bytes, bytearray)):
        file_source = io.BytesIO(file_source)

    y, sr_loaded = librosa.load(file_source, sr=target_sr, mono=True, duration=max_duration_sec)
    sr = int(sr_loaded)

    # Adaptive environmental wind/handling rumble filter
    if apply_denoising:
        y = filter_environmental_noise(y, sr, cutoff_hz=90.0)

    # Peak normalization to prevent volume bias
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
        
    duration = float(len(y) / sr)
    return y, sr, duration


def extract_acoustic_features(y: np.ndarray, sr: int = TARGET_SR) -> dict[str, Any]:
    """Extract bioacoustic features correlated with bird vocalization distress
    and non-avian mammalian / smartphone playback discrimination.
    
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

    # 2. RMS Energy & Vocal Activity Mask
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    max_rms = float(np.max(rms)) if len(rms) > 0 else 1.0

    active_mask = rms > (0.05 * max_rms)
    if not np.any(active_mask):
        active_mask = np.ones_like(rms, dtype=bool)

    # 3. Spectral Centroid (Hz)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    active_centroid = centroid[active_mask]
    
    # 4. Spectral Rolloff (85% and 95% energy thresholds)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)[0]
    active_rolloff = rolloff[active_mask]

    rolloff_95 = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.95)[0]
    active_rolloff_95 = rolloff_95[active_mask]

    # 5. Spectral Flatness (measure of noise/screech harshness vs harmonic purity)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
    active_flatness = flatness[active_mask]

    # 6. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
    active_zcr = zcr[active_mask]

    # 7. Linear FFT Low-Frequency Energy Ratios for Mammalian Larynx vs Avian Syrinx Discrimination
    s_linear = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    total_linear_energy = np.sum(s_linear) + 1e-9
    sub600_ratio = float(np.sum(s_linear[fft_freqs < 600, :]) / total_linear_energy)
    sub800_ratio = float(np.sum(s_linear[fft_freqs < 800, :]) / total_linear_energy)
    sub1k_ratio = float(np.sum(s_linear[fft_freqs < 1000, :]) / total_linear_energy)
    sub350_ratio = float(np.sum(s_linear[fft_freqs < 350, :]) / total_linear_energy)

    # 8. Onset Detection (Call Bursts / Chirp Events)
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
        "spectral_rolloff_95": {
            "mean": float(np.mean(active_rolloff_95)) if len(active_rolloff_95) > 0 else 5000.0,
            "trajectory": rolloff_95
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
        "sub600_ratio": sub600_ratio,
        "sub800_ratio": sub800_ratio,
        "sub1k_ratio": sub1k_ratio,
        "sub350_ratio": sub350_ratio,
        "call_rate": float(call_rate),
        "total_onsets": len(onset_frames),
        "onset_times": onset_times
    }
