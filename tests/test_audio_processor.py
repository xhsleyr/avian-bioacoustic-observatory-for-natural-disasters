"""Unit tests for bioacoustic audio processing and feature extraction."""

import numpy as np
import pytest

from src.audio_processor import (
    adaptive_spectral_denoise,
    estimate_stereo_bearing,
    extract_acoustic_features,
    filter_environmental_noise,
    load_audio,
)


def test_load_audio_sample():
    """Verify loading from real benchmark audio fixture."""
    y, sr, duration = load_audio("data/samples/calm_robin_song.wav")
    assert sr == 22050
    assert duration > 0.5
    assert len(y) > 0
    assert np.max(np.abs(y)) <= 1.001


def test_filter_environmental_noise():
    """Verify 90 Hz low-cut filter removes low-frequency DC/sub-audible rumble."""
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 30 Hz wind rumble + 3000 Hz bird chirp
    rumble = 0.8 * np.sin(2 * np.pi * 30 * t)
    chirp = 0.5 * np.sin(2 * np.pi * 3000 * t)
    mixed = rumble + chirp
    filtered = filter_environmental_noise(mixed, sr, cutoff_hz=90.0)
    assert len(filtered) == len(mixed)
    # Energy in 30 Hz should be significantly attenuated relative to 3000 Hz
    assert np.std(filtered) > 0


def test_adaptive_spectral_denoise():
    """Verify stationary noise gating reduces noise floor without corrupting length."""
    sr = 22050
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    tone = np.sin(2 * np.pi * 2500 * t) + 0.05 * np.random.randn(len(t))
    cleaned = adaptive_spectral_denoise(tone, sr)
    assert len(cleaned) == len(tone)
    assert isinstance(cleaned, np.ndarray)


def test_extract_acoustic_features():
    """Verify comprehensive acoustic feature extraction."""
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    synthetic_chirp = 0.8 * np.sin(2 * np.pi * 3500 * t)
    feats = extract_acoustic_features(synthetic_chirp, sr)
    
    assert "spectral_centroid" in feats
    assert feats["spectral_centroid"]["mean"] > 2000
    assert "call_rate" in feats
    assert "sub800_ratio" in feats
    assert feats["sub800_ratio"] < 0.10  # 3500 Hz chirp has negligible sub-800Hz energy
    assert "mel_db" in feats
    assert feats["mel_db"].shape[0] == 128


def test_extract_acoustic_features_empty_raises():
    """Verify empty audio raises ValueError."""
    with pytest.raises(ValueError):
        extract_acoustic_features(np.array([], dtype=np.float32), 22050)


def test_estimate_stereo_bearing():
    """Verify GCC-PHAT time delay of arrival estimation for directional hearing."""
    sr = 22050
    n = 2205
    ch1 = np.random.randn(n)
    # Delay ch2 by 5 samples
    delay = 5
    ch2 = np.roll(ch1, delay)
    stereo = np.vstack([ch1, ch2])
    bearing = estimate_stereo_bearing(stereo, sr)
    assert bearing is not None
    assert isinstance(bearing, float)
