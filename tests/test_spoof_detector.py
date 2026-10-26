"""Unit tests for avian syrinx biometric shield and anti-spoofing detector."""

import os

import pytest

from src.audio_processor import extract_acoustic_features, load_audio
from src.species_classifier import identify_species
from src.spoof_detector import evaluate_avian_authenticity


@pytest.mark.skipif(not os.path.exists("data/samples/calm_robin_song.wav"), reason="Sample fixture missing")
def test_authentic_bird_passes_shield():
    """Verify authentic bird audio passes biometric verification."""
    y, sr, _ = load_audio("data/samples/calm_robin_song.wav")
    with open("data/samples/calm_robin_song.wav", "rb") as f:
        audio_bytes = f.read()
    features = extract_acoustic_features(y, sr)
    species_info = identify_species(audio_bytes, features, top_k=3)
    spoof_info = evaluate_avian_authenticity(y, sr, species_info, features)
    assert spoof_info["is_authentic"] is True
    assert spoof_info["suppress_alarm"] is False


@pytest.mark.skipif(not os.path.exists("data/samples/domestic_cat_meow_spoof.wav"), reason="Sample fixture missing")
def test_cat_meow_rejected():
    """Verify mammalian cat meow is correctly rejected by biometric shield."""
    y, sr, _ = load_audio("data/samples/domestic_cat_meow_spoof.wav")
    with open("data/samples/domestic_cat_meow_spoof.wav", "rb") as f:
        audio_bytes = f.read()
    features = extract_acoustic_features(y, sr)
    species_info = identify_species(audio_bytes, features, top_k=3)
    spoof_info = evaluate_avian_authenticity(y, sr, species_info, features)
    assert spoof_info["is_authentic"] is False
    assert spoof_info["suppress_alarm"] is True
    assert "cat" in spoof_info["detected_category"] or "mammal" in spoof_info["detected_category"]


@pytest.mark.skipif(not os.path.exists("data/samples/human_whistling_mimic.wav"), reason="Sample fixture missing")
def test_human_whistle_rejected():
    """Verify human whistling mimicry is detected and rejected."""
    y, sr, _ = load_audio("data/samples/human_whistling_mimic.wav")
    with open("data/samples/human_whistling_mimic.wav", "rb") as f:
        audio_bytes = f.read()
    features = extract_acoustic_features(y, sr)
    species_info = identify_species(audio_bytes, features, top_k=3)
    spoof_info = evaluate_avian_authenticity(y, sr, species_info, features)
    assert spoof_info["is_authentic"] is False
    assert spoof_info["suppress_alarm"] is True
