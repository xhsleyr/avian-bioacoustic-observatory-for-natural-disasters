"""Generates realistic synthetic bird vocalization samples and human mimicry for testing."""

import os

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

SR = 22050
DURATION = 5.0
OUT_DIR = "data/samples"
os.makedirs(OUT_DIR, exist_ok=True)


def generate_calm_song():
    """Generates a pleasant, rhythmic bird whistle song."""
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    y = np.zeros_like(t)

    chirp_times = [0.5, 1.6, 2.7, 3.8]
    for ct in chirp_times:
        start_idx = int(ct * SR)
        dur = 0.4
        n_samples = int(dur * SR)
        t_c = np.linspace(0, dur, n_samples)
        
        freq = 2400 + 400 * np.sin(2 * np.pi * 5 * t_c)
        phase = 2 * np.pi * np.cumsum(freq) / SR
        envelope = np.sin(np.pi * (t_c / dur)) ** 2
        chirp = envelope * np.sin(phase)

        end_idx = min(start_idx + n_samples, len(y))
        y[start_idx:end_idx] += chirp[:end_idx - start_idx]

    y = y / (np.max(np.abs(y)) + 1e-6)
    out_path = os.path.join(OUT_DIR, "calm_robin_song.wav")
    sf.write(out_path, y, SR)
    print(f"Generated {out_path}")


def generate_elevated_alarm():
    """Generates rapid alarm warning calls with elevated frequency."""
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    y = np.zeros_like(t)

    chirp_times = np.linspace(0.3, 4.5, 10)
    for ct in chirp_times:
        start_idx = int(ct * SR)
        dur = 0.18
        n_samples = int(dur * SR)
        t_c = np.linspace(0, dur, n_samples)
        
        freq = np.linspace(4800, 3400, n_samples)
        phase = 2 * np.pi * np.cumsum(freq) / SR
        envelope = np.sin(np.pi * (t_c / dur)) ** 1.5
        chirp = envelope * np.sin(phase)

        end_idx = min(start_idx + n_samples, len(y))
        y[start_idx:end_idx] += chirp[:end_idx - start_idx]

    y = y / (np.max(np.abs(y)) + 1e-6)
    out_path = os.path.join(OUT_DIR, "elevated_warning_call.wav")
    sf.write(out_path, y, SR)
    print(f"Generated {out_path}")


def generate_severe_panic():
    """Generates frantic Starling/Tern mobbing shrieks indicative of pre-disaster panic."""
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    y = np.zeros_like(t)

    # 18 rapid chaotic bird screeches (3.6 calls/sec) with avian FM downsweeps
    chirp_times = np.linspace(0.2, 4.7, 18)
    for ct in chirp_times:
        start_idx = int(ct * SR)
        dur = 0.14
        n_samples = int(dur * SR)
        t_c = np.linspace(0, dur, n_samples)
        
        # Avian rapid downsweep from 5800 Hz down to 3600 Hz with dual harmonics
        f1 = np.linspace(5800, 3600, n_samples)
        f2 = np.linspace(7200, 4800, n_samples)
        phase1 = 2 * np.pi * np.cumsum(f1) / SR
        phase2 = 2 * np.pi * np.cumsum(f2) / SR
        envelope = np.sin(np.pi * (t_c / dur)) ** 1.2
        chirp = envelope * (0.7 * np.sin(phase1) + 0.3 * np.sin(phase2))

        end_idx = min(start_idx + n_samples, len(y))
        y[start_idx:end_idx] += chirp[:end_idx - start_idx]

    y = y / (np.max(np.abs(y)) + 1e-6)
    out_path = os.path.join(OUT_DIR, "critical_disaster_panic.wav")
    sf.write(out_path, y, SR)
    print(f"Generated {out_path}")


def generate_human_mimicry():
    """Generates a human whistling attempt with slow breathy vibrato and low-freq airflow."""
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    y = np.zeros_like(t)

    whistle_intervals = [(0.5, 2.0), (2.8, 4.4)]
    for start_t, end_t in whistle_intervals:
        dur = end_t - start_t
        n_samples = int(dur * SR)
        t_w = np.linspace(0, dur, n_samples)
        
        freq = 1950 + 80 * np.sin(2 * np.pi * 3.2 * t_w)
        phase = 2 * np.pi * np.cumsum(freq) / SR
        envelope = np.sin(np.pi * (t_w / dur)) ** 0.8
        whistle_tone = envelope * np.sin(phase)

        raw_noise = np.random.normal(0, 0.15, n_samples)
        b, a = butter(4, 450 / (SR / 2), btype='low')
        breath_noise = lfilter(b, a, raw_noise) * envelope * 0.45

        composite = whistle_tone + breath_noise
        start_idx = int(start_t * SR)
        end_idx = min(start_idx + n_samples, len(y))
        y[start_idx:end_idx] += composite[:end_idx - start_idx]

    y = y / (np.max(np.abs(y)) + 1e-6)
    out_path = os.path.join(OUT_DIR, "human_whistling_mimic.wav")
    sf.write(out_path, y, SR)
    print(f"Generated {out_path}")


if __name__ == "__main__":
    generate_calm_song()
    generate_elevated_alarm()
    generate_severe_panic()
    generate_human_mimicry()
