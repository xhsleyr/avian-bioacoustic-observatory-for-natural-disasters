"""Generates realistic synthetic bird vocalization samples for testing and evaluation."""

import os
import numpy as np
import soundfile as sf

SR = 22050
DURATION = 5.0
OUT_DIR = "data/samples"
os.makedirs(OUT_DIR, exist_ok=True)


def generate_calm_song():
    """Generates a pleasant, rhythmic bird whistle song."""
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    y = np.zeros_like(t)

    # 4 distinct melodic syllables over 5 seconds
    chirp_times = [0.5, 1.6, 2.7, 3.8]
    for ct in chirp_times:
        start_idx = int(ct * SR)
        dur = 0.4
        n_samples = int(dur * SR)
        t_c = np.linspace(0, dur, n_samples)
        
        # Smooth frequency modulation between 2200 and 2800 Hz
        freq = 2400 + 400 * np.sin(2 * np.pi * 5 * t_c)
        phase = 2 * np.pi * np.cumsum(freq) / SR
        envelope = np.sin(np.pi * (t_c / dur)) ** 2
        chirp = envelope * np.sin(phase)

        end_idx = min(start_idx + n_samples, len(y))
        y[start_idx:end_idx] += chirp[:end_idx - start_idx]

    # Normalize
    y = y / (np.max(np.abs(y)) + 1e-6)
    out_path = os.path.join(OUT_DIR, "calm_robin_song.wav")
    sf.write(out_path, y, SR)
    print(f"Generated {out_path}")


def generate_elevated_alarm():
    """Generates rapid alarm warning calls with elevated frequency."""
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    y = np.zeros_like(t)

    # 10 fast chirps over 5 seconds (2 calls/sec)
    chirp_times = np.linspace(0.3, 4.5, 10)
    for ct in chirp_times:
        start_idx = int(ct * SR)
        dur = 0.18
        n_samples = int(dur * SR)
        t_c = np.linspace(0, dur, n_samples)
        
        # Fast downsweep frequency 4500 -> 3200 Hz
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
    """Generates frantic, screeching chaos indicative of pre-disaster panic."""
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    y = np.zeros_like(t)

    # 20 frantic chaotic screech pulses (4 calls/sec)
    chirp_times = np.linspace(0.1, 4.8, 22)
    for ct in chirp_times:
        start_idx = int(ct * SR)
        dur = 0.12
        n_samples = int(dur * SR)
        t_c = np.linspace(0, dur, n_samples)
        
        # High screech with harmonic distortion and harsh noise
        freq = 6200 + 800 * np.sin(2 * np.pi * 35 * t_c)
        phase = 2 * np.pi * np.cumsum(freq) / SR
        noise = np.random.normal(0, 0.25, n_samples)
        envelope = np.sin(np.pi * (t_c / dur))
        chirp = envelope * (np.sin(phase) + 0.5 * np.sin(2 * phase) + noise)

        end_idx = min(start_idx + n_samples, len(y))
        y[start_idx:end_idx] += chirp[:end_idx - start_idx]

    y = y / (np.max(np.abs(y)) + 1e-6)
    out_path = os.path.join(OUT_DIR, "critical_disaster_panic.wav")
    sf.write(out_path, y, SR)
    print(f"Generated {out_path}")


if __name__ == "__main__":
    generate_calm_song()
    generate_elevated_alarm()
    generate_severe_panic()
