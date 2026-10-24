# Project State & Memory: Avian Early Warning System (AEWS)

**Project Title:** Avian Vocal Bioacoustics for Natural Disaster Early Warning  
**Target Environment:** Local macOS (Apple Silicon M1)  
**Stack:** Python 3.11, Streamlit, Librosa, PyTorch, Plotly, NumPy  
**Zero-Cost Principle:** Fully local execution, no paid cloud APIs.

---

## 1. Project Goal & Scientific Foundation
Develop an acoustic early-warning system that analyzes bird vocalizations from uploaded or streamed audio files to detect abnormal collective distress, acoustic panic, or sudden flock dispersion indicators preceding natural disasters (e.g., seismic P-waves, severe storms, cyclones).

### Multi-Factor Acoustic Distress Index (ADI):
1. **Spectral Centroid Shift (Pitch):** Normal songs occupy balanced frequency bands; alarm calls undergo rapid upward pitch shifts.
2. **Call Rate & Temporal Density:** Frantic bursts and elevated call repetitions vs rhythmic melodies.
3. **Harmonicity vs Noise (Spectral Entropy / Flatness):** Measure chaotic harsh screeching vs clean harmonic song.
4. **Energy Modulation (RMS & Onset Strengths):** Sudden acoustic spikes and dispersion calls.

---

## 2. Directory Structure
```
Birds Interaction and Communication/
├── .gitignore
├── PROJECT_STATE.md        # Persistent project memory and progress tracker
├── README.md               # User guide & project presentation documentation
├── requirements.txt        # Python dependencies
├── app.py                  # Streamlit Web Application
├── src/
│   ├── __init__.py
│   ├── audio_processor.py  # Audio loading, VAD masking, Mel-spectrogram & feature extraction
│   └── distress_engine.py  # Acoustic Distress Index (ADI) calculation & warning logic
└── data/
    └── samples/            # Benchmark audio samples (Normal, Elevated, Critical)
```

---

## 3. Progress Tracker & Milestones

- [x] **Milestone 0: Workspace & Environment Setup**
  - Git repository initialized with `.gitignore`.
  - Python 3.11 virtual environment (`.venv`) configured with Apple M1 hardware support.
  - Project state memory tracker established.
- [x] **Milestone 1: Core Audio Processing Engine (`src/audio_processor.py`)**
  - Audio loader with sample rate standardization (22.05 kHz) & peak normalization.
  - Vocal Activity Masking (filters silent gaps between bird chirps).
  - Mel-spectrogram, Spectral Centroid, Rolloff, Spectral Flatness, RMS, and Onset Detection.
- [x] **Milestone 2: Disaster Distress & Warning Scoring Engine (`src/distress_engine.py`)**
  - Formulated calibrated Acoustic Distress Index (ADI 0 - 100).
  - Classifies warning tiers: Normal 🟢 (0-37%), Elevated Alert 🟡 (38-69%), Critical Early Warning 🔴 (70-100%).
  - Detailed diagnostic metrics breakdown.
- [x] **Milestone 3: Streamlit Web Dashboard (`app.py`)**
  - File uploader (`.wav`, `.mp3`, `.ogg`, `.flac`) and audio player.
  - Preset Benchmark demo selector with instant 1-click loading.
  - Interactive Plotly Gauge indicator.
  - Interactive Heatmap Mel-Spectrogram and Waveform with Onset burst markers.
  - Diagnostic factor analysis table.
- [x] **Milestone 4: Sample Testing & Evaluation Documentation**
  - Synthesized calibrated benchmark samples (`calm_robin_song.wav`, `elevated_warning_call.wav`, `critical_disaster_panic.wav`).
  - Created [`README.md`](file:///Users/xhselyr/My%20Projects/Birds%20Interaction%20and%20Communication/README.md) with architectural diagrams and run instructions.

---

## 4. How to Resume in a New Session
When opening a new chat session in Zed, prompt:
> *"Read `PROJECT_STATE.md` and continue where we left off."*
