# Project State & Memory: Avian Early Warning System (AEWS)

**Project Title:** Avian Vocal Bioacoustics for Natural Disaster Early Warning  
**Target Environment:** Local macOS (Apple Silicon M1)  
**Stack:** Python 3.11, Streamlit, Librosa, PyTorch, ONNX Runtime (BirdNET 6K), Plotly, NumPy  
**Zero-Cost Principle:** Fully local execution, no paid cloud APIs.

---

## 1. Project Goal & Scientific Foundation
Develop an acoustic early-warning system that identifies bird species across **6,522 global species** and analyzes vocalizations from audio files to detect abnormal collective distress, acoustic panic, or sudden flock dispersion indicators preceding natural disasters (e.g., seismic P-waves, severe storms, cyclones).

### Multi-Factor Acoustic Distress Index (ADI):
1. **Cornell BirdNET Global Classifier:** Identifies species across 6,522 global species from sliding 3s audio windows with 50% overlap.
2. **Species-Adaptive Baseline:** Calibrates detected frequency against the identified bird's natural pitch range.
3. **Spectral Centroid Shift (Pitch):** Normal songs occupy balanced frequency bands; alarm calls undergo rapid upward pitch shifts.
4. **Call Rate & Temporal Density:** Frantic bursts and elevated call repetitions vs rhythmic melodies.
5. **Harmonicity vs Noise (Spectral Flatness):** Measures chaotic harsh screeching vs clean harmonic song.
6. **Energy Modulation (RMS & Onset Strengths):** Sudden acoustic spikes and dispersion calls.

---

## 2. Directory Structure
```
Birds Interaction and Communication/
├── .gitignore
├── PROJECT_STATE.md        # Persistent project memory and progress tracker
├── README.md               # User guide & project presentation documentation
├── requirements.txt        # Python dependencies
├── app.py                  # Streamlit Web Application with Species Identification Card
├── models/
│   ├── birdnet_model.onnx                      # Cornell BirdNET Global ONNX weights (49.3 MB)
│   └── BirdNET_GLOBAL_6K_V2.4_Labels.txt       # 6,522 Species labels (Common & Scientific names)
├── src/
│   ├── __init__.py
│   ├── audio_processor.py      # Audio loading, VAD masking, Mel-spectrogram & feature extraction
│   ├── distress_engine.py      # Acoustic Distress Index (ADI) calculation & warning logic
│   └── species_classifier.py   # Cornell BirdNET 6,522 Species Engine & Disaster sensitivity heuristics
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
- [x] **Milestone 3: Global Species Classifier (`src/species_classifier.py`)**
  - Integrated Cornell BirdNET v2.4 ONNX model supporting **6,522 global species**.
  - Slices audio into 3-second overlapping windows (50% overlap @ 48 kHz).
  - Max-pooling confidence aggregation across audio recordings.
  - Avian disaster-sensitivity heuristic taxonomy mapping.
- [x] **Milestone 4: Streamlit Web Dashboard (`app.py`)**
  - Bird Species Identification card displaying Common Name, Scientific Name, Match Confidence %, and Disaster Sensitivity.
  - File uploader and Preset Benchmark demo selector with instant 1-click loading.
  - Interactive Plotly Gauge indicator.
  - Interactive Heatmap Mel-Spectrogram and Waveform with Onset burst markers.
  - Top 5 Candidate Species breakdown with probability bars.
  - Diagnostic factor analysis table.
- [x] **Milestone 5: Sample Testing & Evaluation Documentation**
  - Verified on calibrated benchmark samples.
  - Created [`README.md`](file:///Users/xhselyr/My%20Projects/Birds%20Interaction%20and%20Communication/README.md) with architectural diagrams and run instructions.

---

## 4. How to Resume in a New Session
When opening a new chat session in Zed, prompt:
> *"Read `PROJECT_STATE.md` and continue where we left off."*
