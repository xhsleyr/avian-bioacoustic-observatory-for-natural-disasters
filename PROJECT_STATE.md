# Project State & Memory: Avian Early Warning System (AEWS)

**Project Title:** Avian Vocal Bioacoustics for Natural Disaster Early Warning  
**Target Environment:** Local macOS (Apple Silicon M1)  
**Stack:** Python 3.11, Streamlit, Librosa, PyTorch, Transformers, Plotly, NumPy  
**Zero-Cost Principle:** Fully local execution, no paid cloud APIs.

---

## 1. Project Goal & Scientific Foundation
Develop an acoustic early-warning system that identifies bird species and analyzes vocalizations from audio files to detect abnormal collective distress, acoustic panic, or sudden flock dispersion indicators preceding natural disasters (e.g., seismic P-waves, severe storms, cyclones).

### Multi-Factor Acoustic Distress Index (ADI):
1. **Species-Adaptive Baseline:** Calibrates detected frequency against the identified bird's natural pitch range.
2. **Spectral Centroid Shift (Pitch):** Normal songs occupy balanced frequency bands; alarm calls undergo rapid upward pitch shifts.
3. **Call Rate & Temporal Density:** Frantic bursts and elevated call repetitions vs rhythmic melodies.
4. **Harmonicity vs Noise (Spectral Flatness):** Measures chaotic harsh screeching vs clean harmonic song.
5. **Energy Modulation (RMS & Onset Strengths):** Sudden acoustic spikes and dispersion calls.

---

## 2. Directory Structure
```
Birds Interaction and Communication/
├── .gitignore
├── PROJECT_STATE.md        # Persistent project memory and progress tracker
├── README.md               # User guide & project presentation documentation
├── requirements.txt        # Python dependencies
├── app.py                  # Streamlit Web Application with Species Identification Card
├── src/
│   ├── __init__.py
│   ├── audio_processor.py      # Audio loading, VAD masking, Mel-spectrogram & feature extraction
│   ├── distress_engine.py      # Acoustic Distress Index (ADI) calculation & warning logic
│   └── species_classifier.py   # Bird species identification & biological baseline deviation
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
- [x] **Milestone 3: Species Identification Engine (`src/species_classifier.py`)**
  - Avian Bioacoustic Database (Common & Scientific names, Family, baseline frequency, natural call types).
  - Multi-dimensional acoustic similarity & confidence percentage scoring.
  - Species-specific vocal deviation analysis (identifies if the bird is screeching above its natural pitch).
- [x] **Milestone 4: Streamlit Web Dashboard (`app.py`)**
  - Prominent Bird Species Identification card displaying Common Name, Scientific Name, Match Confidence %, and Disaster Sensitivity.
  - File uploader and Preset Benchmark demo selector with instant 1-click loading.
  - Interactive Plotly Gauge indicator.
  - Interactive Heatmap Mel-Spectrogram and Waveform with Onset burst markers.
  - Top Candidate Species breakdown with probability bars.
  - Diagnostic factor analysis table.
- [x] **Milestone 5: Sample Testing & Evaluation Documentation**
  - Verified on calibrated benchmark samples (`American Robin`, `House Sparrow`, `European Starling`).
  - Created [`README.md`](file:///Users/xhselyr/My%20Projects/Birds%20Interaction%20and%20Communication/README.md) with architectural diagrams and run instructions.

---

## 4. How to Resume in a New Session
When opening a new chat session in Zed, prompt:
> *"Read `PROJECT_STATE.md` and continue where we left off."*
