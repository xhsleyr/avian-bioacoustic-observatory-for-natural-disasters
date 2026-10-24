# Project State & Memory: Avian Early Warning System (AEWS)

**Project Title:** Avian Vocal Bioacoustics for Natural Disaster Early Warning  
**Target Environment:** Local macOS (Apple Silicon M1)  
**Stack:** Python 3.11, Streamlit, Librosa, PyTorch, ONNX Runtime (BirdNET 6K), Plotly, NumPy  
**Zero-Cost Principle:** Fully local execution, no paid cloud APIs.

---

## 1. Project Goal & Scientific Foundation
Develop an acoustic early-warning system that captures live microphone environmental sounds or audio files, identifies bird species across **6,522 global species**, verifies biological authenticity via an **Anti-Spoofing & Biometric Shield**, and analyzes authentic vocalizations to detect abnormal collective distress, acoustic panic, or sudden flock dispersion indicators preceding natural disasters (e.g., seismic P-waves, severe storms, cyclones).

### Multi-Factor Pipeline:
1. **🎙️ Multi-Channel Input:**
   - Live Microphone capture (`st.audio_input` at 44.1 kHz) using Mac internal or external USB/Bluetooth mics.
   - File Uploader (`.wav`, `.mp3`, `.ogg`, `.flac`).
   - 6 Calibrated Real-World & Synthetic Benchmark Presets.
2. **🛡️ Bioacoustic Anti-Spoofing Shield:**
   - Detects explicit non-bird and human classes ('Human whistle', 'Human vocal', 'Engine', 'Siren').
   - Checks for sub-600 Hz human exhalation breath turbulence.
   - Evaluates human oral whistling frequency ceiling (< 3,000 Hz).
   - Suppresses Disaster Warnings on unverified/spoofed audio to prevent civil false alarms.
3. **🤖 Cornell BirdNET Global Classifier:**
   - Identifies species across 6,522 global species using 3-second overlapping sliding windows (50% overlap @ 48 kHz).
4. **🚨 Acoustic Distress Index (ADI 0 - 100):**
   - Evaluates Pitch Elevation, Vocal Burst Density, Noise Harshness, and Energy Volatility.
   - Classifies threat into Normal 🟢, Elevated Alert 🟡, and Critical Early Warning 🔴.
5. **🌪️ Documented Real-World Disaster Sentinel Sounds:**
   - **Golden-winged Warbler:** Fled 48h before an outbreak of 84 tornadoes (*Current Biology, 2014*).
   - **Indian Peafowl:** Real pre-monsoon & pre-seismic compressional P-wave honking alarms.

---

## 2. Directory Structure
```
Birds Interaction and Communication/
├── .gitignore
├── PROJECT_STATE.md        # Persistent project memory and progress tracker
├── README.md               # User guide & project presentation documentation
├── requirements.txt        # Python dependencies
├── app.py                  # Streamlit Web Application with Live Mic, Anti-Spoofing & Species Card
├── models/
│   ├── birdnet_model.onnx                      # Cornell BirdNET Global ONNX weights (49.3 MB)
│   └── BirdNET_GLOBAL_6K_V2.4_Labels.txt       # 6,522 Species labels (Common & Scientific names)
├── src/
│   ├── __init__.py
│   ├── audio_processor.py      # Audio loading, VAD masking, Mel-spectrogram & feature extraction
│   ├── distress_engine.py      # Acoustic Distress Index (ADI) calculation & warning logic
│   ├── species_classifier.py   # Cornell BirdNET 6,522 Species Engine & Disaster sensitivity heuristics
│   └── spoof_detector.py       # Bioacoustic Anti-Spoofing Shield & false-alarm suppression
└── data/
    └── samples/            # Benchmark audio samples (Normal, Elevated, Critical, Warbler, Peafowl, Human Mimic)
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
- [x] **Milestone 4: Bioacoustic Anti-Spoofing Shield (`src/spoof_detector.py`)**
  - Identifies human oral whistling mimicry, human speech, and artificial sirens.
  - Low-frequency breath turbulence ratio (0 - 600 Hz) detector.
  - Automatic false-alarm suppression mechanism for civil early-warning safety.
- [x] **Milestone 5: Real-World Disaster Benchmark Recordings**
  - Downloaded authentic field recordings of **Golden-winged Warbler** (99.6% BirdNET match) and **Indian Peafowl** (86.2% BirdNET match).
- [x] **Milestone 6: Live Microphone Stream & Streamlit UI (`app.py`)**
  - Integrated `st.audio_input` for real-time live environmental audio recording from Mac internal/external mics.
  - Anti-Spoofing Biometric Verification Shield banner.
  - Bird Species Identification card displaying Common Name, Scientific Name, Match Confidence %, and Disaster Sensitivity.
  - Interactive Plotly Gauge indicator.
  - Interactive Heatmap Mel-Spectrogram and Waveform with Onset burst markers.
  - Top 5 Candidate Species breakdown with probability bars.
  - Dedicated Anti-Spoofing Biometrics diagnostic tab.
  - Diagnostic factor analysis table.

---

## 4. How to Resume in a New Session
When opening a new chat session in Zed, prompt:
> *"Read `PROJECT_STATE.md` and continue where we left off."*
