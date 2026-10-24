# Project State & Memory: Avian Early Warning System (AEWS)

**Project Title:** Avian Vocal Bioacoustics for Natural Disaster Early Warning  
**Target Environment:** Local macOS (Apple Silicon M1)  
**Design & Typography:** Project CETI Exact Bioacoustic Aesthetic (`Playfair Display` serif, `Plus Jakarta Sans` body, `Space Mono` micro-labels, motion-blurred atmospheric iridescent gradients)  
**Stack:** Python 3.11, Streamlit, Librosa, PyTorch, ONNX Runtime (BirdNET 6K), Plotly, NumPy  
**Zero-Cost Principle:** Fully local execution, no paid cloud APIs.

---

## 1. Project Goal & Scientific Foundation
Develop an acoustic early-warning system that captures live microphone environmental sounds or audio files, identifies bird species across **6,522 global species**, dynamically retrieves high-resolution species photographs, verifies biological authenticity via an **Anti-Spoofing & Biometric Shield**, and analyzes authentic vocalizations to detect abnormal collective distress, acoustic panic, or sudden flock dispersion indicators preceding natural disasters (e.g., seismic P-waves, severe storms, cyclones).

### Project CETI Design System (from `DESIGN-projectceti-org.md`):
* **Atmospheric Visual Atmosphere:** Fluid motion-blurred iridescent background aura blending Mint/Cyan (`#befff7`), Electric Blue (`#4749fe`), and Deep Purple (`#4d1bc7`) over a pristine white canvas.
* **Typography:**
  - Display / Headlines: High-contrast editorial serif (`Playfair Display`, GT Super equivalent) with negative letter-spacing (`-0.025em`).
  - Body & UI: Clean, modern Swiss sans (`Plus Jakarta Sans`, Söhne equivalent).
  - Micro-labels, Badges, Metrics: Technical monospace (`Space Mono`).
* **Architectural Elements:** Sharp 0px card containers with hairline `#e6e8ee` borders, subtle purple drop-shadows on hover, and 100px pill badges (`.ceti-pill-badge`).
* **Signature CETI Gradient Banner:** Linear gradient `linear-gradient(135deg, #befff7 0%, #4749fe 50%, #4d1bc7 100%)` for catastrophe risk classification.
* **Immunity to Dark/Light Glitches:** Explicit container, input, tab, metric, and text overrides with `!important` ensuring zero contrast clashes or unreadable text across macOS system theme transitions.

---

## 2. Directory Structure
```
Birds Interaction and Communication/
├── .gitignore
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration (Project CETI electric palette)
├── PROJECT_STATE.md            # Persistent project memory and progress tracker
├── README.md                   # User guide & project presentation documentation
├── DESIGN-projectceti-org.md   # Project CETI design tokens & specifications
├── requirements.txt            # Python dependencies
├── app.py                      # Project CETI Visual System Web Application
├── models/
│   ├── birdnet_model.onnx                      # Cornell BirdNET Global ONNX weights (49.3 MB)
│   └── BirdNET_GLOBAL_6K_V2.4_Labels.txt       # 6,522 Species labels (Common & Scientific names)
├── src/
│   ├── __init__.py
│   ├── audio_processor.py      # Audio loading, VAD masking, Mel-spectrogram & feature extraction
│   ├── distress_engine.py      # Acoustic Distress Index (ADI) calculation & warning logic
│   ├── species_classifier.py   # Cornell BirdNET 6,522 Species Engine & Disaster sensitivity heuristics
│   ├── spoof_detector.py       # Bioacoustic Anti-Spoofing Shield & false-alarm suppression
│   └── image_fetcher.py        # Dynamic bird photo and taxonomy summary fetcher with offline caching
└── data/
    ├── cache/                  # Local species image and taxonomy cache
    └── samples/                # Benchmark audio samples (Normal, Elevated, Critical, Warbler, Peafowl, Human Mimic)
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
- [x] **Milestone 6: Live Microphone Stream (`app.py`)**
  - Integrated `st.audio_input` for real-time live environmental audio recording from Mac internal/external mics.
- [x] **Milestone 7: Dynamic Species Photography (`src/image_fetcher.py`)**
  - Connected Wikipedia / Wikimedia taxonomy REST API with local caching.
- [x] **Milestone 8: Times New Roman Exploration (Rejected by User)**
  - Attempted academic journal aesthetic; user rejected Times New Roman look in favor of Project CETI.
- [x] **Milestone 9: Full Project CETI Redesign (`DESIGN-projectceti-org.md`)**
  - Fully overhauled the UI to match https://www.projectceti.org/ exactly:
    - Atmospheric motion-blurred iridescent background aura (Cyan `#befff7` -> Electric Blue `#4749fe` -> Deep Purple `#4d1bc7`).
    - High-contrast editorial serif headers (`Playfair Display`), clean modern sans body (`Plus Jakarta Sans`), and technical monospace micro-labels (`Space Mono`).
    - Top navigation bar with live indicator dot and monospace links.
    - Architectural 0px card containers with hairline borders and hover glow.
    - 100px pill buttons for filters and status tags.
    - Oceanic/iridescent heatmaps and Plotly charts matching CETI palette.
    - Cleaned up all quote syntax and dark/light mode glitches.

---

## 4. How to Resume in a New Session
When opening a new chat session in Zed, prompt:
> *"Read `PROJECT_STATE.md` and continue where we left off."*
