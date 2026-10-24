# Project State & Memory: Avian Early Warning System (AEWS)

**Project Title:** Avian Vocal Bioacoustics for Natural Disaster Early Warning  
**Target Environment:** Local macOS (Apple Silicon M1)  
**Design System:** Inspired by [Project CETI](https://www.projectceti.org/) (Cyan-Blue-Purple motion gradients, GT Super editorial serif, Sohne body typography, architectural 0px cards)  
**Stack:** Python 3.11, Streamlit, Librosa, PyTorch, ONNX Runtime (BirdNET 6K), Plotly, NumPy  
**Zero-Cost Principle:** Fully local execution, no paid cloud APIs.

---

## 1. Project Goal & Scientific Foundation
Develop an acoustic early-warning system that captures live microphone environmental sounds or audio files, identifies bird species across **6,522 global species**, dynamically retrieves high-resolution species photographs, verifies biological authenticity via an **Anti-Spoofing & Biometric Shield**, and analyzes authentic vocalizations to detect abnormal collective distress, acoustic panic, or sudden flock dispersion indicators preceding natural disasters (e.g., seismic P-waves, severe storms, cyclones).

### Visual & Design System ([`DESIGN-projectceti-org.md`](file:///Users/xhselyr/My%20Projects/Birds%20Interaction%20and%20Communication/DESIGN-projectceti-org.md)):
* **Color Palette:** Primary Deep Purple (`#4d1bc7`), Electric Blue Accent (`#4749fe`), Cyan Shimmer (`#befff7`), Slate Blue text (`#4e5a7c`), White background (`#ffffff`), and Dark Slate (`#181b22`).
* **Atmospheric Motion Blur:** Radial gradient lighting mesh with ethereal blur overlays that recreate the living bioacoustic depth seen on Project CETI.
* **Typography:** Editorial Serif display headlines (`Cormorant Garamond` / `GT Super Display`) paired with clean geometric body text (`Plus Jakarta Sans` / `Sohne`) and `JetBrains Mono` for metadata tags.
* **Architectural Precision:** Sharp `0px` radius on cards and primary action buttons; `100px` pill badges for filter and status tags.

---

## 2. Directory Structure
```
Birds Interaction and Communication/
├── .gitignore
├── PROJECT_STATE.md        # Persistent project memory and progress tracker
├── README.md               # User guide & project presentation documentation
├── DESIGN-projectceti-org.md # Project CETI design system specifications
├── requirements.txt        # Python dependencies
├── app.py                  # Streamlit Web Application with Project CETI design system
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
    ├── cache/              # Local species image and taxonomy cache
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
- [x] **Milestone 6: Live Microphone Stream (`app.py`)**
  - Integrated `st.audio_input` for real-time live environmental audio recording from Mac internal/external mics.
- [x] **Milestone 7: Dynamic Species Photography (`src/image_fetcher.py`)**
  - Connected Wikipedia / Wikimedia taxonomy REST API with local caching.
- [x] **Milestone 8: Complete Frontend Overhaul Inspired by Project CETI (`app.py`)**
  - Atmospheric motion-blurred gradient backdrop (cyan-electric blue-purple aura).
  - Editorial `GT Super Display` / `Cormorant Garamond` serif display typography.
  - Clean `Sohne` / `Plus Jakarta Sans` body text and `JetBrains Mono` system badges.
  - Architectural 0px card edges, 100px pill buttons, and CETI oceanic Spectrogram palette.

---

## 4. How to Resume in a New Session
When opening a new chat session in Zed, prompt:
> *"Read `PROJECT_STATE.md` and continue where we left off."*
