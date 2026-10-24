# Project State & Memory: Project INTERA

**Project Title:** Avian Vocal Bioacoustics for Natural Disaster Early Warning  
**Target Environment:** Local macOS (Apple Silicon M1)  
**Release Tag:** `v1.0.0-first-final-demo` (First Final Demo Version)  
**Design & Typography:** Times New Roman serif across all headings, body, metrics, tables, and Plotly charts, styled within Project CETI's atmospheric motion-blurred gradient aesthetic with standard Figma-style vector line SVG icons.  
**Stack:** Python 3.11, Streamlit, Librosa, PyTorch, ONNX Runtime (BirdNET 6K), Plotly, NumPy  
**Zero-Cost Principle:** Fully local execution, no paid cloud APIs.

---

## 1. Project Goal & Scientific Foundation
Develop an acoustic early-warning system that captures live microphone environmental sounds or audio files, identifies bird species across **6,522 global species**, dynamically retrieves high-resolution species photographs, verifies biological authenticity via an **Anti-Spoofing & Biometric Shield**, and analyzes authentic vocalizations to detect abnormal collective distress, acoustic panic, or sudden flock dispersion indicators preceding natural disasters (e.g., seismic P-waves, severe storms, cyclones).

### Visual System & Design Architecture:
* **Atmospheric Visual Aura:** Motion-blurred iridescent gradient background blending Mint/Cyan (`#befff7`), Electric Blue (`#4749fe`), and Deep Purple (`#4d1bc7`).
* **Typography:** Pure **Times New Roman** (`'Times New Roman', Times, Georgia, serif`) across all display headlines, navigation, body copy, metrics, and Plotly visualizers.
* **Icons:** Standard Figma-style vector line SVG icons (Lucide / Icons8 minimal outline aesthetic) replacing emojis.
* **Architectural Elements:** 0px sharp white cards with hairline borders (`#e6e8ee`), 100px rounded pill badges, and tri-color disaster warning banner.

---

## 2. Directory Structure
```
Birds Interaction and Communication/
├── .gitignore
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration (font = "serif")
├── PROJECT_STATE.md            # Persistent project memory and version milestones
├── README.md                   # First Final Demo comprehensive documentation
├── DESIGN-projectceti-org.md   # Project CETI design tokens & specifications
├── requirements.txt            # Python dependencies
├── app.py                      # Bioacoustic Early Warning Web Application
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
- [x] **Milestone 1: Core Audio Processing Engine (`src/audio_processor.py`)**
  - Audio loader with sample rate standardization (22.05 kHz) & peak normalization.
  - Vocal Activity Masking (filters silent gaps between bird chirps).
  - Mel-spectrogram, Spectral Centroid, Rolloff, Spectral Flatness, RMS, and Onset Detection.
- [x] **Milestone 2: Disaster Distress & Warning Scoring Engine (`src/distress_engine.py`)**
  - Formulated calibrated Acoustic Distress Index (ADI 0 - 100).
  - Classifies warning tiers: Normal (0-37%), Elevated Alert (38-69%), Critical Early Warning (70-100%).
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
- [x] **Milestone 8: Times New Roman Exploration**
  - Evaluated academic serif styling.
- [x] **Milestone 9: Project CETI Visual Overhaul**
  - Overhauled UI with Project CETI motion-blurred atmospheric gradients, 0px cards, and pill badges.
- [x] **Milestone 10: Standard Figma Vector Line Icons**
  - Replaced all emojis across the interface with crisp, minimalist Figma/Icons8-style vector line SVG icons (`ICON_SHIELD`, `ICON_ALERT`, `ICON_CHECK`, `ICON_CROSS`, `ICON_MIC`, `ICON_WAVE`, `ICON_NETWORK`).
- [x] **Milestone 11: Integration of Google Font `Savate`**
  - Tested Google Font `Savate`.
- [x] **Milestone 12: Times New Roman Font Replacement**
  - Standardized font across `.streamlit/config.toml`, CSS styling, and Plotly visualizers to **Times New Roman** (`'Times New Roman', Times, Georgia, serif`).
- [x] **Milestone 13: First Final Demo Release (`v1.0.0-first-final-demo`)**
  - Preserved, tested, and bookmarked as the official **First Final Demo** of the project.
  - Comprehensive thesis README presentation guide documented.
  - Tagged in git as `v1.0.0-first-final-demo` for safe version restoration.

---

## 4. How to Resume in a New Session
When opening a new chat session in Zed, prompt:
> *"Read `PROJECT_STATE.md` and continue where we left off."*
