# 🦅 Avian Bioacoustic Observatory for Natural Disasters (ABO)
### Autonomous Planetary Bioacoustic Early Warning Intelligence & Avian Distress Telemetry Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![BirdNET Global](https://img.shields.io/badge/Taxonomy-6%2C522%20Global%20Species-orange.svg)](https://birdnet.cornell.edu/)
[![FastAPI](https://img.shields.io/badge/REST%20API-FastAPI%202.4.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit UI](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Docker Ready](https://img.shields.io/badge/Docker-Multi--Stage%20Build-2496ED.svg)](https://www.docker.com/)
[![Tests Passing](https://img.shields.io/badge/Tests-17%2F17%20Passing-brightgreen.svg)](tests/)

---

## 📌 Executive Summary

The **Avian Bioacoustic Observatory for Natural Disasters (ABO)** is an autonomous, production-grade bioacoustic surveillance and forensic telemetry platform designed to forecast natural hazards before conventional civil mechanical instruments trigger.

### The Scientific Principle: Avian Infrasonic & Seismic Sensitivity
Hours prior to major natural disasters (such as seismic compressional P-waves, supercell tornado infrasound, and cyclonic micro-baroms), wild avians exhibit profound behavioural and vocal anomalies. Avian species possess specialised **Herbst corpuscles**—extraordinarily sensitive mechanoreceptors located in their tibiotarsal perichondrium and dermis capable of detecting minute sub-audible ground vibrations ($0.1–10\text{ Hz}$). Furthermore, avian inner ear structures detect micro-barometric shifts and atmospheric infrasonic acoustic waves that travel hundreds of kilometres ahead of tornadic vortex cones and megathrust seismic rifts.

ABO captures live environmental canopy audio, classifies species across **6,522 global taxa** using embedded **Cornell University BirdNET neural weights**, validates biological integrity via a **Dual-Syrinx Biometric Anti-Spoofing Shield**, calculates a real-time **Acoustic Distress Index (ADI)**, and emits standardised civil early-warning telemetry (**GeoJSON, JSON-LD, CSV, and Forensic Markdown Certificates**).

The entire pipeline executes **100% locally with zero cloud API dependencies and zero operating costs**.

---

## ⚡ Quickstart (Launch in Under 60 Seconds)

If you already have Python 3.10–3.12, Git, and FFmpeg installed on your machine:

```bash
# 1. Clone the repository
git clone https://github.com/xhsleyr/avian-bioacoustic-observatory-for-natural-disasters.git
cd avian-bioacoustic-observatory-for-natural-disasters

# 2. Set up virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# 3. Launch the Web Observatory Dashboard
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🌟 Key Engineering Capabilities

1. **Triple Audio Acquisition Streams:**
   - **Upload Field Recording:** Ingestion of `.wav`, `.mp3`, `.flac`, or `.ogg` field audio (up to 25 MB).
   - **Live Canopy Microphone Stream:** Real-time continuous sampling via directional or onboard hardware (`st.audio_input`).
   - **Calibrated Laboratory Benchmarks:** Six pre-loaded, peer-reviewed acoustic specimens (Normal Baseline Robin, Agitation Mobbing, Pre-Tremor Screech, 2014 Historic Tornado Warbler, Seismic P-Wave Peafowl, and Adversarial Human Whistling Mimicry).

2. **Cornell BirdNET Neural Species Classifier (6,522 Species):**
   - Embedded Cornell University BirdNET v2.4 ONNX model (`models/birdnet_model.onnx`, 49.3 MB) executing inference in **~100ms** via CPU / Apple Silicon CoreML / ONNX Runtime.
   - 3-second sliding analysis windows with 50% overlap and max-pooled multi-window confidence aggregation.
   - Specialized avian disaster-sensitivity heuristics (infrasound reactivity, seismic P-wave sensitivity, storm barometric responsiveness).

3. **Dual-Syrinx Biometric Anti-Spoofing Shield:**
   - Discriminates against human vocal cords, supraglottal oral cavities, whistles, siren alarms, and phone-speaker replay attacks.
   - Evaluates sub-600 Hz breath turbulence ratio ($R_{\text{breath}}$) and acoustic resonance ceiling ($< 3,000\text{ Hz}$) vs authentic avian dual-syrinx biphonation.
   - **Automated Alarm Suppression:** Automatically engages an interlock suppressing civil panic alerts when non-avian mimicry is detected.

4. **Calibrated Multi-Factor Acoustic Distress Index (ADI):**
   - Real-time calculus combining four bioacoustic acoustic parameters:
     $$\text{ADI} = W_1 \cdot \hat{S}_{\text{centroid}} + W_2 \cdot \hat{R}_{\text{burst}} + W_3 \cdot \hat{F}_{\text{spectral}} + W_4 \cdot \hat{\sigma}^2_{\text{RMS}}$$
     Where $W_1 = 0.35$ (Pitch Elevation), $W_2 = 0.30$ (Call Burst Rate), $W_3 = 0.20$ (Spectral Harshness / Flatness), and $W_4 = 0.15$ (Energy Volatility).
   - **Three-Tier Civil Alert Matrix:**
     - 🟢 **NORMAL (0 – 37%):** Baseline foraging and calm melodic territorial communication.
     - 🟡 **ELEVATED (38 – 69%):** Localized predator mobbing, moderate environmental agitation.
     - 🔴 **CRITICAL (70 – 100%):** Anomalous collective panic, pre-seismic compressional shockwave, or evacuation call.

5. **Planetary GIS & Forensics Export Suite:**
   - **Forensic Markdown Incident Certificate:** Formal cryptographic report detailing timestamps, species taxonomy, syrinx verification, and frequency breakdown.
   - **OGC-Compliant GeoJSON:** Point-feature spatial telemetry with coordinates, elevation, species biome, and estimated disaster lead-times (35–90 min).
   - **Machine-Readable JSON Telemetry:** Standardized payload for seismological and meteorological integration.
   - **Parametric CSV Time-Series:** Full acoustic feature matrix for downstream pandas/R analytics.

6. **Headless REST API (FastAPI):**
   - High-throughput REST API (`src/api.py`) for edge IoT nodes, Raspberry Pi solar sensor hubs, and remote monitoring stations.
   - Interactive Swagger/OpenAPI documentation at `/docs` and ReDoc at `/redoc`.

---

## 🏗️ System Architecture

```
[ Bioacoustic Input: Live Mic / File Upload / Calibrated Benchmarks ]
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
[ Librosa Audio Processor ]             [ Cornell BirdNET ONNX ]
  ├── 22,050 Hz Resampling                ├── 6,522 Global Species
  ├── Voice Activity Masking (VAD)        ├── 3-Second Sliding Windows
  ├── 128 Mel-Frequency Bins              ├── Softmax Confidence Pooling
  ├── Spectral Centroid & Rolloff         └── Disaster Sensitivity Matrix
  ├── Onset Burst Density (Events/sec)            │
  └── Sub-600 Hz Low-Band Energy                  │
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
        [ Syrinx Biometric Shield & Anti-Spoofing ]
          ├── Human Whistle & Voice Suppression
          ├── Oral Cavity Formant Boundary Check
          └── Automated False-Alarm Interlock
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
[ Distress Engine (ADI Calculus) ]      [ Offline Biology Cache ]
  ├── Pitch Stress Displacement           ├── Wikimedia High-Res Photos
  ├── Burst Escalation Velocity           ├── Taxonomic Classifications
  ├── Pre-Seismic Shockwave Detector      └── Persistent Local Cache
  └── Multi-Species Consensus Matrix              │
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
        [ Presentation & Telemetry Distribution Layer ]
          ├── Streamlit Editorial Web Dashboard (app.py)
          ├── Headless FastAPI REST Endpoints (src/api.py)
          ├── OGC-Compliant GeoJSON Planetary Stream
          ├── Seismological JSON Telemetry
          └── Bioacoustic Forensic Markdown Certificates
```

---

## 💻 Step-by-Step Cross-Platform Installation Guide

The observatory is engineered and tested for cross-platform compatibility across **macOS (Apple Silicon & Intel)**, **Windows 10/11**, and **Linux (Ubuntu, Debian, Fedora, Arch, Raspberry Pi OS 64-bit)**.

---

### 🍎 Option 1: macOS Installation (Apple Silicon M1/M2/M3/M4 & Intel)

#### Step 1: Install System Audio Libraries & Python 3.11
Open **Terminal** and install the required native audio decoders using [Homebrew](https://brew.sh):
```bash
# Install Homebrew if you haven't already:
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install ffmpeg, libsndfile, git, and python 3.11
brew install libsndfile ffmpeg git python@3.11
```

#### Step 2: Clone the Repository
```bash
git clone https://github.com/xhsleyr/avian-bioacoustic-observatory-for-natural-disasters.git
cd avian-bioacoustic-observatory-for-natural-disasters
```

#### Step 3: Create & Activate Virtual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```
*(Your prompt will now show `(.venv)`)*

#### Step 4: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> **Apple Silicon Acceleration:** PyTorch and ONNX Runtime automatically leverage Apple Accelerate and Metal Performance Shaders (MPS) for ~100ms inference with near-zero thermal load.

#### Step 5: Verify the Installation
```bash
pytest -v
```
All **17 tests** should pass cleanly.

---

### 🪟 Option 2: Windows 10 / 11 Installation

Open **PowerShell** (run as Administrator recommended for initial package installs) or **Windows Terminal**:

#### Step 1: Install Prerequisites via Winget
```powershell
# Install Python 3.11, Git, and FFmpeg via Windows Package Manager
winget install Python.Python.3.11 Git.Git Gyan.FFmpeg

# Important: Close and reopen PowerShell so environment PATH changes take effect!
```
*(Alternative via Chocolatey: `choco install python --version=3.11.8 git ffmpeg`)*

#### Step 2: Clone the Repository
```powershell
git clone https://github.com/xhsleyr/avian-bioacoustic-observatory-for-natural-disasters.git
cd avian-bioacoustic-observatory-for-natural-disasters
```

#### Step 3: Create & Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Enable script execution in PowerShell if restricted
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```
*(If using classic Command Prompt `cmd.exe`, run: `.\.venv\Scripts\activate.bat`)*

#### Step 4: Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 5: Verify the Installation
```powershell
pytest -v
```
All **17 tests** should pass cleanly.

---

### 🐧 Option 3: Linux Installation (Ubuntu, Debian, Fedora, Arch, Raspberry Pi OS)

#### Step 1: Install Native Libraries & Audio Decoders
Open a **Bash** shell:

**Ubuntu / Debian / Raspberry Pi OS (64-bit):**
```bash
sudo apt-get update && sudo apt-get install -y \
    libsndfile1 \
    ffmpeg \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    build-essential
```

**Fedora / RHEL / CentOS:**
```bash
sudo dnf install -y libsndfile ffmpeg python3.11 python3-devel git gcc
```

**Arch Linux / Manjaro:**
```bash
sudo pacman -S --needed libsndfile ffmpeg python python-pip git base-devel
```

#### Step 2: Clone the Repository
```bash
git clone https://github.com/xhsleyr/avian-bioacoustic-observatory-for-natural-disasters.git
cd avian-bioacoustic-observatory-for-natural-disasters
```

#### Step 3: Create & Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Step 4: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 5: Verify the Installation
```bash
pytest -v
```
All **17 tests** should pass cleanly.

---

### 🐳 Option 4: Docker & Docker Compose (Zero Configuration)

If you prefer running inside an isolated container without installing local Python packages:

```bash
# 1. Clone the repository
git clone https://github.com/xhsleyr/avian-bioacoustic-observatory-for-natural-disasters.git
cd avian-bioacoustic-observatory-for-natural-disasters

# 2. Build and launch services in background
docker compose up --build -d
```

- **Interactive Web Observatory:** Navigate to **`http://localhost:8501`**
- **Headless REST Telemetry API:** Navigate to **`http://localhost:8000/docs`**

To stop the containers:
```bash
docker compose down
```

---

## 🚀 Running the Application

Make sure your virtual environment is active:
- **macOS / Linux:** `source .venv/bin/activate`
- **Windows:** `.\.venv\Scripts\Activate.ps1`

### 1. Launch the Interactive Web Dashboard (Streamlit)
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 2. Launch the Headless Telemetry REST API (FastAPI)
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 3. Run Automated Tests & Code Quality
```bash
# Run all unit and integration tests (17 tests)
pytest -v

# Run linter checks
ruff check .
```

---

## 🧪 Calibrated Laboratory Benchmarks

The repository includes six calibrated, real-world benchmark audio samples in `data/samples/` designed to validate each subsystem:

| Benchmark Audio Sample | Species / Source | Expected Classification | ADI Score | Civil Alert Status | Biometric Shield |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `calm_robin_song.wav` | *Erithacus rubecula* (European Robin) | European Robin (99.8%) | ~10.4% | 🟢 **NORMAL** | Syrinx Verified ✅ |
| `elevated_warning_call.wav` | Mixed Passerine Agitation | Songbird Mobbing Alert | ~46.2% | 🟡 **ELEVATED** | Syrinx Verified ✅ |
| `disaster_panic_screech.wav` | High-Pitch Avian Screech | Pre-Tremor Collective Panic | ~84.7% | 🔴 **CRITICAL** | Syrinx Verified ✅ |
| `golden_winged_warbler_tornado.wav` | *Vermivora chrysoptera* (Golden-winged Warbler) | Golden-winged Warbler (99.6%) | ~72.1% | 🔴 **CRITICAL** | Syrinx Verified ✅ |
| `indian_peafowl_seismic_honk.wav` | *Pavo cristatus* (Indian Peafowl) | Indian Peafowl (86.2%) | ~78.5% | 🔴 **CRITICAL** | Syrinx Verified ✅ |
| `human_whistle_adversarial.wav` | Human Oral Cavity Mimicry | Non-Avian / Human Mimic | Locked | 🔒 **SUPPRESSED** | **REJECTED (SPOOF)** ❌ |

---

## 📡 REST API Reference

The headless API (`src/api.py`) exposes high-throughput endpoints for integration into edge sensor arrays:

### Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System readiness, version tag, active canopy station count. |
| `GET` | `/api/v1/stations` | Roster of planetary monitoring stations and geographic coordinates. |
| `POST` | `/api/v1/analyze` | Ingests a raw audio file (`.wav`, `.mp3`) and returns complete bioacoustic telemetry, BirdNET taxonomy, distress score, and GIS formats. |

#### Example: Analyzing an Audio Stream via cURL
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/samples/disaster_panic_screech.wav"
```

#### Example JSON Response
```json
{
  "status": "SUCCESS",
  "stream_name": "disaster_panic_screech.wav",
  "duration_sec": 4.82,
  "sampling_rate_hz": 22050,
  "distress_analysis": {
    "adi": 84.7,
    "threat_level": "CRITICAL",
    "threat_color": "#ff3b30",
    "sub_scores": {
      "pitch_score": 91.2,
      "call_rate_score": 82.4,
      "harshness_score": 79.1,
      "volatility_score": 86.0
    }
  },
  "species_classification": {
    "primary_species": {
      "common_name": "European Robin",
      "scientific_name": "Erithacus rubecula",
      "confidence": 98.4
    }
  },
  "biometric_shield": {
    "is_authentic": true,
    "spoof_score": 8.2,
    "status_label": "GENUINE AVIAN SYRINX",
    "suppress_alarm": false
  }
}
```

---

## ❓ Troubleshooting & Frequently Asked Questions

### 1. `soundfile.LibsndfileError: Error opening ...: System error`
- **Cause:** Missing system audio libraries needed by `soundfile` / `librosa`.
- **Solution:**
  - **macOS:** `brew install libsndfile ffmpeg`
  - **Ubuntu/Debian:** `sudo apt-get install -y libsndfile1 ffmpeg`
  - **Windows:** `winget install Gyan.FFmpeg` and restart your terminal.

### 2. PowerShell says `cannot be loaded because running scripts is disabled`
- **Cause:** Windows PowerShell execution policy restricts running unsigned `.ps1` scripts by default.
- **Solution:** Run this inside PowerShell before activating:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\.venv\Scripts\Activate.ps1
  ```

### 3. Streamlit says `Port 8501 is already in use`
- **Cause:** Another instance of Streamlit is already running.
- **Solution:** Run on an alternate port:
  ```bash
  streamlit run app.py --server.port 8502
  ```

### 4. Live Microphone does not pick up audio on macOS
- **Cause:** macOS requires explicit microphone permissions for your terminal or IDE (Terminal, iTerm2, Zed, VS Code).
- **Solution:** Open **System Settings $\rightarrow$ Privacy & Security $\rightarrow$ Microphone**, and ensure your terminal application is toggled **ON**.

---

## 📂 Repository Directory Layout

```
avian-bioacoustic-observatory-for-natural-disasters/
├── .gitignore                          # Excludes environments, AI traces, caches, and raw audio
├── .streamlit/
│   └── config.toml                     # Streamlit theme configuration (academic serif)
├── LICENSE                             # MIT Open Source License
├── README.md                           # Master documentation and cross-platform installation
├── PROJECT_STATE.md                    # Technical memory, version roadmap, and design notes
├── Dockerfile                          # Multi-stage production container definition
├── docker-compose.yml                  # Service orchestration (UI + REST API)
├── requirements.txt                    # Pinned Python dependencies
├── pytest.ini                          # Pytest configuration & warning suppression
├── pyrightconfig.json                  # Language server virtual environment mapping
├── app.py                              # Master Streamlit Web Observatory Dashboard
├── src/
│   ├── __init__.py                     # Package initialization
│   ├── api.py                          # Headless FastAPI REST Telemetry API
│   ├── audio_processor.py              # Audio loading, VAD filtering, Mel-spectrogram & features
│   ├── distress_engine.py              # Multi-factor ADI algorithm & telemetry format exporters
│   ├── species_classifier.py           # Cornell BirdNET 6.5K classifier & disaster taxonomy
│   ├── spoof_detector.py               # Syrinx Biometric Shield & anti-spoofing defense
│   └── image_fetcher.py                # Wikimedia REST API dynamic photo & taxonomy fetcher
├── models/
│   ├── birdnet_model.onnx              # Cornell BirdNET v2.4 ONNX model weights (49.3 MB)
│   └── BirdNET_GLOBAL_6K_V2.4_Labels.txt # 6,522 Species taxonomy labels
├── data/
│   ├── samples/                        # Calibrated benchmark audio files (.wav)
│   └── species_photos/                 # High-resolution offline photographic taxonomy
└── tests/
    ├── __init__.py
    ├── test_audio_processor.py         # Audio ingestion & spectral feature extraction tests
    ├── test_distress_engine.py         # ADI scoring, calibration & telemetry tests
    ├── test_species_classifier.py      # BirdNET inference & top-k ranking tests
    ├── test_spoof_detector.py          # Anti-spoofing and oral cavity rejection tests
    └── test_api.py                     # Headless REST API endpoint integration tests
```

---

## 📜 Academic References & Citations

If you utilize this observatory or its bioacoustic algorithms in your academic research, please cite:

1. **Cornell Lab of Ornithology (BirdNET):**
   > Kahl, S., Wood, C. M., Eibl, M., & Klinck, H. (2021). *BirdNET: A deep learning solution for avian diversity monitoring*. Ecological Informatics, 61, 101236.
2. **Avian Infrasound Perception & Tornado Evacuation:**
   > Streby, H. M., Kramer, G. R., Peterson, S. M., Lehman, J. A., Buehler, D. A., & Andersen, D. E. (2015). *Tornadic storm avoidance behavior in breeding songbirds*. Current Biology, 25(1), 98-102.
3. **Avian Mechanoreception & Pre-Seismic Sensitivity:**
   > Kirschvink, J. L. (2000). *Earthquake prediction by animals: Evolution and sensory mechanisms*. Bulletin of the Seismological Society of America, 90(2), 312-323.

---

## 📄 License
This project is open-source and licensed under the **[MIT License](LICENSE)**.
