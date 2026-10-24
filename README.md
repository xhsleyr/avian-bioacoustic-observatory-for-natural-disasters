# 🦅 Avian Early Warning System (AEWS)
### Predicting Natural Disasters via Avian Bioacoustic Communication & Acoustic Distress Analysis

A final year engineering bioacoustic intelligence system that analyzes bird vocalizations to detect collective panic signatures preceding natural disasters (e.g., pre-seismic P-waves, cyclone infrasound, and rapid barometric pressure drops).

---

## 📌 Project Overview
Animals and birds possess sensory adaptations that detect environmental disturbances long before humans or conventional civil instruments:
* **Seismic P-waves:** Birds detect fast-moving compressional P-waves preceding the destructive shear S-waves of earthquakes, triggering sudden high-frequency shrieking or mass flock evacuation.
* **Severe Storm Infrasound:** Approaching cyclones and tornadoes emit ultra-low-frequency infrasound (< 20 Hz) that birds hear hundreds of kilometers away, resulting in abnormal vocal burst densities.
* **Barometric Pressure Fluctuations:** Air-sac systems in birds trigger rapid agitation chatter when barometric pressure plunges.

This project implements a **Multi-Factor Acoustic Distress Index (ADI)** engine that processes raw bird audio and outputs real-time disaster early-warning threat levels.

---

## 🏗️ Architecture & Processing Pipeline

```
[ Raw Audio Input (.wav / .mp3) ]
                 │
                 ▼
[ Bioacoustic Feature Extraction (Librosa) ]
   ├── Voice Activity Masking (filters silent gaps)
   ├── Mel-Spectrogram (128 Mel Bins, 200 Hz - 10 kHz)
   ├── Spectral Centroid (Pitch Center of Mass)
   ├── Call Burst Rate (Onset Detection per second)
   ├── Spectral Flatness (Harmonic Purity vs Harsh Noise)
   └── RMS Energy Volatility (Call burst variance)
                 │
                 ▼
[ Acoustic Distress Index (ADI) Engine ]
   ├── Evaluates Pitch, Burst Frequency, Chaos & Volatility
   └── Classifies Threat: Normal 🟢 | Alert 🟡 | Critical 🔴
                 │
                 ▼
[ Interactive Streamlit Web Dashboard ]
   ├── Real-time Threat Gauge & Alert Banner
   ├── Interactive Heatmap Spectrogram (Plotly)
   ├── Synchronized Waveform & Call Onset Timeline
   └── Detailed Bioacoustic Diagnostics Breakdown Table
```

---

## 🚀 How to Run Locally (Mac M1 / Apple Silicon)

### 1. Activate the Virtual Environment
```bash
source .venv/bin/activate
```

### 2. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser to: **`http://localhost:8501`**

---

## 🧪 Built-in Test Samples
The system includes pre-generated benchmark samples under `data/samples/` ready for immediate demonstration:
1. `calm_robin_song.wav` -> **🟢 NORMAL (0 - 37%)** — Regular melodic whistle.
2. `elevated_warning_call.wav` -> **🟡 ELEVATED (38 - 69%)** — High-frequency agitation chatter.
3. `critical_disaster_panic.wav` -> **🔴 CRITICAL (70 - 100%)** — Chaotic screeches & rapid panic bursts.

---

## 📂 Project Structure
```
Birds Interaction and Communication/
├── app.py                  # Main Streamlit web application
├── src/
│   ├── __init__.py
│   ├── audio_processor.py  # Audio normalization, VAD masking & feature extraction
│   └── distress_engine.py  # Multi-factor Acoustic Distress Index (ADI) algorithm
├── data/
│   └── samples/            # Benchmark test audio files
├── requirements.txt        # Python dependencies
├── PROJECT_STATE.md        # Persistent project memory and development log
└── README.md               # Documentation and execution guide
```
