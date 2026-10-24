"""Avian Early Warning System (AEWS) - Academic Research Portal.

A bioacoustic intelligence platform styled in prestigious Times New Roman academic
research aesthetic, featuring seamless adaptive light/dark mode support,
Cornell BirdNET species classification, anti-spoofing, and early-warning prediction.
"""

import os
import glob
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.audio_processor import load_audio, extract_acoustic_features
from src.distress_engine import compute_distress_index
from src.species_classifier import identify_species
from src.spoof_detector import evaluate_avian_authenticity
from src.image_fetcher import fetch_bird_visuals

# Page configuration
st.set_page_config(
    page_title="AEWS — Avian Bioacoustics Research Laboratory",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Seamless Adaptive Light/Dark Mode + Times New Roman Styling
st.markdown("""
<style>
    /* Global Times New Roman Typography */
    html, body, [class*="css"], .stApp, p, div, span, h1, h2, h3, h4, h5, h6, input, button, select, label {
        font-family: "Times New Roman", Times, "Nimbus Roman No9 L", Georgia, serif !important;
    }

    /* Light Mode Variables (Default) */
    :root {
        --journal-bg: #FAFAF9;
        --journal-surface: #FFFFFF;
        --journal-surface-alt: #F5F5F4;
        --journal-text: #1C1917;
        --journal-muted: #57534E;
        --journal-border: #D6D3D1;
        --journal-accent: #2A4365;
        --journal-accent-subtle: rgba(42, 67, 101, 0.08);
        --journal-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    /* Dark Mode Auto-Detection & Fix for Theme Glitches */
    @media (prefers-color-scheme: dark) {
        :root {
            --journal-bg: #0F172A;
            --journal-surface: #1E293B;
            --journal-surface-alt: #0F172A;
            --journal-text: #F8FAFC;
            --journal-muted: #94A3B8;
            --journal-border: #334155;
            --journal-accent: #60A5FA;
            --journal-accent-subtle: rgba(96, 165, 250, 0.12);
            --journal-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }
    }

    /* Support Streamlit's internal dark theme toggle */
    [data-theme="dark"], [data-theme="dark"] .stApp {
        --journal-bg: #0F172A !important;
        --journal-surface: #1E293B !important;
        --journal-surface-alt: #0F172A !important;
        --journal-text: #F8FAFC !important;
        --journal-muted: #94A3B8 !important;
        --journal-border: #334155 !important;
        --journal-accent: #60A5FA !important;
        --journal-accent-subtle: rgba(96, 165, 250, 0.12) !important;
        --journal-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
    }

    .stApp {
        background-color: var(--journal-bg) !important;
        color: var(--journal-text) !important;
    }

    /* Academic Journal Header Masthead */
    .journal-masthead {
        border-bottom: 2px solid var(--journal-border);
        padding-bottom: 12px;
        margin-bottom: 28px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--journal-muted);
    }

    .journal-title {
        font-size: 2.8rem;
        font-weight: 400;
        color: var(--journal-text);
        line-height: 1.15;
        margin-bottom: 8px;
        letter-spacing: -0.01em;
    }

    .journal-subtitle {
        font-size: 1.12rem;
        font-style: italic;
        color: var(--journal-muted);
        line-height: 1.6;
        margin-bottom: 32px;
        max-width: 900px;
    }

    /* Scholarly Content Card (Adaptive, Zero-Glitch) */
    .journal-card {
        background-color: var(--journal-surface) !important;
        border: 1px solid var(--journal-border) !important;
        border-radius: 2px;
        padding: 24px 28px;
        margin-bottom: 24px;
        color: var(--journal-text) !important;
        box-shadow: var(--journal-shadow);
    }

    .journal-section-heading {
        font-size: 1.35rem;
        font-weight: 400;
        color: var(--journal-text);
        border-bottom: 1px solid var(--journal-border);
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Species Card Typography */
    .journal-species-title {
        font-size: 2.2rem;
        font-weight: 400;
        color: var(--journal-text);
        line-height: 1.1;
        margin-bottom: 4px;
    }

    .journal-scientific-name {
        font-size: 1.05rem;
        font-style: italic;
        color: var(--journal-muted);
        margin-bottom: 12px;
    }

    /* Academic Status Badges */
    .journal-badge {
        display: inline-block;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 3px 10px;
        border: 1px solid var(--journal-border);
        background-color: var(--journal-accent-subtle);
        color: var(--journal-text);
        margin-bottom: 8px;
    }

    /* Streamlit Button Restyling (Scholarly serif button) */
    .stButton > button {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 0.95rem !important;
        background-color: var(--journal-surface) !important;
        color: var(--journal-text) !important;
        border: 1px solid var(--journal-border) !important;
        border-radius: 2px !important;
        padding: 8px 24px !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background-color: var(--journal-text) !important;
        color: var(--journal-surface) !important;
        border-color: var(--journal-text) !important;
    }

    /* Fix Streamlit Tab Headers in Times New Roman */
    .stTabs [data-baseweb="tab"] {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 1rem !important;
        color: var(--journal-muted) !important;
        padding: 10px 18px !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--journal-text) !important;
        border-bottom: 2px solid var(--journal-text) !important;
        font-weight: 600 !important;
    }

    /* Fix Table Text Contrast */
    table, th, td {
        font-family: "Times New Roman", Times, serif !important;
        color: var(--journal-text) !important;
        border-color: var(--journal-border) !important;
    }
</style>
""", unsafe_allow_html=True)


def create_gauge_chart(adi_score: float, threat_color: str) -> go.Figure:
    """Create a clean academic gauge indicator formatted in Times New Roman."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=adi_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Acoustic Distress Index (ADI)", 'font': {'size': 14, 'family': "Times New Roman", 'color': "#57534E"}},
        number={'suffix': "%", 'font': {'size': 42, 'color': threat_color, 'family': "Times New Roman"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#A8A29E"},
            'bar': {'color': threat_color, 'thickness': 0.26},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 1,
            'bordercolor': "#D6D3D1",
            'steps': [
                {'range': [0, 38], 'color': 'rgba(34, 197, 94, 0.15)'},
                {'range': [38, 70], 'color': 'rgba(234, 179, 8, 0.15)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.20)'}
            ],
            'threshold': {
                'line': {'color': "#EF4444", 'width': 3},
                'thickness': 0.85,
                'value': 70.0
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Times New Roman")
    )
    return fig


def plot_mel_spectrogram(features: dict) -> go.Figure:
    """Create a publication-quality Mel-Spectrogram heatmap."""
    mel_db = features["mel_db"]
    times = features["times"]
    freq_bins = np.linspace(200, 10000, mel_db.shape[0])

    fig = go.Figure(data=go.Heatmap(
        z=mel_db,
        x=times,
        y=freq_bins,
        colorscale="Viridis",
        colorbar=dict(title=dict(text="dB", font=dict(family="Times New Roman", size=12)), len=0.8)
    ))
    fig.update_layout(
        title=dict(text="Fig. 1 — Mel-Frequency Bioacoustic Spectrogram (200 Hz – 10 kHz)", font=dict(family="Times New Roman", size=16)),
        xaxis=dict(title=dict(text="Time (seconds)", font=dict(family="Times New Roman", size=13)), showgrid=False),
        yaxis=dict(title=dict(text="Frequency (Hz)", font=dict(family="Times New Roman", size=13)), showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Times New Roman")
    )
    return fig


def plot_waveform_and_dynamics(y: np.ndarray, sr: int, features: dict) -> go.Figure:
    """Plot synchronized waveform and pitch trajectory in academic paper style."""
    time_wave = np.linspace(0, len(y) / sr, len(y))
    times_feats = features["times"]
    centroid = features["spectral_centroid"]["trajectory"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=("Fig. 2A — Raw Acoustic Signal Waveform & Call Onset Bursts", "Fig. 2B — Spectral Centroid / Pitch Center of Mass (Hz)")
    )

    step = max(1, len(y) // 3000)
    fig.add_trace(
        go.Scatter(x=time_wave[::step], y=y[::step], mode='lines', name='Waveform', line=dict(color="#2563EB", width=1.1)),
        row=1, col=1
    )

    onsets = features.get("onset_times", [])
    if len(onsets) > 0:
        fig.add_trace(
            go.Scatter(
                x=onsets,
                y=[0.85] * len(onsets),
                mode='markers',
                name='Call Bursts',
                marker=dict(symbol="triangle-down", size=7, color="#DC2626")
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(x=times_feats, y=centroid, mode='lines', name='Centroid (Hz)', line=dict(color="#B45309", width=1.6)),
        row=2, col=1
    )

    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        font=dict(family="Times New Roman")
    )
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(title_text="Amplitude", row=1, col=1, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1, gridcolor="rgba(128,128,128,0.15)")

    return fig


# --- SIDEBAR (Scholarly Research Documentation) ---
with st.sidebar:
    st.markdown("""
    <div style="font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em; color: #57534E; margin-bottom: 6px;">
        ENGINEERING DISSERTATION
    </div>
    <div style="font-size: 1.6rem; font-weight: 400; line-height: 1.15; margin-bottom: 8px;">
        Avian Bioacoustics Early Warning System
    </div>
    <div style="font-size: 0.92rem; font-style: italic; color: #78716C; margin-bottom: 20px;">
        Predicting Natural Disasters via Animal Communication.
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("##### 📚 Theoretical Framework")
    st.caption("""
    * **Compressional P-waves:** Birds register early seismic ground vibrations before destructive shear S-waves.
    * **Atmospheric Infrasound:** Sensing low-frequency storm waves (< 20 Hz) produced by approaching cyclones.
    * **Dual Syrinx Biphonation:** Distinguishing authentic avian vocal production from human oral cavity whistling.
    """)

    st.divider()
    st.markdown("##### ⚙️ Laboratory Specifications")
    st.markdown("""
    * **Neural Model:** Cornell BirdNET Global (6,522 Species)
    * **Inference Core:** Apple Silicon M1 (Local Execution)
    * **Anti-Spoofing:** Dual-Syrinx Biometric Gate Active
    """)


# --- MAIN SCHOLARLY MASTHEAD ---
st.markdown("""
<div class="journal-masthead">
    <span>Department of Engineering // Final Year Research Project</span>
    <span>Local Inference Core // Apple Silicon M1</span>
    <span>Catalog: 6,522 Species</span>
</div>
<div class="journal-title">
    Avian Vocal Bioacoustics for Natural Disaster Early Warning
</div>
<div class="journal-subtitle">
    An automated computational framework evaluating avian acoustic panic patterns, dual-syrinx biometrics, and multi-factor distress indices to forecast imminent seismic and meteorological catastrophes.
</div>
""", unsafe_allow_html=True)

# Select Input Channel
input_mode = st.radio(
    "Select Bioacoustic Acquisition Channel:",
    options=[
        "📂 Upload Field Recording",
        "🎙️ Live Microphone Acquisition (Real-Time)",
        "🧪 Calibrated Laboratory Benchmarks (Instant Demo)"
    ],
    horizontal=True
)

audio_data = None
audio_label = ""

if input_mode == "📂 Upload Field Recording":
    uploaded_file = st.file_uploader(
        "Upload raw avian acoustic recording (.wav, .mp3, .ogg, .flac)",
        type=["wav", "mp3", "ogg", "flac"]
    )
    if uploaded_file is not None:
        audio_data = uploaded_file.read()
        audio_label = uploaded_file.name

elif input_mode == "🎙️ Live Microphone Acquisition (Real-Time)":
    st.markdown("##### 🎙️ Real-Time Microphone Sensor Stream")
    st.caption("Acquire live environmental sound from the built-in Mac microphone or external transducer. Click Record to sample ambient signals.")
    
    recorded_audio = st.audio_input(
        "Begin / Terminate Live Sampling",
        sample_rate=44100
    )
    if recorded_audio is not None:
        audio_data = recorded_audio.read()
        audio_label = f"Live Microphonic Sample ({len(audio_data)//1024} KB)"

else:
    sample_files = sorted(glob.glob("data/samples/*.wav"))
    sample_names = {
        os.path.basename(p): p for p in sample_files
    }
    
    col_sel, _ = st.columns([2, 1])
    with col_sel:
        selected_sample_name = st.selectbox(
            "Select Calibrated Laboratory Benchmark:",
            options=list(sample_names.keys()),
            format_func=lambda x: {
                "calm_robin_song.wav": "🟢 Calm Robin Whistle (Baseline Melodic)",
                "elevated_warning_call.wav": "🟡 Elevated Warning Call (Agitation Alert)",
                "critical_disaster_panic.wav": "🔴 Critical Disaster Panic Screech (Pre-Tremor Evacuation)",
                "authentic_warbler_storm_alarm.wav": "🌪️ Golden-winged Warbler (Documented 2014 Tornado Outbreak)",
                "authentic_peafowl_earthquake_alarm.wav": "🦚 Indian Peafowl (Documented Seismic P-Wave Honk)",
                "human_whistling_mimic.wav": "👤 Human Whistle / Oral Cavity Mimicry (Anti-Spoofing Test)"
            }.get(x, x)
        )
    
    if selected_sample_name:
        with open(sample_names[selected_sample_name], "rb") as f:
            audio_data = f.read()
        audio_label = selected_sample_name

if audio_data is not None:
    st.markdown(f"""
    <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--journal-muted); margin-top: 24px; margin-bottom: 2px;">
        Signal Acquired for Analysis
    </div>
    <div style="font-size: 1.4rem; margin-bottom: 10px;">
        {audio_label}
    </div>
    """, unsafe_allow_html=True)
    st.audio(audio_data)

    with st.spinner("Executing bioacoustic feature extraction & BirdNET classification..."):
        y, sr, duration = load_audio(audio_data)
        features = extract_acoustic_features(y, sr)
        results = compute_distress_index(features)
        species_info = identify_species(audio_data, features, top_k=5)
        spoof_info = evaluate_avian_authenticity(y, sr, species_info, features)

    prim_sp = species_info["primary_species"]

    # 1. Biometric Anti-Spoofing Verification Banner
    if spoof_info["is_authentic"]:
        st.markdown(f"""
        <div class="journal-card" style="border-left: 4px solid #10B981 !important; padding: 16px 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="journal-badge" style="background-color: rgba(16, 185, 129, 0.12); color: #047857; border-color: #059669;">
                    ✓ {spoof_info['status_label']}
                </span>
                <span style="font-size: 0.85rem; color: var(--journal-muted);">
                    BirdNET Neural Confidence: <strong>{spoof_info['birdnet_conf']}%</strong> &bull; Syrinx Acoustics Verified
                </span>
            </div>
            <p style="margin: 6px 0 0 0; font-size: 0.95rem; color: var(--journal-muted);">
                {spoof_info['status_summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="journal-card" style="border-left: 4px solid #DC2626 !important; padding: 16px 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="journal-badge" style="background-color: rgba(220, 38, 38, 0.12); color: #B91C1C; border-color: #DC2626;">
                    ⚠ {spoof_info['status_label']}
                </span>
                <span style="font-size: 0.85rem; color: #DC2626;">
                    Adversarial Non-Avian Signal Detected
                </span>
            </div>
            <p style="margin: 6px 0 0 0; font-size: 0.95rem; color: var(--journal-muted);">
                {spoof_info['status_summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 2. Dual-Column Species Identification Card
    if spoof_info["is_authentic"]:
        visuals = fetch_bird_visuals(prim_sp['common_name'], prim_sp['scientific_name'])
        
        st.markdown('<div class="journal-card">', unsafe_allow_html=True)
        col_photo, col_details = st.columns([1, 2.3])
        
        with col_photo:
            st.image(
                visuals["image_url"],
                caption=f"{prim_sp['common_name']} ({visuals['source']})",
                use_container_width=True
            )
        
        with col_details:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span class="journal-badge">
                    CORNELL BIRDNET &bull; {prim_sp['confidence']}% CONFIDENCE
                </span>
                <span style="font-size: 0.85rem; color: var(--journal-muted);">GLOBAL DATABASE: 6,522 SPECIES</span>
            </div>
            <div class="journal-species-title">{prim_sp['common_name']}</div>
            <div class="journal-scientific-name">{prim_sp['scientific_name']}</div>
            <p style="font-size: 0.98rem; line-height: 1.6; color: var(--journal-text); margin-bottom: 12px;">
                {visuals['description']}
            </p>
            <div style="border-top: 1px solid var(--journal-border); padding-top: 10px; margin-top: 10px; font-size: 0.92rem;">
                <p style="margin: 0 0 4px 0; color: var(--journal-muted);">
                    <strong>Acoustic State:</strong> {prim_sp['pitch_status']} &bull; Pitch Centroid: <strong>{prim_sp['detected_pitch']:.0f} Hz</strong>
                </p>
                <p style="margin: 4px 0 0 0; color: var(--journal-text);">
                    <strong>Disaster Sensitivity:</strong> {prim_sp['disaster_sensitivity']}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="journal-card" style="border-left: 4px solid #DC2626 !important;">
            <span class="journal-badge">NON-AVIAN ACOUSTIC PROFILE</span>
            <div class="journal-species-title" style="margin-top: 8px;">{prim_sp['common_name']}</div>
            <p style="font-size: 0.98rem; line-height: 1.6; color: var(--journal-muted); margin-top: 8px;">
                The signal envelope conforms to human oral cavity resonance and exhalation turbulence rather than the high-frequency micro-modulations of an avian syrinx.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 3. Early Warning Threat Assessment Banner
    if spoof_info["suppress_alarm"]:
        st.markdown("""
        <div class="journal-card" style="border-left: 4px solid #F59E0B !important;">
            <h4 style="font-size: 1.35rem; color: #B45309; margin: 0 0 6px 0;">
                ⚠ Early Warning Trigger Suppressed (False-Alarm Protection Protocol)
            </h4>
            <p style="color: var(--journal-muted); font-size: 0.95rem; margin: 0;">
                The signal does not originate from an authentic wild bird. In accordance with civil disaster safety standards, automated alerts are locked to prevent spurious civilian evacuations.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="journal-card" style="border-left: 4px solid {results['threat_color']} !important;">
            <div style="font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--journal-muted); margin-bottom: 4px;">
                Catastrophe Risk Classification
            </div>
            <h3 style="font-size: 1.6rem; color: var(--journal-text); margin: 0 0 8px 0;">
                {results['alert_title']}
            </h3>
            <p style="color: var(--journal-muted); font-size: 1rem; line-height: 1.6; margin: 0;">
                {results['alert_summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 4. Quantitative Metrics Row
    col_gauge, col_stats = st.columns([1, 1.2])

    with col_gauge:
        st.markdown('<div class="journal-card" style="padding: 16px;">', unsafe_allow_html=True)
        gauge_fig = create_gauge_chart(results["adi"], results["threat_color"])
        st.plotly_chart(gauge_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_stats:
        st.markdown('<div class="journal-card" style="padding: 22px 26px;">', unsafe_allow_html=True)
        st.markdown("<div class=\"journal-section-heading\">Table 1 — Bioacoustic Signal Descriptors</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Pitch Center (Centroid)", f"{features['spectral_centroid']['mean']:.0f} Hz", delta=f"{results['sub_scores']['pitch']}% stress")
            st.metric("Call Burst Rate", f"{features['call_rate']:.2f} calls/sec", delta=f"{results['sub_scores']['call_rate']}% density")
        with c2:
            st.metric("Noise Flatness (Harshness)", f"{features['spectral_flatness']['mean']:.4f}", delta=f"{results['sub_scores']['flatness']}% chaos")
            st.metric("Breath Turbulence Ratio", f"{spoof_info['breath_ratio']}%", delta="Human Breath" if spoof_info['breath_ratio'] > 20 else "Avian Syrinx")
        st.markdown('</div>', unsafe_allow_html=True)

    # 5. Scientific Visualizers & Tables
    tab_spec, tab_wave, tab_species, tab_anti_spoof, tab_table = st.tabs([
        "📊 Spectrogram Analysis",
        "📈 Waveform & Onsets",
        "🐦 Species Taxonomy Breakdown",
        "🛡️ Biometric Verification",
        "📋 Diagnostic Data Table"
    ])

    with tab_spec:
        st.markdown('<div class="journal-card">', unsafe_allow_html=True)
        st.plotly_chart(plot_mel_spectrogram(features), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_wave:
        st.markdown('<div class="journal-card">', unsafe_allow_html=True)
        st.plotly_chart(plot_waveform_and_dynamics(y, sr, features), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_species:
        st.markdown('<div class="journal-card">', unsafe_allow_html=True)
        st.markdown("<div class=\"journal-section-heading\">Top Species Probabilities (Cornell BirdNET 6,522 Model)</div>", unsafe_allow_html=True)
        for cand in species_info["top_candidates"]:
            col_info, col_bar = st.columns([2.5, 1])
            with col_info:
                st.markdown(f"**{cand['common_name']}** (*{cand['scientific_name']}*)")
                st.caption(f"Disaster Profile: {cand['disaster_sensitivity']}")
            with col_bar:
                st.progress(min(1.0, cand['confidence'] / 100.0))
                st.caption(f"Confidence: **{cand['confidence']}%**")
            st.divider()
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_anti_spoof:
        st.markdown('<div class="journal-card">', unsafe_allow_html=True)
        st.markdown("<div class=\"journal-section-heading\">Biometric Authenticity & Anti-Spoofing Assessment</div>", unsafe_allow_html=True)
        st.write(f"**Authenticity Determination:** {'✓ Authentic Avian' if spoof_info['is_authentic'] else '✗ Rejected (Human Mimicry / Spoofing)'}")
        st.write(f"**Spoofing Likelihood Score:** `{spoof_info['spoof_score']} / 100`")
        st.write(f"**Sub-600Hz Breath Turbulence Ratio:** `{spoof_info['breath_ratio']}%` (Threshold: > 28% indicates human exhalation)")
        
        if spoof_info["spoof_factors"]:
            st.markdown("##### Detected Anti-Spoofing Flags:")
            for factor in spoof_info["spoof_factors"]:
                st.warning(f"• {factor}")
        else:
            st.success("• Dual syrinx harmonic purity confirmed. Zero human breath turbulence or oral cavity formants detected.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_table:
        st.markdown('<div class="journal-card">', unsafe_allow_html=True)
        st.markdown("<div class=\"journal-section-heading\">Table 2 — Bioacoustic Diagnostic Feature Vectors</div>", unsafe_allow_html=True)
        diag_rows = []
        for d in results["diagnostics"]:
            diag_rows.append({
                "Diagnostic Factor": d["name"],
                "Detected Value": d["value"],
                "Normal Range": d["normal_range"],
                "Stress Score (0-100)": f"{d['score']}%",
                "Status": d["status"]
            })
        st.table(diag_rows)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="journal-card" style="text-align: center; padding: 64px 32px;">
        <div style="font-size: 2rem; margin-bottom: 12px; color: var(--journal-text);">
            Awaiting Bioacoustic Acoustic Feed
        </div>
        <p style="color: var(--journal-muted); font-size: 1.05rem; max-width: 600px; margin: 0 auto 24px auto; line-height: 1.6;">
            Select a calibrated laboratory benchmark, upload an avian field recording, or initiate live microphonic acquisition to commence neural species identification and disaster threat forecasting.
        </p>
        <span class="journal-badge">
            6,522 Species Active &bull; Biometric Anti-Spoofing Enabled
        </span>
    </div>
    """, unsafe_allow_html=True)
