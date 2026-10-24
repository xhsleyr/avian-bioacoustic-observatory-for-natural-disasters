"""Avian Early Warning System (AEWS) - Web Application.

A bioacoustic intelligence platform inspired by Project CETI's design system:
featuring ethereal motion-blurred gradient atmospheres, GT Super editorial serif
typography, sharp architectural components, displays dynamic species photography, and deep
bioacoustic intelligence.
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
    page_title="AEWS // Planetary Bioacoustics Early Warning",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Project CETI Design System CSS
st.markdown("""
<style>
    /* Google Fonts: Editorial Serif (GT Super equivalent) & Sohne equivalent */
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --primary: #4d1bc7;
        --primary-accent: #4749fe;
        --primary-button-hover: #000000;
        --nav-link-hover: #4f1ccb;
        --background: #ffffff;
        --text-primary: #4e5a7c;
        --text-inverse: #ffffff;
        --text-dark: #000000;
        --text-muted: #a2a7b7;
        --surface-dark: #181b22;
        --border-light: #d6d6d6;
        --cyan-glow: #befff7;
    }

    /* Ambient Motion-Blurred Atmospheric Gradient Background */
    .stApp {
        background-color: #ffffff;
        background-image: 
            radial-gradient(circle at 15% 15%, rgba(190, 255, 247, 0.40) 0%, transparent 45%),
            radial-gradient(circle at 85% 20%, rgba(71, 73, 254, 0.18) 0%, transparent 50%),
            radial-gradient(circle at 50% 80%, rgba(77, 27, 199, 0.15) 0%, transparent 60%);
        background-attachment: fixed;
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-primary);
    }

    /* Subtle Animated Atmospheric Fog / Motion Blur Orbs */
    @keyframes cetiFloat {
        0% { transform: translate(0px, 0px) scale(1); }
        50% { transform: translate(30px, -20px) scale(1.08); }
        100% { transform: translate(-20px, 20px) scale(0.96); }
    }

    /* Project CETI Navigation & Editorial Header */
    .ceti-header-bar {
        border-bottom: 1px solid var(--border-light);
        padding: 14px 0;
        margin-bottom: 32px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
    }

    .ceti-nav-tag {
        color: var(--primary);
        font-weight: 500;
    }

    /* Editorial Display Typography (GT Super Display style) */
    .ceti-display-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 3.4rem;
        font-weight: 400;
        line-height: 1.15;
        color: var(--text-dark);
        margin-bottom: 12px;
        letter-spacing: -0.02em;
    }

    .ceti-subtitle {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.05rem;
        line-height: 1.6;
        color: var(--text-primary);
        max-width: 880px;
        margin-bottom: 36px;
    }

    /* Architectural Content Cards (0px radius, sharp edges, generous whitespace) */
    .ceti-card {
        background-color: #ffffff;
        border: 1px solid var(--border-light);
        border-radius: 0px;
        padding: 32px;
        margin-bottom: 24px;
        position: relative;
        transition: border-color 0.2s ease;
    }

    .ceti-card:hover {
        border-color: var(--primary);
    }

    /* Gradient Hero Box */
    .ceti-hero-box {
        background: linear-gradient(135deg, #befff7 0%, #4749fe 50%, #4d1bc7 100%);
        color: #ffffff;
        padding: 48px 40px;
        border-radius: 0px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }

    .ceti-hero-box h2 {
        font-family: 'Cormorant Garamond', serif;
        font-size: 2.2rem;
        font-weight: 400;
        color: #ffffff !important;
        margin-bottom: 8px;
    }

    .ceti-hero-box p {
        font-size: 1rem;
        line-height: 1.6;
        color: rgba(255, 255, 255, 0.92);
        max-width: 720px;
        margin: 0;
    }

    /* Pill Badges (100px radius for category filters / status) */
    .ceti-pill-badge {
        display: inline-flex;
        align-items: center;
        padding: 5px 16px;
        border-radius: 100px;
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }

    .ceti-pill-verified {
        background-color: rgba(71, 73, 254, 0.10);
        color: var(--primary-accent);
        border: 1px solid var(--primary-accent);
    }

    .ceti-pill-warning {
        background-color: rgba(230, 57, 70, 0.10);
        color: #E63946;
        border: 1px solid #E63946;
    }

    /* Streamlit Buttons: CETI Primary Button (0px radius, electric blue -> black hover) */
    .stButton > button {
        background-color: var(--primary-accent) !important;
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        padding: 10px 32px !important;
        border-radius: 0px !important;
        border: 1px solid #ffffff !important;
        transition: background-color 0.2s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }

    .stButton > button:hover {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    /* Streamlit Tabs: Clean CETI Underline Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid var(--border-light);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
        border-radius: 0px !important;
        padding: 12px 4px !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--primary) !important;
        border-bottom: 2px solid var(--primary) !important;
        font-weight: 600 !important;
    }

    /* Species Card Visual Styling */
    .ceti-species-name {
        font-family: 'Cormorant Garamond', serif;
        font-size: 2.2rem;
        font-weight: 400;
        color: var(--text-dark);
        line-height: 1.1;
        margin-top: 4px;
    }

    .ceti-scientific-name {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.95rem;
        font-style: italic;
        color: var(--text-muted);
        margin-bottom: 8px14px;
    }
</style>
""", unsafe_allow_html=True)


def create_gauge_chart(adi_score: float, threat_color: str) -> go.Figure:
    """Create a responsive Plotly gauge indicator styled with Project CETI palette."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=adi_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "ACOUSTIC DISTRESS INDEX (ADI)", 'font': {'size': 13, 'color': "#4e5a7c", 'family': "JetBrains Mono"}},
        number={'suffix': "%", 'font': {'size': 44, 'color': "#000000", 'family': "Cormorant Garamond"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#d6d6d6"},
            'bar': {'color': "#4749fe", 'thickness': 0.28},
            'bgcolor': "#ffffff",
            'borderwidth': 1,
            'bordercolor': "#d6d6d6",
            'steps': [
                {'range': [0, 38], 'color': 'rgba(190, 255, 247, 0.45)'},
                {'range': [38, 70], 'color': 'rgba(71, 73, 254, 0.18)'},
                {'range': [70, 100], 'color': 'rgba(77, 27, 199, 0.35)'}
            ],
            'threshold': {
                'line': {'color': "#4d1bc7", 'width': 3},
                'thickness': 0.85,
                'value': 70.0
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Plus Jakarta Sans")
    )
    return fig


def plot_mel_spectrogram(features: dict) -> go.Figure:
    """Create a Mel-Spectrogram heatmap styled with Project CETI colors."""
    mel_db = features["mel_db"]
    times = features["times"]
    freq_bins = np.linspace(200, 10000, mel_db.shape[0])

    # Project CETI Oceanic-Purple Palette
    colorscale_ceti = [
        [0.0, "#181b22"],
        [0.3, "#4d1bc7"],
        [0.65, "#4749fe"],
        [1.0, "#befff7"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=mel_db,
        x=times,
        y=freq_bins,
        colorscale=colorscale_ceti,
        colorbar=dict(title=dict(text="dB", font=dict(family="JetBrains Mono", size=11)), len=0.8)
    ))
    fig.update_layout(
        title=dict(text="Mel-Frequency Bioacoustic Spectrogram", font=dict(family="Cormorant Garamond", size=20, color="#000000")),
        xaxis=dict(title=dict(text="Time (seconds)", font=dict(family="Plus Jakarta Sans", size=12)), showgrid=False),
        yaxis=dict(title=dict(text="Frequency (Hz)", font=dict(family="Plus Jakarta Sans", size=12)), showgrid=True, gridcolor="#f1f1f1"),
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_waveform_and_dynamics(y: np.ndarray, sr: int, features: dict) -> go.Figure:
    """Plot synchronized waveform and pitch trajectory in Project CETI minimalist style."""
    time_wave = np.linspace(0, len(y) / sr, len(y))
    times_feats = features["times"]
    centroid = features["spectral_centroid"]["trajectory"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=("Acoustic Signal Waveform & Call Onset Bursts", "Pitch / Spectral Centroid Trajectory (Hz)")
    )

    step = max(1, len(y) // 3000)
    fig.add_trace(
        go.Scatter(x=time_wave[::step], y=y[::step], mode='lines', name='Waveform', line=dict(color="#4749fe", width=1.2)),
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
                marker=dict(symbol="triangle-down", size=8, color="#4d1bc7")
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(x=times_feats, y=centroid, mode='lines', name='Centroid (Hz)', line=dict(color="#4d1bc7", width=1.8)),
        row=2, col=1
    )

    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        font=dict(family="Plus Jakarta Sans")
    )
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1, gridcolor="#f1f1f1")
    fig.update_yaxes(title_text="Amplitude", row=1, col=1, gridcolor="#f1f1f1")
    fig.update_yaxes(title_text="Centroid (Hz)", row=2, col=1, gridcolor="#f1f1f1")

    return fig


# --- SIDEBAR (Project CETI Minimalist Architecture) ---
with st.sidebar:
    st.markdown("""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: #4749fe; margin-bottom: 8px;">
        RESEARCH INITIATIVE
    </div>
    <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.8rem; font-weight: 400; color: #000000; line-height: 1.1; margin-bottom: 6px;">
        Project AEWS
    </div>
    <div style="font-size: 0.88rem; color: #4e5a7c; margin-bottom: 24px; line-height: 1.5;">
        Avian Bioacoustic Translation & Planetary Early Warning Intelligence.
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("##### 🔬 Bioacoustic Principles")
    st.caption("""
    * **Non-Human Communication:** Decoding avian flock vocalizations before seismic and atmospheric disruptions.
    * **Dual Syrinx Biphonation:** Verifying authentic multi-tonal avian biological markers.
    * **Planetary Sensing:** Detecting compressional P-waves & storm infrasound (< 20 Hz).
    """)

    st.divider()
    st.markdown("##### 🌐 Neural Architecture")
    st.markdown("""
    * **Classifier:** Cornell BirdNET v2.4 (6,522 Species)
    * **Biometrics:** Syrinx vs Oral Cavity Anti-Spoofing
    * **Compute Core:** Apple Silicon M1 Local Neural Inference
    """)


# --- MAIN EDITORIAL HEADER ---
st.markdown("""
<div class="ceti-header-bar">
    <span>SYSTEM: ACTIVE // HARDWARE: APPLE SILICON M1</span>
    <span class="ceti-nav-tag">DESIGN SYSTEM // PROJECT CETI INSPIRED</span>
    <span>CATALOG: 6,522 GLOBAL SPECIES</span>
</div>
<div class="ceti-display-title">
    Listening to the Earth's Sentinels
</div>
<div class="ceti-subtitle">
    An AI-powered bioacoustic listening station that translates avian vocal communication into planetary early-warning intelligence &mdash; detecting seismic micro-tremors, tornado infrasound, and environmental anomalies before civil instruments register them.
</div>
""", unsafe_allow_html=True)

# Select Input Mode (Styled with Pill-button philosophy)
input_mode = st.radio(
    "SELECT INPUT CHANNEL:",
    options=[
        "📂 Upload Audio File",
        "🎙️ Live Microphone (Real-Time Environment)",
        "🧪 Preset Benchmark Samples (Instant Demo)"
    ],
    horizontal=True
)

audio_data = None
audio_label = ""

if input_mode == "📂 Upload Audio File":
    uploaded_file = st.file_uploader(
        "Upload raw bioacoustic field recording (.wav, .mp3, .ogg, .flac)",
        type=["wav", "mp3", "ogg", "flac"]
    )
    if uploaded_file is not None:
        audio_data = uploaded_file.read()
        audio_label = uploaded_file.name

elif input_mode == "🎙️ Live Microphone (Real-Time Environment)":
    st.markdown("##### 🎙️ Real-Time Environmental Audio Stream")
    st.caption("Point your Mac or external microphone towards the open window or canopy. Click Record to capture wild avian signals.")
    
    recorded_audio = st.audio_input(
        "Start / Stop Live Recording",
        sample_rate=44100
    )
    if recorded_audio is not None:
        audio_data = recorded_audio.read()
        audio_label = f"Live Environmental Stream ({len(audio_data)//1024} KB)"

else:
    sample_files = sorted(glob.glob("data/samples/*.wav"))
    sample_names = {
        os.path.basename(p): p for p in sample_files
    }
    
    col_sel, _ = st.columns([2, 1])
    with col_sel:
        selected_sample_name = st.selectbox(
            "Select Calibrated Field Benchmark:",
            options=list(sample_names.keys()),
            format_func=lambda x: {
                "calm_robin_song.wav": "🟢 Calm Robin Whistle (Baseline Melodic)",
                "elevated_warning_call.wav": "🟡 Elevated Warning Call (Agitation Alert)",
                "critical_disaster_panic.wav": "🔴 Critical Disaster Panic Screech (Pre-Tremor Evacuation)",
                "authentic_warbler_storm_alarm.wav": "🌪️ Golden-winged Warbler (Documented Tornado Evacuation)",
                "authentic_peafowl_earthquake_alarm.wav": "🦚 Indian Peafowl (Documented Seismic P-Wave Honk)",
                "human_whistling_mimic.wav": "👤 Human Whistle / Oral Mimicry (Anti-Spoofing Test)"
            }.get(x, x)
        )
    
    if selected_sample_name:
        with open(sample_names[selected_sample_name], "rb") as f:
            audio_data = f.read()
        audio_label = selected_sample_name

if audio_data is not None:
    st.markdown(f"""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: #4749fe; margin-top: 24px; margin-bottom: 4px;">
        INPUT STREAM ACQUIRED
    </div>
    <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.5rem; color: #000000; margin-bottom: 12px;">
        {audio_label}
    </div>
    """, unsafe_allow_html=True)
    st.audio(audio_data)

    with st.spinner("Decoding bioacoustic signals with Cornell BirdNET & verifying biometrics..."):
        y, sr, duration = load_audio(audio_data)
        features = extract_acoustic_features(y, sr)
        results = compute_distress_index(features)
        species_info = identify_species(audio_data, features, top_k=5)
        spoof_info = evaluate_avian_authenticity(y, sr, species_info, features)

    prim_sp = species_info["primary_species"]

    # 1. Anti-Spoofing & Biometric Verification Badge
    if spoof_info["is_authentic"]:
        st.markdown(f"""
        <div style="background-color: #ffffff; border-left: 3px solid #4749fe; border: 1px solid #d6d6d6; padding: 14px 20px; margin-bottom: 24px; border-radius: 0px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="ceti-pill-badge ceti-pill-verified">
                    🛡️ {spoof_info['status_label']}
                </span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a2a7b7;">
                    CONFIDENCE: {spoof_info['birdnet_conf']}% &bull; SYRINX BIOMETRICS CONFIRMED
                </span>
            </div>
            <p style="margin: 8px 0 0 0; font-size: 0.9rem; color: #4e5a7c;">
                {spoof_info['status_summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #ffffff; border-left: 3px solid #E63946; border: 1px solid #d6d6d6; padding: 14px 20px; margin-bottom: 24px; border-radius: 0px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="ceti-pill-badge ceti-pill-warning">
                    ⚠️ {spoof_info['status_label']}
                </span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #E63946;">
                    ADVERSARIAL SPOOFING DETECTED
                </span>
            </div>
            <p style="margin: 8px 0 0 0; font-size: 0.9rem; color: #4e5a7c;">
                {spoof_info['status_summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 2. Project CETI Visual Species Card (Dual Column with Dynamic High-Res Photo)
    if spoof_info["is_authentic"]:
        visuals = fetch_bird_visuals(prim_sp['common_name'], prim_sp['scientific_name'])
        
        st.markdown('<div class="ceti-card">', unsafe_allow_html=True)
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
                <span style="background-color: #4749fe; color: #ffffff; font-family: 'JetBrains Mono', monospace; font-size: 10px; padding: 3px 12px; letter-spacing: 0.08em; text-transform: uppercase;">
                    CORNELL BIRDNET &bull; {prim_sp['confidence']}% MATCH
                </span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #a2a7b7;">GLOBAL DATABASE: 6,522 SPECIES</span>
            </div>
            <div class="ceti-species-name">{prim_sp['common_name']}</div>
            <div class="ceti-scientific-name">{prim_sp['scientific_name']}</div>
            <p style="font-size: 0.93rem; color: #4e5a7c; line-height: 1.6; margin-bottom: 12px;">
                {visuals['description']}
            </p>
            <div style="border-top: 1px solid #d6d6d6; padding-top: 10px; margin-top: 10px; font-size: 0.9rem;">
                <p style="margin: 0 0 4px 0; color: #4e5a7c;">
                    <strong>Acoustic State:</strong> {prim_sp['pitch_status']} &bull; Pitch Centroid: <strong>{prim_sp['detected_pitch']:.0f} Hz</strong>
                </p>
                <p style="margin: 4px 0 0 0; color: #4749fe;">
                    <strong>Disaster Sensitivity:</strong> {prim_sp['disaster_sensitivity']}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ceti-card" style="border-left: 3px solid #E63946;">
            <span style="background-color: #000000; color: #ffffff; font-family: 'JetBrains Mono', monospace; font-size: 10px; padding: 3px 12px; letter-spacing: 0.08em; text-transform: uppercase;">
                NON-AVIAN ACOUSTIC PROFILE
            </span>
            <div class="ceti-species-name" style="margin-top: 8px;">{prim_sp['common_name']}</div>
            <p style="font-size: 0.95rem; color: #4e5a7c; margin-top: 8px; line-height: 1.6;">
                The sound profile exhibits human oral cavity formant constraints and breath turbulence rather than authentic avian syrinx modulation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 3. Early Warning Status Banner
    if spoof_info["suppress_alarm"]:
        st.markdown("""
        <div class="ceti-card" style="background-color: #fff8f8; border: 1px solid #ffccd0;">
            <h4 style="font-family: 'Cormorant Garamond', serif; font-size: 1.5rem; color: #E63946; margin: 0 0 4px 0;">
                ⚠️ Disaster Early Warning Trigger Suppressed
            </h4>
            <p style="color: #4e5a7c; font-size: 0.92rem; margin: 0;">
                Because non-avian or human acoustic mimicry was identified, automated early-warning triggers have been locked to prevent civil false alarms.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ceti-hero-box">
            <h2>{results['alert_title']}</h2>
            <p>{results['alert_summary']}</p>
        </div>
        """, unsafe_allow_html=True)

    # 4. Key Metrics Row
    col_gauge, col_stats = st.columns([1, 1.2])

    with col_gauge:
        st.markdown('<div class="ceti-card" style="padding: 16px;">', unsafe_allow_html=True)
        gauge_fig = create_gauge_chart(results["adi"], results["threat_color"])
        st.plotly_chart(gauge_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_stats:
        st.markdown('<div class="ceti-card" style="padding: 24px 28px;">', unsafe_allow_html=True)
        st.markdown("<h4 style=\"font-family: 'Cormorant Garamond', serif; font-size: 1.4rem; color: #000000; margin-bottom: 16px;\">Bioacoustic Signal Descriptors</h4>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Pitch Center (Centroid)", f"{features['spectral_centroid']['mean']:.0f} Hz", delta=f"{results['sub_scores']['pitch']}% stress")
            st.metric("Call Burst Rate", f"{features['call_rate']:.2f} calls/sec", delta=f"{results['sub_scores']['call_rate']}% density")
        with c2:
            st.metric("Noise Harshness (Flatness)", f"{features['spectral_flatness']['mean']:.4f}", delta=f"{results['sub_scores']['flatness']}% chaos")
            st.metric("Breath Turbulence Ratio", f"{spoof_info['breath_ratio']}%", delta="Human Breath" if spoof_info['breath_ratio'] > 20 else "Avian Syrinx")
        st.markdown('</div>', unsafe_allow_html=True)

    # 5. Interactive Visualizers & Detailed Diagnostics
    tab_spec, tab_wave, tab_species, tab_anti_spoof, tab_table = st.tabs([
        "📊 Mel-Spectrogram Heatmap",
        "📈 Waveform & Call Onsets",
        "🐦 Top Candidate Species (BirdNET 6.5K)",
        "🛡️ Anti-Spoofing Biometrics",
        "📋 Diagnostic Factor Breakdown"
    ])

    with tab_spec:
        st.markdown('<div class="ceti-card">', unsafe_allow_html=True)
        st.plotly_chart(plot_mel_spectrogram(features), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_wave:
        st.markdown('<div class="ceti-card">', unsafe_allow_html=True)
        st.plotly_chart(plot_waveform_and_dynamics(y, sr, features), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_species:
        st.markdown('<div class="ceti-card">', unsafe_allow_html=True)
        st.markdown("<h4 style=\"font-family: 'Cormorant Garamond', serif; font-size: 1.5rem; margin-bottom: 16px;\">Top 5 Species Candidates (from 6,522 Global Species)</h4>", unsafe_allow_html=True)
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
        st.markdown('<div class="ceti-card">', unsafe_allow_html=True)
        st.markdown("<h4 style=\"font-family: 'Cormorant Garamond', serif; font-size: 1.5rem; margin-bottom: 16px;\">Biometric Verification & Anti-Spoofing Analysis</h4>", unsafe_allow_html=True)
        st.write(f"**Authenticity Determination:** {'✅ Authentic Avian' if spoof_info['is_authentic'] else '❌ Rejected (Human Mimicry / Spoofing)'}")
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
        st.markdown('<div class="ceti-card">', unsafe_allow_html=True)
        st.markdown("<h4 style=\"font-family: 'Cormorant Garamond', serif; font-size: 1.5rem; margin-bottom: 16px;\">Detailed Bioacoustic Diagnostic Parameters</h4>", unsafe_allow_html=True)
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
    <div class="ceti-card" style="text-align: center; padding: 64px 32px;">
        <div style="font-family: 'Cormorant Garamond', serif; font-size: 2.2rem; color: #000000; margin-bottom: 10px;">
            Awaiting Bioacoustic Acoustic Feed
        </div>
        <p style="color: #4e5a7c; font-size: 0.95rem; max-width: 560px; margin: 0 auto 24px auto;">
            Select a calibrated benchmark sound, upload an avian recording, or activate your live microphone to initiate neural species classification and early warning prediction.
        </p>
        <span class="ceti-pill-badge ceti-pill-verified">
            6,522 Species Ready &bull; Anti-Spoofing Active
        </span>
    </div>
    """, unsafe_allow_html=True)
