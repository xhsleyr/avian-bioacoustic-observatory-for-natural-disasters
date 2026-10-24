"""Avian Early Warning System (AEWS) — Designed in the visual system of Project CETI.

Typography: Strict Times New Roman serif across all headings, body text, metrics, and charts.
- Fluid motion-blurred iridescent gradient aura (Cyan #befff7 -> Electric Blue #4749fe -> Deep Purple #4d1bc7)
- Times New Roman typography for classic editorial elegance across all DOM elements
- Responsive, glitch-free full-width layout with zero empty blocks and sidebar removed
- Global Cornell BirdNET 6,522 Species Intelligence, Biometric Anti-Spoofing, and Early Warning
- Standard Figma-style vector SVG icons (Lucide / Icons8 minimal outline style)
- 60 FPS zero-lag cached neural inference pipeline
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

# --- FIGMA / ICONS8 STANDARD OUTLINE SVG ICONS ---
ICON_SHIELD = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>"""
ICON_ALERT = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>"""
ICON_CHECK = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>"""
ICON_CROSS = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#e63946" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>"""
ICON_MIC = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-3px; margin-right:6px;"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>"""
ICON_WAVE = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-3px; margin-right:6px;"><path d="M2 10v4"/><path d="M6 6v12"/><path d="M10 3v18"/><path d="M14 8v8"/><path d="M18 5v14"/><path d="M22 10v4"/></svg>"""
ICON_NETWORK = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-3px; margin-right:6px;"><rect width="6" height="6" x="9" y="2" rx="1"/><rect width="6" height="6" x="2" y="16" rx="1"/><rect width="6" height="6" x="16" y="16" rx="1"/><path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3"/><path d="M12 12V8"/></svg>"""

# Page configuration — full-width, clean layout without sidebar
st.set_page_config(
    page_title="Project AEWS — Listening to the Sentinels",
    page_icon="https://cdn.prod.website-files.com/643ddd7ffdf12273933a8cec/645bdf9d4bac53320316d064_CETI-Favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Project CETI Design System + Times New Roman Typography CSS
st.markdown("""
<style>
    :root {
        --ceti-primary: #4d1bc7;
        --ceti-primary-accent: #4749fe;
        --ceti-button-hover: #000000;
        --ceti-nav-hover: #4f1ccb;
        --ceti-bg: #ffffff;
        --ceti-text-primary: #4e5a7c;
        --ceti-text-inverse: #ffffff;
        --ceti-text-dark: #000000;
        --ceti-text-muted: #a2a7b7;
        --ceti-surface-dark: #181b22;
        --ceti-border-light: #e6e8ee;
        --ceti-cyan: #befff7;
    }

    /* Universal Times New Roman Font Across ALL Text Elements */
    html, body, .stApp, 
    h1, h2, h3, h4, h5, h6, p, span, div, label, input, button, select, textarea,
    table, th, td, tr, caption, small, strong, b, em, i, a,
    .stMarkdown, .stMarkdown *, .stCaption, .stCaption *, .stMetric, .stMetric * {
        font-family: 'Times New Roman', Times, 'Nimbus Roman No9 L', Georgia, serif !important;
    }

    /* Preserve SVG Icon rendering so vector graphics are crisp */
    svg, svg * {
        font-family: inherit !important;
    }

    /* COMPLETELY REMOVE SIDEBAR AND LEFT TOGGLE BUTTON */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    section[data-testid="stSidebar"],
    button[data-testid="stSidebarCollapseButton"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        pointer-events: none !important;
    }

    /* Atmospheric Motion-Blurred Gradient Canvas */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
        background-image: 
            radial-gradient(circle at 12% 18%, rgba(190, 255, 247, 0.45) 0%, transparent 42%),
            radial-gradient(circle at 88% 22%, rgba(71, 73, 254, 0.20) 0%, transparent 48%),
            radial-gradient(circle at 50% 85%, rgba(77, 27, 199, 0.16) 0%, transparent 55%) !important;
        background-attachment: fixed !important;
        color: var(--ceti-text-primary) !important;
    }

    /* Main Container Padding — Full Width without Sidebar */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1260px !important;
        margin: 0 auto !important;
    }

    /* Top Transparent Header */
    [data-testid="stHeader"] {
        background: transparent !important;
        display: none !important;
    }

    /* Global Typography Rules — Times New Roman */
    h1, h2, h3, h4, h5 {
        font-weight: 700 !important;
        line-height: 1.25 !important;
        color: var(--ceti-text-dark) !important;
        letter-spacing: -0.01em !important;
    }

    /* Project CETI Navigation Bar */
    .ceti-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 0 18px 0;
        border-bottom: 1px solid var(--ceti-border-light);
        margin-bottom: 32px;
        flex-wrap: wrap;
        gap: 12px;
    }

    .ceti-logo-mark {
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--ceti-text-dark);
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .ceti-logo-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--ceti-primary-accent);
        display: inline-block;
    }

    .ceti-nav-links {
        display: flex;
        gap: 24px;
        font-size: 13px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--ceti-text-muted);
        flex-wrap: wrap;
    }

    .ceti-nav-links a {
        color: var(--ceti-text-dark);
        text-decoration: none;
        transition: color 0.2s ease;
    }

    .ceti-nav-links a:hover {
        color: var(--ceti-primary);
    }

    /* Hero Section — Fluid Responsive Typography */
    .ceti-hero-title {
        font-size: clamp(2.2rem, 4.2vw, 3.4rem);
        font-weight: 700;
        line-height: 1.2;
        color: var(--ceti-text-dark);
        margin-bottom: 14px;
        letter-spacing: -0.015em;
        word-break: break-word;
        overflow-wrap: break-word;
    }

    .ceti-hero-subtitle {
        font-size: 1.15rem;
        line-height: 1.6;
        color: var(--ceti-text-primary);
        max-width: 860px;
        margin-bottom: 32px;
        font-weight: 400;
    }

    /* Content Cards & Streamlit Bordered Containers */
    .ceti-content-card,
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid var(--ceti-border-light) !important;
        border-radius: 0px !important;
        box-shadow: 0 4px 20px -2px rgba(77, 27, 199, 0.04) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        margin-bottom: 22px !important;
    }

    .ceti-content-card:hover,
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: var(--ceti-primary-accent) !important;
        box-shadow: 0 8px 30px -4px rgba(71, 73, 254, 0.08) !important;
    }

    .ceti-content-card {
        padding: 28px 32px;
    }

    /* Suppress any empty container or empty card */
    .ceti-content-card:empty,
    [data-testid="stVerticalBlockBorderWrapper"]:empty {
        display: none !important;
    }

    /* Signature CETI Gradient Banner */
    .ceti-gradient-banner {
        background: linear-gradient(135deg, #befff7 0%, #4749fe 50%, #4d1bc7 100%);
        color: #ffffff;
        padding: 38px 36px;
        border-radius: 0px;
        margin-bottom: 26px;
        position: relative;
        overflow: hidden;
    }

    .ceti-gradient-banner h2 {
        font-size: 2.1rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 8px;
        line-height: 1.25;
    }

    .ceti-gradient-banner p {
        font-size: 1.05rem;
        line-height: 1.6;
        color: rgba(255, 255, 255, 0.95) !important;
        max-width: 760px;
        margin: 0;
    }

    /* Pill Badges */
    .ceti-pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 16px;
        border-radius: 100px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .ceti-pill-verified {
        background-color: rgba(71, 73, 254, 0.08);
        color: var(--ceti-primary-accent);
        border: 1px solid var(--ceti-primary-accent);
    }

    .ceti-pill-warning {
        background-color: rgba(230, 57, 70, 0.08);
        color: #e63946;
        border: 1px solid #e63946;
    }

    /* Radio button filters */
    div[data-testid="stRadio"] > label {
        font-size: 12px !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: var(--ceti-text-dark) !important;
        font-weight: 700 !important;
        margin-bottom: 8px !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 10px !important;
        flex-wrap: wrap !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background-color: #ffffff !important;
        border: 1px solid var(--ceti-border-light) !important;
        border-radius: 100px !important;
        padding: 8px 20px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        border-color: var(--ceti-primary-accent) !important;
    }

    /* Selectbox */
    div[data-testid="stSelectbox"] label {
        font-size: 13px !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: var(--ceti-text-dark) !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        border-radius: 0px !important;
        border: 1px solid var(--ceti-border-light) !important;
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--ceti-primary-accent) !important;
        color: #ffffff !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 10px 32px !important;
        border-radius: 0px !important;
        border: 1px solid var(--ceti-primary-accent) !important;
        transition: background-color 0.2s ease, border-color 0.2s ease !important;
    }

    .stButton > button:hover {
        background-color: #000000 !important;
        border-color: #000000 !important;
        color: #ffffff !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 1px solid var(--ceti-border-light);
        background: transparent;
        flex-wrap: wrap;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 12px !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: var(--ceti-text-muted) !important;
        border-radius: 0px !important;
        padding: 10px 4px !important;
        background: transparent !important;
        white-space: nowrap !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--ceti-primary-accent) !important;
        border-bottom: 2px solid var(--ceti-primary-accent) !important;
        font-weight: 700 !important;
    }

    /* Metrics Sizing & Spacing */
    [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: var(--ceti-text-dark) !important;
        line-height: 1.25 !important;
        word-break: break-word !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 12px !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: var(--ceti-text-muted) !important;
        line-height: 1.3 !important;
        white-space: normal !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 12px !important;
        line-height: 1.3 !important;
    }

    /* Species Card Typography */
    .ceti-species-heading {
        font-size: clamp(1.8rem, 3.2vw, 2.5rem);
        font-weight: 700;
        color: var(--ceti-text-dark);
        line-height: 1.2;
        margin: 6px 0 2px 0;
        word-break: break-word;
        overflow-wrap: break-word;
    }

    .ceti-species-sci {
        font-size: 1.05rem;
        font-style: italic;
        color: var(--ceti-text-muted);
        margin-bottom: 12px;
    }

    /* Diagnostic Table */
    table {
        border-collapse: collapse !important;
        width: 100% !important;
        font-size: 13.5px !important;
    }
    thead tr th {
        font-size: 12.5px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: var(--ceti-text-muted) !important;
        border-bottom: 2px solid var(--ceti-border-light) !important;
        padding: 10px 14px !important;
        text-align: left !important;
    }
    tbody tr td {
        border-bottom: 1px solid #f4f5f8 !important;
        padding: 11px 14px !important;
        color: var(--ceti-surface-dark) !important;
    }

    audio {
        width: 100%;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --- 60 FPS ZERO-LAG PERFORMANCE CACHING PIPELINE ---
@st.cache_data(show_spinner=False)
def cached_process_audio(audio_bytes: bytes):
    """Zero-lag cached bioacoustic extraction and neural inference pipeline."""
    y, sr, duration = load_audio(audio_bytes)
    features = extract_acoustic_features(y, sr)
    results = compute_distress_index(features)
    species_info = identify_species(audio_bytes, features, top_k=5)
    spoof_info = evaluate_avian_authenticity(y, sr, species_info, features)
    return y, sr, duration, features, results, species_info, spoof_info


@st.cache_data(show_spinner=False)
def cached_fetch_visuals(common_name: str, scientific_name: str):
    """Cached species photographic imagery and taxonomy description."""
    return fetch_bird_visuals(common_name, scientific_name)


# --- PLOTLY VISUALIZATIONS (TIMES NEW ROMAN TYPOGRAPHY) ---
def create_gauge_chart(adi_score: float, threat_color: str) -> go.Figure:
    """Create a minimalist Plotly gauge indicator styled with Times New Roman."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=adi_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "ACOUSTIC DISTRESS INDEX (ADI)", 'font': {'size': 12, 'color': "#a2a7b7", 'family': "Times New Roman, Times, serif"}},
        number={'suffix': "%", 'font': {'size': 46, 'color': "#000000", 'family': "Times New Roman, Times, serif"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#e6e8ee"},
            'bar': {'color': "#4749fe", 'thickness': 0.28},
            'bgcolor': "#ffffff",
            'borderwidth': 1,
            'bordercolor': "#e6e8ee",
            'steps': [
                {'range': [0, 38], 'color': 'rgba(190, 255, 247, 0.50)'},
                {'range': [38, 70], 'color': 'rgba(71, 73, 254, 0.20)'},
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
        height=250,
        margin=dict(l=15, r=15, t=35, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Times New Roman, Times, serif")
    )
    return fig


def plot_mel_spectrogram(features: dict) -> go.Figure:
    """Create a Mel-Spectrogram heatmap with Times New Roman typography."""
    mel_db = features["mel_db"]
    times = features["times"]
    freq_bins = np.linspace(200, 10000, mel_db.shape[0])

    colorscale_ceti = [
        [0.0, "#181b22"],
        [0.32, "#4d1bc7"],
        [0.68, "#4749fe"],
        [1.0, "#befff7"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=mel_db,
        x=times,
        y=freq_bins,
        colorscale=colorscale_ceti,
        colorbar=dict(title=dict(text="dB", font=dict(family="Times New Roman, Times, serif", size=12)), len=0.8)
    ))
    fig.update_layout(
        title=dict(text="Mel-Frequency Bioacoustic Spectrogram", font=dict(family="Times New Roman, Times, serif", size=18, color="#000000")),
        xaxis=dict(title=dict(text="Time (seconds)", font=dict(family="Times New Roman, Times, serif", size=13)), showgrid=False),
        yaxis=dict(title=dict(text="Frequency (Hz)", font=dict(family="Times New Roman, Times, serif", size=13)), showgrid=True, gridcolor="#f4f5f8"),
        height=310,
        margin=dict(l=20, r=20, t=35, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_waveform_and_dynamics(y: np.ndarray, sr: int, features: dict) -> go.Figure:
    """Plot synchronized waveform and pitch trajectory with Times New Roman styling."""
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
        height=340,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        font=dict(family="Times New Roman, Times, serif")
    )
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1, gridcolor="#f4f5f8")
    fig.update_yaxes(title_text="Amplitude", row=1, col=1, gridcolor="#f4f5f8")
    fig.update_yaxes(title_text="Centroid (Hz)", row=2, col=1, gridcolor="#f4f5f8")

    return fig


# --- TOP NAVIGATION BAR ---
st.markdown("""
<div class="ceti-navbar">
    <div class="ceti-logo-mark">
        <span class="ceti-logo-dot"></span> PROJECT AEWS // BIOACOUSTICS
    </div>
    <div class="ceti-nav-links">
        <span>RESEARCH</span>
        <span>6,522 SPECIES</span>
        <span>ANTI-SPOOFING</span>
        <span>DISASTER INTELLIGENCE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
<div class="ceti-hero-title">
    Understand What Birds are Saying
</div>
<div class="ceti-hero-subtitle">
    What if avian vocalizations became planetary early-warning stories? Applying advanced machine learning and bioacoustic translation to decode avian communication &mdash; forecasting earthquakes, cyclones, and atmospheric shockwaves before disaster strikes.
</div>
""", unsafe_allow_html=True)

# Select Input Mode
input_mode = st.radio(
    "SELECT BIOACOUSTIC CHANNEL:",
    options=[
        "Calibrated Field Benchmarks",
        "Upload Audio Recording",
        "Live Microphone Stream"
    ],
    horizontal=True
)

audio_data = None
audio_label = ""

if input_mode == "Upload Audio Recording":
    uploaded_file = st.file_uploader(
        "Upload raw avian acoustic recording (.wav, .mp3, .ogg, .flac)",
        type=["wav", "mp3", "ogg", "flac"]
    )
    if uploaded_file is not None:
        audio_data = uploaded_file.read()
        audio_label = uploaded_file.name

elif input_mode == "Live Microphone Stream":
    st.markdown(f"##### {ICON_MIC} Real-Time Environmental Audio Stream", unsafe_allow_html=True)
    st.caption("Point your Mac or external microphone towards the open window or canopy. Click Record to sample ambient avian signals.")
    
    recorded_audio = st.audio_input(
        "Start / Stop Live Sampling",
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
            "Select Calibrated Laboratory Benchmark:",
            options=list(sample_names.keys()),
            format_func=lambda x: {
                "calm_robin_song.wav": "[CALM] Robin Whistle — Baseline Melodic",
                "elevated_warning_call.wav": "[ELEVATED] Warning Call — Agitation Alert",
                "critical_disaster_panic.wav": "[CRITICAL] Disaster Panic Screech — Pre-Tremor Evacuation",
                "authentic_warbler_storm_alarm.wav": "[SENTINEL] Golden-winged Warbler — Documented 2014 Tornado Outbreak",
                "authentic_peafowl_earthquake_alarm.wav": "[SENTINEL] Indian Peafowl — Documented Seismic P-Wave Honk",
                "human_whistling_mimic.wav": "[ADVERSARIAL] Human Whistling Mimicry — Anti-Spoofing Test"
            }.get(x, x)
        )
    
    if selected_sample_name:
        with open(sample_names[selected_sample_name], "rb") as f:
            audio_data = f.read()
        audio_label = selected_sample_name

if audio_data is not None:
    # 0ms Instant Response via Cached Pipeline
    with st.spinner("Decoding bioacoustic signals with Cornell BirdNET & verifying biometrics..."):
        y, sr, duration, features, results, species_info, spoof_info = cached_process_audio(audio_data)

    prim_sp = species_info["primary_species"]

    # Header for acquired feed
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 24px; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
        <div>
            <div style="font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: #4749fe; margin-bottom: 2px;">
                ACOUSTIC FEED ACQUIRED
            </div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #000000;">
                {audio_label}
            </div>
        </div>
        <div style="font-size: 13px; color: #a2a7b7;">
            Duration: {duration:.2f}s &bull; Sample Rate: {sr:,} Hz
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.audio(audio_data)

    # 1. Anti-Spoofing Biometric Verification Badge
    if spoof_info["is_authentic"]:
        st.markdown(f"""
        <div class="ceti-content-card" style="border-left: 3px solid #4749fe; padding: 18px 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <span class="ceti-pill-badge ceti-pill-verified">
                    {ICON_SHIELD} {spoof_info['status_label']}
                </span>
                <span style="font-size: 12px; color: #a2a7b7;">
                    BIRDNET CONFIDENCE: {spoof_info['birdnet_conf']}% &bull; SYRINX CONFIRMED
                </span>
            </div>
            <p style="margin: 8px 0 0 0; font-size: 1rem; color: #4e5a7c;">
                {spoof_info['status_summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ceti-content-card" style="border-left: 3px solid #e63946; padding: 18px 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <span class="ceti-pill-badge ceti-pill-warning">
                    {ICON_ALERT} {spoof_info['status_label']}
                </span>
                <span style="font-size: 12px; color: #e63946;">
                    ADVERSARIAL SPOOFING DETECTED
                </span>
            </div>
            <p style="margin: 8px 0 0 0; font-size: 1rem; color: #4e5a7c;">
                {spoof_info['status_summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 2. Visual Species Identification Card (Contained in clean bordered card without empty boxes)
    if spoof_info["is_authentic"]:
        visuals = cached_fetch_visuals(prim_sp['common_name'], prim_sp['scientific_name'])
        
        with st.container(border=True):
            col_photo, col_details = st.columns([1, 2.2])
            
            with col_photo:
                st.image(
                    visuals["image_url"],
                    caption=f"{prim_sp['common_name']} ({visuals['source']})",
                    use_container_width=True
                )
            
            with col_details:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; flex-wrap: wrap; gap: 6px;">
                    <span style="background-color: #4749fe; color: #ffffff; font-size: 11px; padding: 3px 12px; letter-spacing: 0.08em; text-transform: uppercase;">
                        CORNELL BIRDNET &bull; {prim_sp['confidence']}% MATCH
                    </span>
                    <span style="font-size: 12px; color: #a2a7b7;">GLOBAL DATABASE: 6,522 SPECIES</span>
                </div>
                <div class="ceti-species-heading">{prim_sp['common_name']}</div>
                <div class="ceti-species-sci">{prim_sp['scientific_name']}</div>
                <p style="font-size: 1rem; color: #4e5a7c; line-height: 1.6; margin-bottom: 14px;">
                    {visuals['description']}
                </p>
                <div style="border-top: 1px solid #e6e8ee; padding-top: 12px; margin-top: 12px; font-size: 0.95rem;">
                    <p style="margin: 0 0 6px 0; color: #4e5a7c;">
                        <strong>Acoustic State:</strong> {prim_sp['pitch_status']} &bull; Pitch Centroid: <strong>{prim_sp['detected_pitch']:.0f} Hz</strong>
                    </p>
                    <p style="margin: 4px 0 0 0; color: #4749fe;">
                        <strong>Disaster Sensitivity:</strong> {prim_sp['disaster_sensitivity']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ceti-content-card" style="border-left: 3px solid #e63946;">
            <span style="background-color: #000000; color: #ffffff; font-size: 11px; padding: 3px 12px; letter-spacing: 0.08em; text-transform: uppercase;">
                NON-AVIAN ACOUSTIC PROFILE
            </span>
            <div class="ceti-species-heading" style="margin-top: 8px;">{prim_sp['common_name']}</div>
            <p style="font-size: 1rem; color: #4e5a7c; margin-top: 8px; line-height: 1.6;">
                The sound profile exhibits human oral cavity formant constraints and breath turbulence rather than authentic avian syrinx modulation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 3. Early Warning Threat Assessment Banner
    if spoof_info["suppress_alarm"]:
        st.markdown(f"""
        <div class="ceti-content-card" style="border-left: 3px solid #e63946;">
            <h3 style="font-size: 1.7rem; color: #e63946; margin: 0 0 6px 0;">
                {ICON_ALERT} Disaster Early Warning Trigger Suppressed
            </h3>
            <p style="color: #4e5a7c; font-size: 1rem; margin: 0;">
                Because non-avian or human acoustic mimicry was identified, automated early-warning triggers have been locked to prevent civil false alarms.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ceti-gradient-banner">
            <h2>{results['alert_title']}</h2>
            <p>{results['alert_summary']}</p>
        </div>
        """, unsafe_allow_html=True)

    # 4. Quantitative Metrics Row (Protected from Column Collisions & Empty Box Free)
    col_gauge, col_stats = st.columns([1, 1.2])

    with col_gauge:
        with st.container(border=True):
            gauge_fig = create_gauge_chart(results["adi"], results["threat_color"])
            st.plotly_chart(gauge_fig, use_container_width=True)

    with col_stats:
        with st.container(border=True):
            st.markdown("""<h4 style="font-size: 1.5rem; color: #000000; margin-bottom: 16px;">Bioacoustic Signal Descriptors</h4>""", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Pitch Center (Centroid)", f"{features['spectral_centroid']['mean']:.0f} Hz", delta=f"{results['sub_scores']['pitch']}% stress")
                st.metric("Call Burst Rate", f"{features['call_rate']:.2f} calls/s", delta=f"{results['sub_scores']['call_rate']}% density")
            with c2:
                st.metric("Noise Harshness (Flatness)", f"{features['spectral_flatness']['mean']:.4f}", delta=f"{results['sub_scores']['flatness']}% chaos")
                st.metric("Breath Turbulence Ratio", f"{spoof_info['breath_ratio']}%", delta="Human Breath" if spoof_info['breath_ratio'] > 20 else "Avian Syrinx")

    # 5. Scientific Visualizers & Tables (Clean Containerized Tabs)
    tab_spec, tab_wave, tab_species, tab_anti_spoof, tab_table = st.tabs([
        "SPECTROGRAM HEATMAP",
        "WAVEFORM & ONSETS",
        "SPECIES TAXONOMY (6.5K)",
        "BIOMETRIC VERIFICATION",
        "DIAGNOSTIC DATA TABLE"
    ])

    with tab_spec:
        with st.container(border=True):
            st.plotly_chart(plot_mel_spectrogram(features), use_container_width=True)

    with tab_wave:
        with st.container(border=True):
            st.plotly_chart(plot_waveform_and_dynamics(y, sr, features), use_container_width=True)

    with tab_species:
        with st.container(border=True):
            st.markdown("""<h4 style="font-size: 1.5rem; margin-bottom: 16px;">Top Species Candidates (from 6,522 Global Species)</h4>""", unsafe_allow_html=True)
            for cand in species_info["top_candidates"]:
                col_info, col_bar = st.columns([2.5, 1])
                with col_info:
                    st.markdown(f"**{cand['common_name']}** (*{cand['scientific_name']}*)")
                    st.caption(f"Disaster Profile: {cand['disaster_sensitivity']}")
                with col_bar:
                    st.progress(min(1.0, cand['confidence'] / 100.0))
                    st.caption(f"Confidence: **{cand['confidence']}%**")
                st.divider()

    with tab_anti_spoof:
        with st.container(border=True):
            st.markdown("""<h4 style="font-size: 1.5rem; margin-bottom: 16px;">Biometric Verification & Anti-Spoofing Assessment</h4>""", unsafe_allow_html=True)
            
            status_html = f"{ICON_CHECK} <strong>Authentic Avian Syrinx Verified</strong>" if spoof_info['is_authentic'] else f"{ICON_CROSS} <strong>Rejected (Human Oral Mimicry / Spoofing)</strong>"
            st.markdown(f"<div style='margin-bottom: 10px;'>{status_html}</div>", unsafe_allow_html=True)
            st.write(f"**Spoofing Likelihood Score:** `{spoof_info['spoof_score']} / 100`")
            st.write(f"**Sub-600Hz Breath Turbulence Ratio:** `{spoof_info['breath_ratio']}%` (Threshold: > 28% indicates human exhalation)")
            
            if spoof_info["spoof_factors"]:
                st.markdown("##### Detected Anti-Spoofing Flags:")
                for factor in spoof_info["spoof_factors"]:
                    st.warning(f"• {factor}")
            else:
                st.success("• Dual syrinx harmonic purity confirmed. Zero human breath turbulence or oral cavity formants detected.")

    with tab_table:
        with st.container(border=True):
            st.markdown("""<h4 style="font-size: 1.5rem; margin-bottom: 16px;">Detailed Bioacoustic Diagnostic Parameters</h4>""", unsafe_allow_html=True)
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

else:
    st.markdown(f"""
    <div class="ceti-content-card" style="text-align: center; padding: 64px 32px;">
        <div style="font-size: 2.3rem; font-weight: 700; color: #000000; margin-bottom: 12px; line-height: 1.25;">
            Awaiting Bioacoustic Acoustic Feed
        </div>
        <p style="color: #4e5a7c; font-size: 1.05rem; max-width: 620px; margin: 0 auto 26px auto; line-height: 1.6;">
            Select a calibrated laboratory benchmark, upload an avian field recording, or initiate live microphonic acquisition to commence neural species identification and disaster threat forecasting.
        </p>
        <span class="ceti-pill-badge ceti-pill-verified">
            {ICON_SHIELD} 6,522 Species Active &bull; Biometric Anti-Spoofing Enabled
        </span>
    </div>
    """, unsafe_allow_html=True)
