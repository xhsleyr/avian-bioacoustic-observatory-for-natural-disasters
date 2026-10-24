"""Project INTERA — Avian Bioacoustic Translation & Planetary Early Warning Intelligence.
Designed in the visual system of Project CETI with strict Times New Roman editorial typography.

- Visual theme: Fluid motion-blurred iridescent gradient aura (Cyan #befff7 -> Electric Blue #4749fe -> Deep Purple #4d1bc7)
- Strict Times New Roman serif across all DOM elements, metrics, tables, and Plotly figures
- Perfectly centered, balanced layout with enlarged central content and zero empty white blocks
- Complete elimination of sidebar and toggle arrow
- Zero-lag, instant 0ms cached neural species identification across 6,522 global species
- Biometric Dual Syrinx Anti-Spoofing & Planetary Disaster Intelligence
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

# --- FIGMA / ICONS8 OUTLINE SVG ICONS ---
ICON_SHIELD = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>"""
ICON_ALERT = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>"""
ICON_CHECK = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>"""
ICON_CROSS = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#e63946" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>"""
ICON_MIC = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-3px; margin-right:6px;"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>"""

# Page configuration — full-width, clean layout without sidebar
st.set_page_config(
    page_title="INTERA — Listening to the Sentinels",
    page_icon="https://cdn.prod.website-files.com/643ddd7ffdf12273933a8cec/645bdf9d4bac53320316d064_CETI-Favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Project CETI Design System + Strict Times New Roman Typography CSS
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
        --ceti-text-muted: #8c93a8;
        --ceti-surface-dark: #181b22;
        --ceti-border-light: #e2e5ec;
        --ceti-cyan: #befff7;
    }

    /* Universal Times New Roman Font Across ALL Elements */
    html, body, .stApp, 
    h1, h2, h3, h4, h5, h6, p, span, div, label, input, button, select, textarea,
    table, th, td, tr, caption, small, strong, b, em, i, a,
    .stMarkdown, .stMarkdown *, .stCaption, .stCaption *, .stMetric, .stMetric *,
    .stSelectbox, .stRadio, .stTabs, [data-baseweb="tab"], [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        font-family: 'Times New Roman', Times, 'Nimbus Roman No9 L', Georgia, serif !important;
    }

    svg, svg * {
        font-family: inherit !important;
    }

    /* COMPLETELY REMOVE SIDEBAR AND LEFT TOGGLE ARROW BUTTON */
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

    /* Atmospheric Motion-Blurred Iridescent Gradient Canvas */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
        background-image: 
            radial-gradient(circle at 15% 15%, rgba(190, 255, 247, 0.45) 0%, transparent 45%),
            radial-gradient(circle at 85% 20%, rgba(71, 73, 254, 0.20) 0%, transparent 50%),
            radial-gradient(circle at 50% 85%, rgba(77, 27, 199, 0.16) 0%, transparent 55%) !important;
        background-attachment: fixed !important;
        color: var(--ceti-text-primary) !important;
    }

    /* Main Container Padding — Centered with Balanced Max Width */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        max-width: 1180px !important;
        margin: 0 auto !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
        display: none !important;
    }

    /* Global Headings */
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
        padding: 14px 0 20px 0;
        border-bottom: 1px solid var(--ceti-border-light);
        margin-bottom: 38px;
        flex-wrap: wrap;
        gap: 16px;
    }

    .ceti-logo-mark {
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 0.14em;
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
        gap: 28px;
        font-size: 13px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--ceti-text-muted);
        flex-wrap: wrap;
    }

    /* Centered Hero Section */
    .ceti-hero-wrap {
        text-align: center;
        margin-bottom: 36px;
    }

    .ceti-hero-eyebrow {
        font-size: 13px;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--ceti-primary-accent);
        margin-bottom: 12px;
        font-weight: 700;
    }

    .ceti-hero-title {
        font-size: clamp(2.4rem, 4.4vw, 3.8rem);
        font-weight: 700;
        line-height: 1.18;
        color: var(--ceti-text-dark);
        margin-bottom: 18px;
        letter-spacing: -0.015em;
        max-width: 960px;
        margin-left: auto;
        margin-right: auto;
    }

    .ceti-hero-subtitle {
        font-size: 1.22rem;
        line-height: 1.65;
        color: var(--ceti-text-primary);
        max-width: 880px;
        margin: 0 auto 36px auto;
        font-weight: 400;
    }

    /* Content Cards & Streamlit Containers */
    .ceti-content-card,
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid var(--ceti-border-light) !important;
        border-radius: 0px !important;
        box-shadow: 0 4px 20px -2px rgba(77, 27, 199, 0.04) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        margin-bottom: 24px !important;
    }

    .ceti-content-card:hover,
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: var(--ceti-primary-accent) !important;
        box-shadow: 0 8px 30px -4px rgba(71, 73, 254, 0.08) !important;
    }

    .ceti-content-card {
        padding: 30px 36px;
    }

    /* SUPPRESS ANY EMPTY WHITE CONTAINERS */
    .ceti-content-card:empty,
    [data-testid="stVerticalBlockBorderWrapper"]:empty,
    [data-testid="stVerticalBlockBorderWrapper"] > div:empty {
        display: none !important;
    }

    /* Signature CETI Gradient Banner */
    .ceti-gradient-banner {
        background: linear-gradient(135deg, #befff7 0%, #4749fe 50%, #4d1bc7 100%);
        color: #ffffff;
        padding: 36px 40px;
        border-radius: 0px;
        margin-bottom: 26px;
        position: relative;
        overflow: hidden;
    }

    .ceti-gradient-banner h2 {
        font-size: 2.3rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 10px;
        line-height: 1.25;
    }

    .ceti-gradient-banner p {
        font-size: 1.15rem;
        line-height: 1.65;
        color: rgba(255, 255, 255, 0.98) !important;
        max-width: 820px;
        margin: 0;
    }

    /* Pill Badges */
    .ceti-pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 18px;
        border-radius: 100px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.05em;
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

    /* Center-aligned Radio Filters */
    div[data-testid="stRadio"] {
        text-align: center !important;
        margin: 0 auto 20px auto !important;
    }
    div[data-testid="stRadio"] > label {
        font-size: 13px !important;
        letter-spacing: 0.10em !important;
        text-transform: uppercase !important;
        color: var(--ceti-text-dark) !important;
        font-weight: 700 !important;
        margin-bottom: 12px !important;
        display: block !important;
        text-align: center !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        justify-content: center !important;
        gap: 12px !important;
        flex-wrap: wrap !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background-color: #ffffff !important;
        border: 1px solid var(--ceti-border-light) !important;
        border-radius: 100px !important;
        padding: 9px 24px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
        font-size: 13.5px !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        border-color: var(--ceti-primary-accent) !important;
    }

    /* Centered Selectbox */
    div[data-testid="stSelectbox"] {
        margin: 0 auto !important;
    }
    div[data-testid="stSelectbox"] label {
        font-size: 13.5px !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: var(--ceti-text-dark) !important;
        font-weight: 700 !important;
        text-align: center !important;
        display: block !important;
        margin-bottom: 6px !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        border-radius: 0px !important;
        border: 1px solid var(--ceti-border-light) !important;
        background-color: #ffffff !important;
        color: #000000 !important;
        font-size: 14px !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--ceti-primary-accent) !important;
        color: #ffffff !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 11px 36px !important;
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
        gap: 24px;
        border-bottom: 1px solid var(--ceti-border-light);
        background: transparent;
        flex-wrap: wrap;
        justify-content: center;
        margin-bottom: 18px;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 12.5px !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: var(--ceti-text-muted) !important;
        border-radius: 0px !important;
        padding: 12px 6px !important;
        background: transparent !important;
        white-space: nowrap !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--ceti-primary-accent) !important;
        border-bottom: 2px solid var(--ceti-primary-accent) !important;
        font-weight: 700 !important;
    }

    /* Enlarged Central Metrics Sizing & Spacing */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: var(--ceti-text-dark) !important;
        line-height: 1.2 !important;
        word-break: break-word !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 13px !important;
        letter-spacing: 0.07em !important;
        text-transform: uppercase !important;
        color: var(--ceti-text-muted) !important;
        line-height: 1.35 !important;
        white-space: normal !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 12.5px !important;
        line-height: 1.3 !important;
    }

    /* Enlarged Species Card Typography */
    .ceti-species-heading {
        font-size: clamp(2.3rem, 4vw, 3.3rem);
        font-weight: 700;
        color: var(--ceti-text-dark);
        line-height: 1.15;
        margin: 6px 0 2px 0;
        word-break: break-word;
        overflow-wrap: break-word;
    }

    .ceti-species-sci {
        font-size: 1.25rem;
        font-style: italic;
        color: var(--ceti-primary-accent);
        margin-bottom: 16px;
    }

    .ceti-species-desc {
        font-size: 1.18rem;
        color: #374151;
        line-height: 1.7;
        margin-bottom: 18px;
    }

    /* Diagnostic Table */
    table {
        border-collapse: collapse !important;
        width: 100% !important;
        font-size: 14px !important;
    }
    thead tr th {
        font-size: 13px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
        color: var(--ceti-text-muted) !important;
        border-bottom: 2px solid var(--ceti-border-light) !important;
        padding: 12px 16px !important;
        text-align: left !important;
        font-weight: 700 !important;
    }
    tbody tr td {
        border-bottom: 1px solid #f4f5f8 !important;
        padding: 13px 16px !important;
        color: var(--ceti-surface-dark) !important;
    }

    audio {
        width: 100%;
        margin-top: 10px;
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
        title={'text': "ACOUSTIC DISTRESS INDEX (ADI)", 'font': {'size': 13, 'color': "#8c93a8", 'family': "Times New Roman, Times, serif"}},
        number={'suffix': "%", 'font': {'size': 54, 'color': "#000000", 'family': "Times New Roman, Times, serif"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#e2e5ec"},
            'bar': {'color': "#4749fe", 'thickness': 0.28},
            'bgcolor': "#ffffff",
            'borderwidth': 1,
            'bordercolor': "#e2e5ec",
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
        height=280,
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
        height=320,
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
        height=350,
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
        <span class="ceti-logo-dot"></span> PROJECT INTERA // BIOACOUSTICS
    </div>
    <div class="ceti-nav-links">
        <span>RESEARCH</span>
        <span>6,522 SPECIES</span>
        <span>ANTI-SPOOFING</span>
        <span>DISASTER INTELLIGENCE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- CENTERED HERO SECTION ---
st.markdown("""
<div class="ceti-hero-wrap">
    <div class="ceti-hero-eyebrow">RESEARCH INITIATIVE</div>
    <div class="ceti-hero-title">Understand What Birds are Saying</div>
    <div class="ceti-hero-subtitle">
        What if avian vocalizations became planetary early-warning stories? Applying advanced machine learning and bioacoustic translation to decode avian communication &mdash; forecasting earthquakes, cyclones, and atmospheric shockwaves before disaster strikes.
    </div>
</div>
""", unsafe_allow_html=True)

# Centered Bioacoustic Channel Selector
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
    col_u_l, col_u_m, col_u_r = st.columns([1, 2.8, 1])
    with col_u_m:
        uploaded_file = st.file_uploader(
            "Upload raw avian acoustic recording (.wav, .mp3, .ogg, .flac)",
            type=["wav", "mp3", "ogg", "flac"]
        )
        if uploaded_file is not None:
            audio_data = uploaded_file.read()
            audio_label = uploaded_file.name

elif input_mode == "Live Microphone Stream":
    col_m_l, col_m_m, col_m_r = st.columns([1, 2.8, 1])
    with col_m_m:
        st.markdown(f"<div style='text-align: center; margin-bottom: 6px;'><strong>{ICON_MIC} Real-Time Environmental Audio Stream</strong></div>", unsafe_allow_html=True)
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
    
    col_sel_l, col_sel_m, col_sel_r = st.columns([1, 2.8, 1])
    with col_sel_m:
        selected_sample_name = st.selectbox(
            "Select Calibrated Laboratory Benchmark:",
            options=list(sample_names.keys()),
            format_func=lambda x: {
                "calm_robin_song.wav": "[CALM] European Robin Song — Baseline Melodic",
                "elevated_warning_call.wav": "[ELEVATED] Song Sparrow Alarm — Agitation Alert",
                "critical_disaster_panic.wav": "[CRITICAL] Northern Cardinal Screech — Pre-Disaster Panic",
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
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 28px; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
        <div>
            <div style="font-size: 13px; letter-spacing: 0.10em; text-transform: uppercase; color: #4749fe; margin-bottom: 4px; font-weight: 700;">
                ACOUSTIC FEED ACQUIRED
            </div>
            <div style="font-size: 2rem; font-weight: 700; color: #000000;">
                {audio_label}
            </div>
        </div>
        <div style="font-size: 14px; color: #8c93a8;">
            Duration: <strong>{duration:.2f}s</strong> &bull; Sample Rate: <strong>{sr:,} Hz</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.audio(audio_data)

    # 1. Anti-Spoofing Biometric Verification Badge
    if spoof_info["is_authentic"]:
        st.markdown(f"""
        <div class="ceti-content-card" style="border-left: 4px solid #4749fe; padding: 20px 28px; margin-top: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <span class="ceti-pill-badge ceti-pill-verified">
                    {ICON_SHIELD} {spoof_info['status_label']}
                </span>
                <span style="font-size: 13px; color: #8c93a8;">
                    BIRDNET CONFIDENCE: <strong>{spoof_info['birdnet_conf']}%</strong> &bull; SYRINX CONFIRMED
                </span>
            </div>
            <p style="margin: 10px 0 0 0; font-size: 1.08rem; color: #4e5a7c; line-height: 1.55;">
                {spoof_info['status_summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ceti-content-card" style="border-left: 4px solid #e63946; padding: 20px 28px; margin-top: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <span class="ceti-pill-badge ceti-pill-warning">
                    {ICON_ALERT} {spoof_info['status_label']}
                </span>
                <span style="font-size: 13px; color: #e63946; font-weight: 700;">
                    ADVERSARIAL SPOOFING DETECTED
                </span>
            </div>
            <p style="margin: 10px 0 0 0; font-size: 1.08rem; color: #4e5a7c; line-height: 1.55;">
                {spoof_info['status_summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 2. Visual Species Identification Card — Enlarged Central Content & Zero-Lag Local Photo
    if spoof_info["is_authentic"]:
        visuals = cached_fetch_visuals(prim_sp['common_name'], prim_sp['scientific_name'])
        
        with st.container(border=True):
            col_photo, col_details = st.columns([1.1, 2.1])
            
            with col_photo:
                st.image(
                    visuals["image_url"],
                    caption=f"{prim_sp['common_name']} ({visuals['source']})",
                    width="stretch"
                )
            
            with col_details:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; flex-wrap: wrap; gap: 6px;">
                    <span style="background-color: #4749fe; color: #ffffff; font-size: 12px; padding: 4px 14px; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700;">
                        CORNELL BIRDNET &bull; {prim_sp['confidence']}% MATCH
                    </span>
                    <span style="font-size: 13px; color: #8c93a8;">GLOBAL DATABASE: 6,522 SPECIES</span>
                </div>
                <div class="ceti-species-heading">{prim_sp['common_name']}</div>
                <div class="ceti-species-sci">{prim_sp['scientific_name']}</div>
                <div class="ceti-species-desc">
                    {visuals['description']}
                </div>
                <div style="border-top: 1px solid #e2e5ec; padding-top: 14px; margin-top: 14px; font-size: 1.05rem;">
                    <p style="margin: 0 0 6px 0; color: #4e5a7c;">
                        <strong>Acoustic State:</strong> {prim_sp['pitch_status']} &bull; Pitch Centroid: <strong>{prim_sp['detected_pitch']:.0f} Hz</strong>
                    </p>
                    <p style="margin: 4px 0 0 0; color: #4749fe; font-weight: 700;">
                        <strong>Disaster Sensitivity:</strong> <span style="font-weight: 400; color: #4e5a7c;">{prim_sp['disaster_sensitivity']}</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ceti-content-card" style="border-left: 4px solid #e63946; padding: 28px 36px;">
            <span style="background-color: #000000; color: #ffffff; font-size: 12px; padding: 4px 14px; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700;">
                NON-AVIAN ACOUSTIC PROFILE
            </span>
            <div class="ceti-species-heading" style="margin-top: 10px;">{prim_sp['common_name']}</div>
            <p style="font-size: 1.15rem; color: #4e5a7c; margin-top: 12px; line-height: 1.65;">
                The sound profile exhibits human oral cavity formant constraints and breath turbulence rather than authentic avian syrinx modulation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 3. Early Warning Threat Assessment Banner
    if spoof_info["suppress_alarm"]:
        st.markdown(f"""
        <div class="ceti-content-card" style="border-left: 4px solid #e63946; padding: 26px 36px;">
            <h3 style="font-size: 1.85rem; color: #e63946; margin: 0 0 8px 0;">
                {ICON_ALERT} Disaster Early Warning Trigger Suppressed
            </h3>
            <p style="color: #4e5a7c; font-size: 1.12rem; margin: 0; line-height: 1.6;">
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

    # 4. Quantitative Metrics Row (Enlarged Central Content & Clean Containerization)
    col_gauge, col_stats = st.columns([1, 1.25])

    with col_gauge:
        with st.container(border=True):
            gauge_fig = create_gauge_chart(results["adi"], results["threat_color"])
            st.plotly_chart(gauge_fig, width="stretch")

    with col_stats:
        with st.container(border=True):
            st.markdown("""<h4 style="font-size: 1.55rem; color: #000000; margin-bottom: 18px;">Bioacoustic Signal Descriptors</h4>""", unsafe_allow_html=True)
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
            st.plotly_chart(plot_mel_spectrogram(features), width="stretch")

    with tab_wave:
        with st.container(border=True):
            st.plotly_chart(plot_waveform_and_dynamics(y, sr, features), width="stretch")

    with tab_species:
        with st.container(border=True):
            st.markdown("""<h4 style="font-size: 1.55rem; margin-bottom: 18px;">Top Species Candidates (from 6,522 Global Species)</h4>""", unsafe_allow_html=True)
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
            st.markdown("""<h4 style="font-size: 1.55rem; margin-bottom: 18px;">Biometric Verification & Anti-Spoofing Assessment</h4>""", unsafe_allow_html=True)
            
            status_html = f"{ICON_CHECK} <strong>Authentic Avian Syrinx Verified</strong>" if spoof_info['is_authentic'] else f"{ICON_CROSS} <strong>Rejected (Human Oral Mimicry / Spoofing)</strong>"
            st.markdown(f"<div style='margin-bottom: 12px; font-size: 1.15rem;'>{status_html}</div>", unsafe_allow_html=True)
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
            st.markdown("""<h4 style="font-size: 1.55rem; margin-bottom: 18px;">Detailed Bioacoustic Diagnostic Parameters</h4>""", unsafe_allow_html=True)
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
    <div class="ceti-content-card" style="text-align: center; padding: 72px 36px;">
        <div style="font-size: 2.4rem; font-weight: 700; color: #000000; margin-bottom: 14px; line-height: 1.25;">
            Awaiting Bioacoustic Acoustic Feed
        </div>
        <p style="color: #4e5a7c; font-size: 1.18rem; max-width: 660px; margin: 0 auto 28px auto; line-height: 1.65;">
            Select a calibrated laboratory benchmark, upload an avian field recording, or initiate live microphonic acquisition to commence neural species identification and disaster threat forecasting.
        </p>
        <span class="ceti-pill-badge ceti-pill-verified">
            {ICON_SHIELD} 6,522 Species Active &bull; Biometric Anti-Spoofing Enabled
        </span>
    </div>
    """, unsafe_allow_html=True)
