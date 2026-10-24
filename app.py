"""Avian Early Warning System (AEWS) — Designed in the visual system of Project CETI.

Recreating https://www.projectceti.org/ exactly:
- Authentic Project CETI CDN background gradient & iridescent color atmosphere
- Official GT Super Display serif font for display headlines (white upright + black italic)
- Official Söhne font for navigation, metadata, and body
- Hero composition matching the website screenshot:
  - Top black logo badge ([ ceti ]) + clean navigation + social icons
  - Authentic sperm whale emerging with mix-blend-mode multiply
  - Glowing circular neon purple halo ring
  - 'WHAT WOULD IT MEAN TO' -> 'Understand' (white serif) -> 'What Birds are Saying' (black italic serif)
  - Floating translucent card with tote graphic, copy, and electric blue button
  - Organic white wave crest separating hero from bioacoustic tools
- Cornell BirdNET 6,522 Species Intelligence, Biometric Anti-Spoofing, and Early Warning
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
    page_title="Project CETI •-- Listening to Sentinels",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Project CETI Exact Typography & Styling CSS
st.markdown("""
<style>
    /* 1. Official Project CETI Webfonts from CDN (GT Super Display + Söhne) */
    @font-face {
        font-family: 'GT Super Display';
        src: url('https://cdn.prod.website-files.com/643ddd7ffdf12273933a8cec/64595f83c1fc195e83c9e816_GT-Super-Display-Bold.woff2') format('woff2');
        font-weight: 700;
        font-style: normal;
        font-display: swap;
    }
    @font-face {
        font-family: 'GT Super Display';
        src: url('https://cdn.prod.website-files.com/643ddd7ffdf12273933a8cec/64595f837812b5eb0a449bd4_GT-Super-Display-Bold-Italic.woff2') format('woff2');
        font-weight: 700;
        font-style: italic;
        font-display: swap;
    }
    @font-face {
        font-family: 'GT Super Display';
        src: url('https://cdn.prod.website-files.com/643ddd7ffdf12273933a8cec/64595f83f76c2508a0a92666_GT-Super-Display-Medium.woff2') format('woff2');
        font-weight: 500;
        font-style: normal;
        font-display: swap;
    }
    @font-face {
        font-family: 'Sohne';
        src: url('https://cdn.prod.website-files.com/643ddd7ffdf12273933a8cec/64596026870b9b4285fe2498_soehne-buch.woff2') format('woff2');
        font-weight: 400;
        font-style: normal;
        font-display: swap;
    }
    @font-face {
        font-family: 'Sohne';
        src: url('https://cdn.prod.website-files.com/643ddd7ffdf12273933a8cec/64596026a59218839d6d964c_soehne-dreiviertelfett.woff2') format('woff2');
        font-weight: 700;
        font-style: normal;
        font-display: swap;
    }

    /* Fallback Google Fonts if CDN delayed */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

    :root {
        --ceti-primary: #4d1bc7;
        --ceti-primary-accent: #4749fe;
        --ceti-button-hover: #000000;
        --ceti-bg: #ffffff;
        --ceti-text-primary: #4e5a7c;
        --ceti-text-dark: #000000;
        --ceti-text-muted: #a2a7b7;
        --ceti-border-light: #e6e8ee;
    }

    /* Streamlit Canvas Overrides with Ambient Fixed Atmospheric Gradient */
    .stApp {
        background-color: #ffffff !important;
        background-image: 
            radial-gradient(circle at 10% 18%, rgba(190, 255, 247, 0.45) 0%, transparent 45%),
            radial-gradient(circle at 90% 28%, rgba(71, 73, 254, 0.20) 0%, transparent 48%),
            radial-gradient(circle at 50% 85%, rgba(77, 27, 199, 0.15) 0%, transparent 55%) !important;
        background-attachment: fixed !important;
        font-family: 'Sohne', 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        color: var(--ceti-text-primary) !important;
    }

    /* Remove default Streamlit top header padding */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 4rem !important;
        max-width: 1320px !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* --- HERO CONTAINER (MATCHES SCREENSHOT EXACTLY) --- */
    .ceti-hero-wrapper {
        position: relative;
        width: 100%;
        min-height: 580px;
        background-color: #befff7;
        background-image: url('https://cdn.prod.website-files.com/643ddd7ffdf12273933a8cec/6448200903d27e9cbbec4b16_BG-0-home-gradient-wide.jpg');
        background-size: cover;
        background-position: center top;
        background-repeat: no-repeat;
        border-radius: 0px;
        overflow: hidden;
        margin-bottom: 40px;
        box-shadow: 0 10px 40px -10px rgba(71, 73, 254, 0.15);
    }

    /* Top Navigation inside Hero */
    .ceti-hero-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 24px;
        position: relative;
        z-index: 10;
    }

    .ceti-hero-brand {
        background-color: #000000;
        padding: 18px 24px 14px 24px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.2);
    }

    .ceti-hero-brand img {
        height: 26px;
        width: auto;
    }

    .ceti-hero-links {
        display: flex;
        gap: 22px;
        font-family: 'Sohne', 'Plus Jakarta Sans', sans-serif;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #000000;
        align-items: center;
    }

    .ceti-hero-links span {
        cursor: pointer;
        transition: color 0.2s ease;
    }

    .ceti-hero-links span.active {
        color: var(--ceti-primary-accent);
    }

    .ceti-hero-links span:hover {
        color: var(--ceti-primary);
    }

    .ceti-hero-socials {
        display: flex;
        flex-direction: column;
        gap: 12px;
        font-size: 13px;
        color: #000000;
        opacity: 0.75;
        padding-right: 12px;
    }

    /* Whale Image from Screenshot */
    .ceti-hero-whale {
        position: absolute;
        left: -120px;
        top: 20px;
        width: 620px;
        max-width: 52vw;
        mix-blend-mode: multiply;
        transform: rotate(4deg);
        z-index: 2;
        pointer-events: none;
        user-select: none;
    }

    /* Neon Purple Toroidal Halo Ring */
    .ceti-hero-halo {
        position: absolute;
        right: 16%;
        top: 14%;
        width: 460px;
        max-width: 40vw;
        z-index: 1;
        pointer-events: none;
        opacity: 0.95;
    }

    /* Main Hero Headline Layout */
    .ceti-hero-content {
        position: relative;
        z-index: 5;
        padding: 30px 48px 75px 48px;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        text-align: left;
    }

    .ceti-hero-text-block {
        max-width: 700px;
        width: 100%;
        margin-left: auto;
    }

    .ceti-hero-eyebrow {
        font-family: 'Sohne', 'Plus Jakarta Sans', sans-serif;
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #ffffff;
        margin-bottom: 2px;
        text-shadow: 0 1px 6px rgba(0, 0, 0, 0.2);
    }

    .ceti-hero-title-white {
        font-family: 'GT Super Display', 'Playfair Display', serif;
        font-size: 104px;
        font-weight: 700;
        color: #ffffff;
        line-height: 0.85;
        margin: 0;
        letter-spacing: -0.025em;
        text-shadow: 0 4px 24px rgba(0, 0, 0, 0.18);
    }

    .ceti-hero-title-black {
        font-family: 'GT Super Display', 'Playfair Display', serif;
        font-size: 64px;
        font-weight: 700;
        font-style: italic;
        color: #000000;
        line-height: 1.05;
        margin: 4px 0 28px 0;
        letter-spacing: -0.02em;
    }

    /* Floating Card from Screenshot */
    .ceti-floating-card {
        background-color: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(10px);
        border: 1px solid #ffffff;
        border-radius: 0px;
        padding: 22px 28px;
        display: flex;
        align-items: center;
        gap: 24px;
        max-width: 550px;
        box-shadow: 0 8px 32px rgba(71, 73, 254, 0.08);
    }

    .ceti-card-left {
        flex: 0 0 auto;
    }

    .ceti-card-badge {
        height: 64px;
        width: auto;
        object-fit: contain;
    }

    .ceti-card-right {
        flex: 1;
    }

    .ceti-card-text {
        font-family: 'Sohne', 'Plus Jakarta Sans', sans-serif;
        font-size: 13px;
        line-height: 1.45;
        color: #4e5a7c;
        margin: 0 0 12px 0;
    }

    .ceti-card-btn {
        display: inline-block;
        background-color: var(--ceti-primary-accent);
        color: #ffffff !important;
        font-family: 'Sohne', 'Plus Jakarta Sans', sans-serif;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 8px 20px;
        border-radius: 0px;
        text-decoration: none;
        transition: background-color 0.2s ease;
    }

    .ceti-card-btn:hover {
        background-color: #000000;
    }

    /* Wave Crest at bottom of Hero */
    .ceti-hero-wave {
        position: absolute;
        bottom: -1px;
        left: 0;
        width: 100%;
        line-height: 0;
        z-index: 4;
        pointer-events: none;
    }

    .ceti-hero-wave svg {
        display: block;
        width: 100%;
        height: 52px;
    }

    /* Responsive adjustments */
    @media (max-width: 900px) {
        .ceti-hero-title-white {
            font-size: 64px !important;
        }
        .ceti-hero-title-black {
            font-size: 40px !important;
        }
        .ceti-hero-whale {
            opacity: 0.35;
            left: -180px;
        }
    }

    /* --- CONTENT CARDS (0px radius, clean hairline) --- */
    .ceti-content-card {
        background-color: #ffffff;
        border: 1px solid var(--ceti-border-light);
        border-radius: 0px;
        padding: 32px 36px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(77, 27, 199, 0.04);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .ceti-content-card:hover {
        border-color: var(--ceti-primary-accent);
        box-shadow: 0 8px 30px -4px rgba(71, 73, 254, 0.08);
    }

    /* CETI Gradient Banner */
    .ceti-gradient-banner {
        background: linear-gradient(135deg, #befff7 0%, #4749fe 50%, #4d1bc7 100%);
        color: #ffffff;
        padding: 44px 40px;
        border-radius: 0px;
        margin-bottom: 28px;
    }

    .ceti-gradient-banner h2 {
        font-family: 'GT Super Display', 'Playfair Display', serif;
        font-size: 2.3rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 10px;
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
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-family: 'Space Mono', monospace;
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
        font-family: 'Space Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: var(--ceti-text-dark) !important;
        font-weight: 700 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 12px !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background-color: #ffffff !important;
        border: 1px solid var(--ceti-border-light) !important;
        border-radius: 100px !important;
        padding: 8px 20px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        border-color: var(--ceti-primary-accent) !important;
    }

    /* Selectbox */
    div[data-testid="stSelectbox"] label {
        font-family: 'Space Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.08em !important;
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
        font-family: 'Sohne', 'Plus Jakarta Sans', sans-serif !important;
        font-size: 12px !important;
        font-weight: 600 !important;
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
        gap: 28px;
        border-bottom: 1px solid var(--ceti-border-light);
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: var(--ceti-text-muted) !important;
        border-radius: 0px !important;
        padding: 12px 2px !important;
        background: transparent !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--ceti-primary-accent) !important;
        border-bottom: 2px solid var(--ceti-primary-accent) !important;
        font-weight: 700 !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'GT Super Display', 'Playfair Display', serif !important;
        font-size: 2.1rem !important;
        color: var(--ceti-text-dark) !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Space Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: var(--ceti-text-muted) !important;
    }

    /* Species Card Visual Styling */
    .ceti-species-heading {
        font-family: 'GT Super Display', 'Playfair Display', serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--ceti-text-dark);
        line-height: 1.1;
        margin: 6px 0 2px 0;
    }

    .ceti-species-sci {
        font-size: 1rem;
        font-style: italic;
        color: var(--ceti-text-muted);
        margin-bottom: 14px;
    }

    /* Diagnostic Table Clean CETI Style */
    table {
        border-collapse: collapse !important;
        width: 100% !important;
        font-family: 'Sohne', 'Plus Jakarta Sans', sans-serif !important;
        font-size: 13px !important;
    }
    thead tr th {
        font-family: 'Space Mono', monospace !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: var(--ceti-text-muted) !important;
        border-bottom: 2px solid var(--ceti-border-light) !important;
        padding: 10px 14px !important;
        text-align: left !important;
    }
    tbody tr td {
        border-bottom: 1px solid #f4f5f8 !important;
        padding: 12px 14px !important;
        color: #181b22 !important;
    }
</style>
""", unsafe_allow_html=True)


def create_gauge_chart(adi_score: float, threat_color: str) -> go.Figure:
    """Create a minimalist Plotly gauge indicator styled with Project CETI palette."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=adi_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "ACOUSTIC DISTRESS INDEX (ADI)", 'font': {'size': 11, 'color': "#a2a7b7", 'family': "Space Mono"}},
        number={'suffix': "%", 'font': {'size': 46, 'color': "#000000", 'family': "GT Super Display, Playfair Display"}},
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
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Sohne, Plus Jakarta Sans")
    )
    return fig


def plot_mel_spectrogram(features: dict) -> go.Figure:
    """Create a Mel-Spectrogram heatmap in Project CETI's signature oceanic purple palette."""
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
        colorbar=dict(title=dict(text="dB", font=dict(family="Space Mono", size=11)), len=0.8)
    ))
    fig.update_layout(
        title=dict(text="Mel-Frequency Bioacoustic Spectrogram", font=dict(family="GT Super Display, Playfair Display", size=19, color="#000000")),
        xaxis=dict(title=dict(text="Time (seconds)", font=dict(family="Sohne, Plus Jakarta Sans", size=12)), showgrid=False),
        yaxis=dict(title=dict(text="Frequency (Hz)", font=dict(family="Sohne, Plus Jakarta Sans", size=12)), showgrid=True, gridcolor="#f4f5f8"),
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
        font=dict(family="Sohne, Plus Jakarta Sans")
    )
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1, gridcolor="#f4f5f8")
    fig.update_yaxes(title_text="Amplitude", row=1, col=1, gridcolor="#f4f5f8")
    fig.update_yaxes(title_text="Centroid (Hz)", row=2, col=1, gridcolor="#f4f5f8")

    return fig


# --- HERO SECTION (MATCHES PROJECT CETI WEBSITE SCREENSHOT EXACTLY) ---
st.markdown("""
<div class="ceti-hero-wrapper">
    <!-- Top Navigation inside Hero -->
    <div class="ceti-hero-nav">
        <div class="ceti-hero-brand">
            <img src="https://cdn.prod.website-files.com/643ddd7ffdf12273933a8cec/645d24eb63dfb01f70696649_logo-ceti.svg" alt="CETI Logo" />
        </div>
        <div class="ceti-hero-links">
            <span class="active">HOME</span>
            <span>ABOUT ▾</span>
            <span>GET INVOLVED ▾</span>
            <span>RESEARCH</span>
            <span>DOMINICA</span>
            <span>NEWS</span>
            <span>DONATE</span>
            <span>SHOP</span>
        </div>
        <div class="ceti-hero-socials">
            <span>📷</span>
            <span>▶</span>
            <span>in</span>
            <span>𝕏</span>
        </div>
    </div>

    <!-- Sperm Whale from Screenshot -->
    <img src="https://cdn.prod.website-files.com/643ddd7ffdf12273933a8cec/645d2b2186435d15f32cc462_whale-home.jpg" class="ceti-hero-whale" alt="Sperm Whale" />

    <!-- Neon Purple Toroidal Halo Ring -->
    <img src="https://cdn.prod.website-files.com/643ddd7ffdf12273933a8cec/64482522a0d34620804b6acf_Element-0-home-circle.png" class="ceti-hero-halo" alt="Purple Halo Ring" />

    <!-- Hero Headlines & Floating Card -->
    <div class="ceti-hero-content">
        <div class="ceti-hero-text-block">
            <div class="ceti-hero-eyebrow">WHAT WOULD IT MEAN TO</div>
            <div class="ceti-hero-title-white">Understand</div>
            <div class="ceti-hero-title-black">What Birds are Saying</div>

            <!-- Floating Card -->
            <div class="ceti-floating-card">
                <div class="ceti-card-left">
                    <img src="https://cdn.prod.website-files.com/643ddd7ffdf12273933a8cec/66d8adc03d225ccfda813fc5_CETI-Tote-Design.png" class="ceti-card-badge" alt="Listen to the Whales" />
                </div>
                <div class="ceti-card-right">
                    <p class="ceti-card-text">
                        What if their sounds became stories? Take the plunge and discover a hidden planetary bioacoustic radar.
                    </p>
                    <a href="#bioacoustic-radar" class="ceti-card-btn">Visit now</a>
                </div>
            </div>
        </div>
    </div>

    <!-- Organic Wave Crest at Bottom -->
    <div class="ceti-hero-wave">
        <svg viewBox="0 0 1440 60" fill="none" preserveAspectRatio="none">
            <path d="M0,25 C360,65 680,5 1020,35 C1240,55 1380,20 1440,30 L1440,60 L0,60 Z" fill="#ffffff"></path>
        </svg>
    </div>
</div>
""", unsafe_allow_html=True)


# --- BIOACOUSTIC CHANNEL SELECTION ---
st.markdown('<div id="bioacoustic-radar"></div>', unsafe_allow_html=True)

input_mode = st.radio(
    "SELECT BIOACOUSTIC CHANNEL:",
    options=[
        "📂 Upload Audio Recording",
        "🎙️ Live Microphone Stream",
        "🧪 Calibrated Field Benchmarks"
    ],
    horizontal=True
)

audio_data = None
audio_label = ""

if input_mode == "📂 Upload Audio Recording":
    uploaded_file = st.file_uploader(
        "Upload raw avian acoustic recording (.wav, .mp3, .ogg, .flac)",
        type=["wav", "mp3", "ogg", "flac"]
    )
    if uploaded_file is not None:
        audio_data = uploaded_file.read()
        audio_label = uploaded_file.name

elif input_mode == "🎙️ Live Microphone Stream":
    st.markdown("##### 🎙️ Real-Time Environmental Audio Stream")
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
    <div style="font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: #4749fe; margin-top: 24px; margin-bottom: 2px;">
        ACOUSTIC FEED ACQUIRED
    </div>
    <div style="font-family: 'GT Super Display', 'Playfair Display', serif; font-size: 1.8rem; font-weight: 700; color: #000000; margin-bottom: 12px;">
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

    # 1. Anti-Spoofing Biometric Verification Badge
    if spoof_info["is_authentic"]:
        st.markdown(f"""
        <div class="ceti-content-card" style="border-left: 3px solid #4749fe; padding: 18px 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="ceti-pill-badge ceti-pill-verified">
                    🛡️ {spoof_info['status_label']}
                </span>
                <span style="font-family: 'Space Mono', monospace; font-size: 11px; color: #a2a7b7;">
                    BIRDNET CONFIDENCE: {spoof_info['birdnet_conf']}% &bull; SYRINX CONFIRMED
                </span>
            </div>
            <p style="margin: 8px 0 0 0; font-size: 0.92rem; color: #4e5a7c;">
                {spoof_info['status_summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ceti-content-card" style="border-left: 3px solid #e63946; padding: 18px 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="ceti-pill-badge ceti-pill-warning">
                    ⚠️ {spoof_info['status_label']}
                </span>
                <span style="font-family: 'Space Mono', monospace; font-size: 11px; color: #e63946;">
                    ADVERSARIAL SPOOFING DETECTED
                </span>
            </div>
            <p style="margin: 8px 0 0 0; font-size: 0.92rem; color: #4e5a7c;">
                {spoof_info['status_summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 2. Visual Species Identification Card (Dual Column with Dynamic High-Res Photo)
    if spoof_info["is_authentic"]:
        visuals = fetch_bird_visuals(prim_sp['common_name'], prim_sp['scientific_name'])
        
        st.markdown('<div class="ceti-content-card">', unsafe_allow_html=True)
        col_photo, col_details = st.columns([1, 2.2])
        
        with col_photo:
            st.image(
                visuals["image_url"],
                caption=f"{prim_sp['common_name']} ({visuals['source']})",
                use_container_width=True
            )
        
        with col_details:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="background-color: #4749fe; color: #ffffff; font-family: 'Space Mono', monospace; font-size: 10px; padding: 3px 12px; letter-spacing: 0.08em; text-transform: uppercase;">
                    CORNELL BIRDNET &bull; {prim_sp['confidence']}% MATCH
                </span>
                <span style="font-family: 'Space Mono', monospace; font-size: 11px; color: #a2a7b7;">GLOBAL DATABASE: 6,522 SPECIES</span>
            </div>
            <div class="ceti-species-heading">{prim_sp['common_name']}</div>
            <div class="ceti-species-sci">{prim_sp['scientific_name']}</div>
            <p style="font-size: 0.95rem; color: #4e5a7c; line-height: 1.6; margin-bottom: 14px;">
                {visuals['description']}
            </p>
            <div style="border-top: 1px solid #e6e8ee; padding-top: 12px; margin-top: 12px; font-size: 0.9rem;">
                <p style="margin: 0 0 6px 0; color: #4e5a7c;">
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
        <div class="ceti-content-card" style="border-left: 3px solid #e63946;">
            <span style="background-color: #000000; color: #ffffff; font-family: 'Space Mono', monospace; font-size: 10px; padding: 3px 12px; letter-spacing: 0.08em; text-transform: uppercase;">
                NON-AVIAN ACOUSTIC PROFILE
            </span>
            <div class="ceti-species-heading" style="margin-top: 8px;">{prim_sp['common_name']}</div>
            <p style="font-size: 0.95rem; color: #4e5a7c; margin-top: 8px; line-height: 1.6;">
                The sound profile exhibits human oral cavity formant constraints and breath turbulence rather than authentic avian syrinx modulation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 3. Early Warning Threat Assessment Banner (Signature CETI Gradient)
    if spoof_info["suppress_alarm"]:
        st.markdown("""
        <div class="ceti-content-card" style="border-left: 3px solid #e63946;">
            <h3 style="font-family: 'GT Super Display', 'Playfair Display', serif; font-size: 1.6rem; color: #e63946; margin: 0 0 6px 0;">
                ⚠️ Disaster Early Warning Trigger Suppressed
            </h3>
            <p style="color: #4e5a7c; font-size: 0.95rem; margin: 0;">
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

    # 4. Quantitative Metrics Row
    col_gauge, col_stats = st.columns([1, 1.2])

    with col_gauge:
        st.markdown('<div class="ceti-content-card" style="padding: 20px;">', unsafe_allow_html=True)
        gauge_fig = create_gauge_chart(results["adi"], results["threat_color"])
        st.plotly_chart(gauge_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_stats:
        st.markdown('<div class="ceti-content-card" style="padding: 26px 30px;">', unsafe_allow_html=True)
        st.markdown("""<h4 style="font-family: 'GT Super Display', 'Playfair Display', serif; font-size: 1.5rem; color: #000000; margin-bottom: 18px;">Bioacoustic Signal Descriptors</h4>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Pitch Center (Centroid)", f"{features['spectral_centroid']['mean']:.0f} Hz", delta=f"{results['sub_scores']['pitch']}% stress")
            st.metric("Call Burst Rate", f"{features['call_rate']:.2f} calls/sec", delta=f"{results['sub_scores']['call_rate']}% density")
        with c2:
            st.metric("Noise Harshness (Flatness)", f"{features['spectral_flatness']['mean']:.4f}", delta=f"{results['sub_scores']['flatness']}% chaos")
            st.metric("Breath Turbulence Ratio", f"{spoof_info['breath_ratio']}%", delta="Human Breath" if spoof_info['breath_ratio'] > 20 else "Avian Syrinx")
        st.markdown('</div>', unsafe_allow_html=True)

    # 5. Scientific Visualizers & Tables
    tab_spec, tab_wave, tab_species, tab_anti_spoof, tab_table = st.tabs([
        "📊 Mel-Spectrogram Heatmap",
        "📈 Waveform & Call Onsets",
        "🐦 Top Candidate Species (BirdNET 6.5K)",
        "🛡️ Anti-Spoofing Biometrics",
        "📋 Diagnostic Factor Breakdown"
    ])

    with tab_spec:
        st.markdown('<div class="ceti-content-card">', unsafe_allow_html=True)
        st.plotly_chart(plot_mel_spectrogram(features), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_wave:
        st.markdown('<div class="ceti-content-card">', unsafe_allow_html=True)
        st.plotly_chart(plot_waveform_and_dynamics(y, sr, features), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_species:
        st.markdown('<div class="ceti-content-card">', unsafe_allow_html=True)
        st.markdown("""<h4 style="font-family: 'GT Super Display', 'Playfair Display', serif; font-size: 1.5rem; margin-bottom: 16px;">Top Species Candidates (from 6,522 Global Species)</h4>""", unsafe_allow_html=True)
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
        st.markdown('<div class="ceti-content-card">', unsafe_allow_html=True)
        st.markdown("""<h4 style="font-family: 'GT Super Display', 'Playfair Display', serif; font-size: 1.5rem; margin-bottom: 16px;">Biometric Verification & Anti-Spoofing Assessment</h4>""", unsafe_allow_html=True)
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
        st.markdown('<div class="ceti-content-card">', unsafe_allow_html=True)
        st.markdown("""<h4 style="font-family: 'GT Super Display', 'Playfair Display', serif; font-size: 1.5rem; margin-bottom: 16px;">Detailed Bioacoustic Diagnostic Parameters</h4>""", unsafe_allow_html=True)
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
    <div class="ceti-content-card" style="text-align: center; padding: 72px 32px;">
        <div style="font-family: 'GT Super Display', 'Playfair Display', serif; font-size: 2.4rem; font-weight: 700; color: #000000; margin-bottom: 12px;">
            Awaiting Bioacoustic Acoustic Feed
        </div>
        <p style="color: #4e5a7c; font-size: 1.05rem; max-width: 620px; margin: 0 auto 28px auto; line-height: 1.6;">
            Select a calibrated laboratory benchmark, upload an avian field recording, or initiate live microphonic acquisition to commence neural species identification and disaster threat forecasting.
        </p>
        <span class="ceti-pill-badge ceti-pill-verified">
            6,522 Species Active &bull; Biometric Anti-Spoofing Enabled
        </span>
    </div>
    """, unsafe_allow_html=True)
