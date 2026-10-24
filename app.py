"""INTERA — Designed in the Ponder Design System.

Ponder is a minimal, cinematic dark-mode product experience.
- Background: Pure Black (#000000)
- Surface 1: Dark Charcoal (#171717)
- Surface 2: Elevated Interactive Charcoal (#252525)
- Surface Hover: #2f2f2f
- Borders: Low-opacity hairline (#ffffff14 / #ffffff24)
- Typography: DM Sans (Lay Grotesk fallback) + JetBrains Mono (Fragment Mono fallback)
- Shapes: Binary — Perfect Pills (rounded: 9999px) for buttons/badges, 14px for cards
- Accent: Sky Blue (#4a9eed) strictly for focus and active indicators
- Status: #53b365 (Syrinx Authentic) & #fb2c36 (Adversarial Spoof / Critical)
- Performance: Cached inference pipelines for 60 FPS zero-lag interaction
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

# --- FIGMA / ICONS8 STANDARD OUTLINE SVG ICONS (STROKE: 1.8 - 2.0) ---
ICON_SHIELD = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>"""
ICON_ALERT = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>"""
ICON_CHECK = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#53b365" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>"""
ICON_CROSS = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#fb2c36" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>"""
ICON_MIC = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>"""
ICON_WAVE = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><path d="M2 10v4"/><path d="M6 6v12"/><path d="M10 3v18"/><path d="M14 8v8"/><path d="M18 5v14"/><path d="M22 10v4"/></svg>"""
ICON_NETWORK = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><rect width="6" height="6" x="9" y="2" rx="1"/><rect width="6" height="6" x="2" y="16" rx="1"/><rect width="6" height="6" x="16" y="16" rx="1"/><path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3"/><path d="M12 12V8"/></svg>"""
ICON_RADAR = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><circle cx="12" cy="12" r="10"/><path d="m16.2 7.8-4.2 4.2V2"/><path d="M12 16a4 4 0 1 0 0-8"/></svg>"""
ICON_AUDIO = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>"""

# Streamlit page configuration
st.set_page_config(
    page_title="INTERA // Bioacoustics",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PONDER DESIGN SYSTEM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --ds-background: #000000;
        --ds-surface: #171717;
        --ds-surface-2: #252525;
        --ds-surface-hover: #2f2f2f;
        --ds-on-surface: #fafafa;
        --ds-on-surface-secondary: rgba(250, 250, 250, 0.58);
        --ds-on-surface-tertiary: rgba(250, 250, 250, 0.38);
        --ds-border: rgba(255, 255, 255, 0.08);
        --ds-border-strong: rgba(255, 255, 255, 0.14);
        --ds-primary: #fafafa;
        --ds-on-primary: #000000;
        --ds-secondary: #252525;
        --ds-on-secondary: #fafafa;
        --ds-accent: #4a9eed;
        --ds-success: #53b365;
        --ds-error: #fb2c36;
        --ds-violet: #8b5cf6;
        --ds-radius-sm: 6px;
        --ds-radius-md: 10px;
        --ds-radius-lg: 14px;
        --ds-radius-full: 9999px;
    }

    /* Cinematic Pure Black Page Canvas */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: var(--ds-background) !important;
        background-image: none !important;
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        color: var(--ds-on-surface) !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Custom Minimal Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--ds-background);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--ds-surface-2);
        border-radius: var(--ds-radius-full);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--ds-surface-hover);
    }

    /* Text Selection */
    ::selection {
        background-color: rgba(74, 158, 237, 0.28);
        color: #ffffff;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0b0b0b !important;
        border-right: 1px solid var(--ds-border) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    [data-testid="stSidebar"] * {
        color: var(--ds-on-surface) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5 {
        font-family: 'DM Sans', sans-serif !important;
        color: var(--ds-on-surface) !important;
        letter-spacing: -0.01em !important;
    }

    /* Global Typography Rules — Ponder Restrained Scale */
    h1, h2, h3, h4, h5 {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        color: var(--ds-on-surface) !important;
        letter-spacing: -0.012em !important;
    }

    p, span, div, label {
        font-family: 'DM Sans', sans-serif;
        color: var(--ds-on-surface);
    }

    code, pre, .stCode, [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Ponder Minimal Header — Centered */
    .ponder-header {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px 0 24px 0;
        border-bottom: 1px solid var(--ds-border);
        margin-bottom: 36px;
        gap: 14px;
    }

    .ponder-brand {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        font-size: 15px;
        font-weight: 500;
        letter-spacing: -0.01em;
        color: var(--ds-on-surface);
    }

    .ponder-brand-dot {
        width: 8px;
        height: 8px;
        border-radius: var(--ds-radius-full);
        background-color: var(--ds-accent);
        display: inline-block;
        box-shadow: 0 0 0 2px rgba(74, 158, 237, 0.25);
    }

    .ponder-nav-pills {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }

    .ponder-nav-chip {
        font-size: 12px;
        font-weight: 500;
        color: var(--ds-on-surface-secondary);
        background: var(--ds-surface);
        border: 1px solid var(--ds-border);
        padding: 6px 14px;
        border-radius: var(--ds-radius-full);
        transition: all 0.15s ease;
    }

    .ponder-nav-chip:hover {
        color: var(--ds-on-surface);
        background: var(--ds-surface-2);
        border-color: var(--ds-border-strong);
    }

    /* Hero Section — Centered, Cinematic, High-Contrast */
    .ponder-hero-display {
        font-size: 34px;
        font-weight: 500;
        line-height: 42px;
        letter-spacing: -0.015em;
        color: var(--ds-on-surface);
        margin-bottom: 12px;
        text-align: center;
    }

    .ponder-hero-sub {
        font-size: 16px;
        font-weight: 400;
        line-height: 26px;
        color: var(--ds-on-surface-secondary);
        max-width: 820px;
        margin: 0 auto 32px auto;
        letter-spacing: -0.005em;
        text-align: center;
    }

    /* Tonal Surface Cards (z=1) */
    .ponder-card {
        background-color: var(--ds-surface);
        border: 1px solid var(--ds-border);
        border-radius: var(--ds-radius-lg);
        padding: 24px 28px;
        margin-bottom: 20px;
        transition: border-color 0.15s ease;
    }

    .ponder-card:hover {
        border-color: var(--ds-border-strong);
    }

    .ponder-card-elevated {
        background-color: var(--ds-surface-2);
        border: 1px solid var(--ds-border-strong);
        border-radius: var(--ds-radius-lg);
        padding: 24px 28px;
        margin-bottom: 20px;
    }

    /* Ponder Status Pill Badges */
    .ponder-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: var(--ds-radius-full);
        font-size: 12px;
        font-weight: 500;
        letter-spacing: -0.005em;
    }

    .ponder-pill-verified {
        background-color: rgba(83, 179, 101, 0.12);
        color: var(--ds-success);
        border: 1px solid rgba(83, 179, 101, 0.28);
    }

    .ponder-pill-warning {
        background-color: rgba(251, 44, 54, 0.12);
        color: var(--ds-error);
        border: 1px solid rgba(251, 44, 54, 0.28);
    }

    .ponder-pill-accent {
        background-color: rgba(74, 158, 237, 0.12);
        color: var(--ds-accent);
        border: 1px solid rgba(74, 158, 237, 0.28);
    }

    .ponder-pill-neutral {
        background-color: var(--ds-surface-2);
        color: var(--ds-on-surface);
        border: 1px solid var(--ds-border);
    }

    /* Streamlit Radio Buttons styled as Ponder Pills */
    div[data-testid="stRadio"] > label {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: var(--ds-on-surface-secondary) !important;
        letter-spacing: -0.005em !important;
        margin-bottom: 12px !important;
        text-align: center !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 8px !important;
        justify-content: center !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background-color: var(--ds-surface) !important;
        border: 1px solid var(--ds-border) !important;
        border-radius: var(--ds-radius-full) !important;
        padding: 8px 18px !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        color: var(--ds-on-surface) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background-color: var(--ds-surface-2) !important;
        border-color: var(--ds-border-strong) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
        background-color: var(--ds-surface-2) !important;
        border-color: var(--ds-accent) !important;
        color: #ffffff !important;
    }

    /* Selectbox */
    div[data-testid="stSelectbox"] label {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: var(--ds-on-surface-secondary) !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        border-radius: var(--ds-radius-sm) !important;
        border: 1px solid var(--ds-border-strong) !important;
        background-color: var(--ds-surface) !important;
        color: var(--ds-on-surface) !important;
        transition: border-color 0.15s ease;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within {
        border-color: var(--ds-accent) !important;
        box-shadow: 0 0 0 1px var(--ds-accent) !important;
    }

    /* Buttons — Ponder Signature Pill CTAs */
    .stButton > button {
        background-color: var(--ds-primary) !important;
        color: var(--ds-on-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        letter-spacing: -0.005em !important;
        padding: 9px 24px !important;
        border-radius: var(--ds-radius-full) !important;
        border: 1px solid var(--ds-primary) !important;
        transition: opacity 0.15s ease !important;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
    }

    /* Tabs — Minimal Hairline Underline */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 1px solid var(--ds-border);
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: var(--ds-on-surface-secondary) !important;
        border-radius: 0px !important;
        padding: 12px 4px !important;
        background: transparent !important;
        transition: color 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--ds-on-surface) !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--ds-on-surface) !important;
        border-bottom: 2px solid var(--ds-accent) !important;
    }

    /* Metrics Readout */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 26px !important;
        font-weight: 500 !important;
        color: var(--ds-on-surface) !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: var(--ds-on-surface-secondary) !important;
        letter-spacing: -0.005em !important;
    }

    /* Species Card Typography */
    .ponder-species-heading {
        font-size: 26px;
        font-weight: 500;
        color: var(--ds-on-surface);
        line-height: 32px;
        letter-spacing: -0.01em;
        margin: 4px 0 2px 0;
    }
    .ponder-species-sci {
        font-family: 'DM Sans', sans-serif;
        font-size: 14px;
        font-style: italic;
        color: var(--ds-on-surface-secondary);
        margin-bottom: 12px;
    }

    /* Diagnostic Table */
    table {
        border-collapse: collapse !important;
        width: 100% !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        background-color: var(--ds-surface) !important;
        border-radius: var(--ds-radius-md) !important;
        overflow: hidden !important;
    }
    thead tr th {
        font-size: 12px !important;
        font-weight: 500 !important;
        color: var(--ds-on-surface-tertiary) !important;
        border-bottom: 1px solid var(--ds-border) !important;
        padding: 12px 16px !important;
        text-align: left !important;
        background-color: var(--ds-surface-2) !important;
    }
    tbody tr td {
        border-bottom: 1px solid var(--ds-border) !important;
        padding: 12px 16px !important;
        color: var(--ds-on-surface) !important;
    }
    tbody tr:last-child td {
        border-bottom: none !important;
    }

    /* Audio Player dark frame */
    audio {
        filter: invert(1) hue-rotate(180deg) brightness(0.95);
        border-radius: var(--ds-radius-full);
        width: 100%;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --- 60 FPS PERFORMANCE CACHING PIPELINE ---
@st.cache_data(show_spinner=False)
def cached_process_audio(audio_bytes: bytes):
    """Zero-lag cached neural bioacoustics & biometric inference pipeline."""
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


# --- PLOTLY VISUALIZATION ENGINE (PONDER MONOCHROME + SKY BLUE ACCENT) ---
def create_gauge_chart(adi_score: float, threat_color: str) -> go.Figure:
    """Minimalist, dark-mode Plotly gauge with JetBrains Mono readout."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=adi_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "ACOUSTIC DISTRESS INDEX", 'font': {'size': 11, 'color': "rgba(250, 250, 250, 0.45)", 'family': "DM Sans, sans-serif"}},
        number={'suffix': "%", 'font': {'size': 44, 'color': "#fafafa", 'family': "JetBrains Mono, monospace"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(255, 255, 255, 0.12)"},
            'bar': {'color': "#4a9eed", 'thickness': 0.22},
            'bgcolor': "#171717",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.08)",
            'steps': [
                {'range': [0, 38], 'color': 'rgba(255, 255, 255, 0.04)'},
                {'range': [38, 70], 'color': 'rgba(74, 158, 237, 0.10)'},
                {'range': [70, 100], 'color': 'rgba(251, 44, 54, 0.14)'}
            ],
            'threshold': {
                'line': {'color': "#fb2c36", 'width': 2},
                'thickness': 0.8,
                'value': 70.0
            }
        }
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=15, r=15, t=35, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="DM Sans, sans-serif")
    )
    return fig


def plot_mel_spectrogram(features: dict) -> go.Figure:
    """Mel-Spectrogram heatmap matching Ponder obsidian & sky-blue tonal palette."""
    mel_db = features["mel_db"]
    times = features["times"]
    freq_bins = np.linspace(200, 10000, mel_db.shape[0])

    colorscale_ponder = [
        [0.0, "#0a0a0a"],
        [0.25, "#171717"],
        [0.55, "#1d3557"],
        [0.80, "#4a9eed"],
        [1.0, "#fafafa"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=mel_db,
        x=times,
        y=freq_bins,
        colorscale=colorscale_ponder,
        colorbar=dict(
            title=dict(text="dB", font=dict(family="JetBrains Mono, monospace", size=11, color="rgba(250,250,250,0.6)")),
            len=0.8,
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color="rgba(250,250,250,0.5)")
        )
    ))
    fig.update_layout(
        title=dict(text="Mel-Frequency Bioacoustic Spectrogram", font=dict(family="DM Sans, sans-serif", size=16, color="#fafafa")),
        xaxis=dict(title=dict(text="Time (seconds)", font=dict(family="DM Sans, sans-serif", size=12, color="rgba(250,250,250,0.6)")), showgrid=False, tickfont=dict(family="JetBrains Mono, monospace", size=10, color="rgba(250,250,250,0.5)")),
        yaxis=dict(title=dict(text="Frequency (Hz)", font=dict(family="DM Sans, sans-serif", size=12, color="rgba(250,250,250,0.6)")), showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(family="JetBrains Mono, monospace", size=10, color="rgba(250,250,250,0.5)")),
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_waveform_and_dynamics(y: np.ndarray, sr: int, features: dict) -> go.Figure:
    """Waveform and pitch trajectory visualizer with Ponder precision styling."""
    time_wave = np.linspace(0, len(y) / sr, len(y))
    times_feats = features["times"]
    centroid = features["spectral_centroid"]["trajectory"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=("Acoustic Signal Waveform & Call Onset Bursts", "Spectral Centroid Trajectory (Hz)")
    )

    step = max(1, len(y) // 3000)
    fig.add_trace(
        go.Scatter(x=time_wave[::step], y=y[::step], mode='lines', name='Waveform', line=dict(color="#4a9eed", width=1.2)),
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
                marker=dict(symbol="triangle-down", size=7, color="#fafafa")
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(x=times_feats, y=centroid, mode='lines', name='Centroid (Hz)', line=dict(color="#fafafa", width=1.5)),
        row=2, col=1
    )

    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(font=dict(family="DM Sans, sans-serif", size=11, color="rgba(250,250,250,0.7)")),
        font=dict(family="DM Sans, sans-serif")
    )
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(family="JetBrains Mono", size=10, color="rgba(250,250,250,0.5)"))
    fig.update_yaxes(title_text="Amplitude", row=1, col=1, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(family="JetBrains Mono", size=10, color="rgba(250,250,250,0.5)"))
    fig.update_yaxes(title_text="Hz", row=2, col=1, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(family="JetBrains Mono", size=10, color="rgba(250,250,250,0.5)"))

    return fig


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div style="font-size: 11px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: #4a9eed; margin-bottom: 6px;">
        RESEARCH INITIATIVE
    </div>
    <div style="font-size: 22px; font-weight: 500; color: #fafafa; line-height: 28px; letter-spacing: -0.01em; margin-bottom: 8px;">
        INTERA
    </div>
    <div style="font-size: 14px; color: rgba(250, 250, 250, 0.58); margin-bottom: 24px; line-height: 20px;">
        Avian Bioacoustic Translation & Planetary Early Warning Intelligence.
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown(f"##### {ICON_WAVE} Bioacoustic Principles", unsafe_allow_html=True)
    st.caption("""
    * **Non-Human Vocal Translation:** Decoding flock panic patterns prior to seismic and atmospheric events.
    * **Dual Syrinx Biphonation:** Verifying authentic multi-tonal avian biological markers.
    * **Planetary Sentinel Waves:** Detecting seismic P-waves & storm infrasound (< 20 Hz).
    """)

    st.divider()
    st.markdown(f"##### {ICON_NETWORK} Neural Architecture", unsafe_allow_html=True)
    st.markdown("""
    * **Classifier:** Cornell BirdNET v2.4 (6,522 Species)
    * **Biometrics:** Syrinx vs Oral Cavity Anti-Spoofing
    * **Performance:** Cached M1 Local Neural Core (0ms rerun)
    """)


# --- TOP NAVIGATION BAR ---
st.markdown(f"""
<div class="ponder-header">
    <div class="ponder-brand">
        <span class="ponder-brand-dot"></span>
        <span>INTERA // BIOACOUSTICS</span>
    </div>
    <div class="ponder-nav-pills">
        <span class="ponder-nav-chip">{ICON_RADAR} 6,522 SPECIES</span>
        <span class="ponder-nav-chip">{ICON_SHIELD} BIOMETRIC SHIELD</span>
    </div>
</div>
""", unsafe_allow_html=True)


# --- HERO SECTION ---
st.markdown("""
<div class="ponder-hero-display">
    Understand What Birds are Saying
</div>
<div class="ponder-hero-sub">
    Applying deep learning and bioacoustic translation to decode avian flock vocalizations &mdash; forecasting earthquakes, cyclones, and atmospheric shockwaves before disaster strikes.
</div>
""", unsafe_allow_html=True)


# --- INPUT SELECTION ---
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
                "calm_robin_song.wav": "[CALM] American Robin Whistle — Baseline Melodic",
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


# --- INFERENCE & PRESENTATION ---
if audio_data is not None:
    # 0ms Lag Cached Pipeline
    with st.spinner("Decoding bioacoustic signals with Cornell BirdNET & verifying biometrics..."):
        y, sr, duration, features, results, species_info, spoof_info = cached_process_audio(audio_data)

    prim_sp = species_info["primary_species"]

    # Header for acquired feed
    st.markdown(f"""
    <div style="text-align: center; margin-top: 24px; margin-bottom: 8px;">
        <div style="font-size: 11px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: #4a9eed;">
            ACOUSTIC FEED ACQUIRED
        </div>
        <div style="font-size: 22px; font-weight: 500; color: #fafafa; letter-spacing: -0.01em;">
            {audio_label}
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: rgba(250, 250, 250, 0.58); margin-top: 4px;">
            Duration: {duration:.2f}s • Sample Rate: {sr:,} Hz
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.audio(audio_data)

    # 1. Biometric Anti-Spoofing Verification Panel
    if spoof_info["is_authentic"]:
        st.markdown(f"""
        <div class="ponder-card" style="border-left: 2px solid #53b365;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="ponder-pill ponder-pill-verified">
                    {ICON_SHIELD} {spoof_info['status_label']}
                </span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: rgba(250, 250, 250, 0.58);">
                    BIRDNET CONFIDENCE: {spoof_info['birdnet_conf']}% • SYRINX CONFIRMED
                </span>
            </div>
            <p style="margin: 8px 0 0 0; font-size: 14px; color: rgba(250, 250, 250, 0.75); line-height: 20px;">
                {spoof_info['status_summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ponder-card" style="border-left: 2px solid #fb2c36;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="ponder-pill ponder-pill-warning">
                    {ICON_ALERT} {spoof_info['status_label']}
                </span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #fb2c36;">
                    ADVERSARIAL SPOOFING DETECTED
                </span>
            </div>
            <p style="margin: 8px 0 0 0; font-size: 14px; color: rgba(250, 250, 250, 0.75); line-height: 20px;">
                {spoof_info['status_summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 2. Species Intelligence Card
    if spoof_info["is_authentic"]:
        visuals = cached_fetch_visuals(prim_sp['common_name'], prim_sp['scientific_name'])
        
        st.markdown('<div class="ponder-card">', unsafe_allow_html=True)
        col_photo, col_details = st.columns([1, 2.2])
        
        with col_photo:
            st.image(
                visuals["image_url"],
                caption=f"{prim_sp['common_name']} ({visuals['source']})",
                use_container_width=True
            )
        
        with col_details:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span class="ponder-pill ponder-pill-accent">
                    CORNELL BIRDNET • {prim_sp['confidence']}% MATCH
                </span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: rgba(250, 250, 250, 0.38);">GLOBAL TAXONOMY: 6,522 SPECIES</span>
            </div>
            <div class="ponder-species-heading">{prim_sp['common_name']}</div>
            <div class="ponder-species-sci">{prim_sp['scientific_name']}</div>
            <p style="font-size: 14px; color: rgba(250, 250, 250, 0.75); line-height: 22px; margin-bottom: 16px;">
                {visuals['description']}
            </p>
            <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 12px; margin-top: 12px; font-size: 13px;">
                <p style="margin: 0 0 6px 0; color: rgba(250, 250, 250, 0.75);">
                    <strong style="color: #fafafa;">Acoustic State:</strong> {prim_sp['pitch_status']} &bull; Pitch Centroid: <strong style="color: #4a9eed; font-family: 'JetBrains Mono';">{prim_sp['detected_pitch']:.0f} Hz</strong>
                </p>
                <p style="margin: 4px 0 0 0; color: #4a9eed;">
                    <strong style="color: #fafafa;">Disaster Sensitivity:</strong> {prim_sp['disaster_sensitivity']}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ponder-card" style="border-left: 2px solid #fb2c36;">
            <span class="ponder-pill ponder-pill-warning">
                NON-AVIAN ACOUSTIC PROFILE
            </span>
            <div class="ponder-species-heading" style="margin-top: 8px;">{prim_sp['common_name']}</div>
            <p style="font-size: 14px; color: rgba(250, 250, 250, 0.75); margin-top: 8px; line-height: 22px;">
                The sound profile exhibits human oral cavity formant constraints and breath turbulence rather than authentic avian syrinx modulation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 3. Early Warning Threat Assessment
    if spoof_info["suppress_alarm"]:
        st.markdown(f"""
        <div class="ponder-card-elevated" style="border-left: 2px solid #fb2c36;">
            <h3 style="font-size: 18px; color: #fb2c36; margin: 0 0 6px 0;">
                {ICON_ALERT} Disaster Early Warning Trigger Suppressed
            </h3>
            <p style="color: rgba(250, 250, 250, 0.75); font-size: 14px; margin: 0; line-height: 20px;">
                Because non-avian or human acoustic mimicry was identified, automated early-warning triggers have been locked to prevent civil false alarms.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        banner_border_color = "#53b365" if results["threat_level"] == "LOW" else ("#e0a96d" if results["threat_level"] == "MEDIUM" else "#fb2c36")
        st.markdown(f"""
        <div class="ponder-card-elevated" style="border-left: 2px solid {banner_border_color};">
            <div style="font-size: 11px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: {banner_border_color}; margin-bottom: 4px;">
                SENTINEL STATE: {results['threat_level']} RISK
            </div>
            <h3 style="font-size: 20px; font-weight: 500; color: #fafafa; margin: 0 0 6px 0;">
                {results['alert_title']}
            </h3>
            <p style="color: rgba(250, 250, 250, 0.75); font-size: 14px; margin: 0; line-height: 22px;">
                {results['alert_summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 4. Quantitative Metrics Row
    col_gauge, col_stats = st.columns([1, 1.2])

    with col_gauge:
        st.markdown('<div class="ponder-card" style="padding: 16px 20px;">', unsafe_allow_html=True)
        gauge_fig = create_gauge_chart(results["adi"], results["threat_color"])
        st.plotly_chart(gauge_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_stats:
        st.markdown('<div class="ponder-card" style="padding: 22px 26px;">', unsafe_allow_html=True)
        st.markdown("""<div style="font-size: 14px; font-weight: 500; color: #fafafa; margin-bottom: 16px;">Bioacoustic Signal Descriptors</div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Pitch Center (Centroid)", f"{features['spectral_centroid']['mean']:.0f} Hz", delta=f"{results['sub_scores']['pitch']}% stress")
            st.metric("Call Burst Rate", f"{features['call_rate']:.2f} calls/s", delta=f"{results['sub_scores']['call_rate']}% density")
        with c2:
            st.metric("Noise Harshness (Flatness)", f"{features['spectral_flatness']['mean']:.4f}", delta=f"{results['sub_scores']['flatness']}% chaos")
            st.metric("Breath Turbulence Ratio", f"{spoof_info['breath_ratio']}%", delta="Human Breath" if spoof_info['breath_ratio'] > 20 else "Avian Syrinx")
        st.markdown('</div>', unsafe_allow_html=True)

    # 5. Scientific Visualizers & Tables
    tab_spec, tab_wave, tab_species, tab_anti_spoof, tab_table = st.tabs([
        "SPECTROGRAM",
        "WAVEFORM & ONSETS",
        "SPECIES TAXONOMY (6.5K)",
        "BIOMETRIC VERIFICATION",
        "DIAGNOSTIC DATA"
    ])

    with tab_spec:
        st.markdown('<div class="ponder-card">', unsafe_allow_html=True)
        st.plotly_chart(plot_mel_spectrogram(features), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_wave:
        st.markdown('<div class="ponder-card">', unsafe_allow_html=True)
        st.plotly_chart(plot_waveform_and_dynamics(y, sr, features), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_species:
        st.markdown('<div class="ponder-card">', unsafe_allow_html=True)
        st.markdown("""<div style="font-size: 16px; font-weight: 500; color: #fafafa; margin-bottom: 16px;">Top Neural Species Matches (from 6,522 Global Species)</div>""", unsafe_allow_html=True)
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
        st.markdown('<div class="ponder-card">', unsafe_allow_html=True)
        st.markdown("""<div style="font-size: 16px; font-weight: 500; color: #fafafa; margin-bottom: 16px;">Biometric Syrinx vs Oral Cavity Verification</div>""", unsafe_allow_html=True)
        
        status_html = f"{ICON_CHECK} <strong style='color:#53b365;'>Authentic Avian Syrinx Verified</strong>" if spoof_info['is_authentic'] else f"{ICON_CROSS} <strong style='color:#fb2c36;'>Rejected (Human Oral Mimicry / Spoofing)</strong>"
        st.markdown(f"<div style='margin-bottom: 12px;'>{status_html}</div>", unsafe_allow_html=True)
        st.write(f"**Spoofing Likelihood Score:** `{spoof_info['spoof_score']} / 100`")
        st.write(f"**Sub-600Hz Breath Turbulence Ratio:** `{spoof_info['breath_ratio']}%` (Threshold: > 28% indicates human breath exhalation)")
        
        if spoof_info["spoof_factors"]:
            st.markdown("##### Detected Anti-Spoofing Flags:")
            for factor in spoof_info["spoof_factors"]:
                st.warning(f"• {factor}")
        else:
            st.success("• Dual syrinx harmonic purity confirmed. Zero human oral cavity formant constraints detected.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_table:
        st.markdown('<div class="ponder-card">', unsafe_allow_html=True)
        st.markdown("""<div style="font-size: 16px; font-weight: 500; color: #fafafa; margin-bottom: 16px;">Acoustic Feature Diagnostics</div>""", unsafe_allow_html=True)
        diag_rows = []
        for d in results["diagnostics"]:
            diag_rows.append({
                "Factor": d["name"],
                "Detected Value": d["value"],
                "Normal Baseline": d["normal_range"],
                "Stress Score": f"{d['score']}%",
                "Status": d["status"]
            })
        st.table(diag_rows)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div class="ponder-card" style="text-align: center; padding: 64px 32px;">
        <div style="font-size: 24px; font-weight: 500; color: #fafafa; margin-bottom: 10px; letter-spacing: -0.01em;">
            Awaiting Bioacoustic Acoustic Feed
        </div>
        <p style="color: rgba(250, 250, 250, 0.58); font-size: 15px; max-width: 620px; margin: 0 auto 24px auto; line-height: 24px;">
            Select a calibrated laboratory benchmark, upload an avian field recording, or initiate live microphonic acquisition to commence neural species identification and disaster threat forecasting.
        </p>
        <span class="ponder-pill ponder-pill-neutral">
            {ICON_SHIELD} 6,522 Species Active &bull; Biometric Anti-Spoofing Guard Ready
        </span>
    </div>
    """, unsafe_allow_html=True)
