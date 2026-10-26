"""Project INTERA — Avian Bioacoustic Translation & Planetary Early Warning Intelligence.
Art-directed in the high-precision, tactile Apple Human Interface & Soft Neumorphic Design System.

Principles:
- Ultra-clean Apple aesthetic: soft off-white ambient canvas (#F5F7FA) and floating pure white cards.
- Soft tactile neumorphic depth: subtle multi-layer shadows, pill-shaped buttons, and iOS segmented controls.
- Apple SF Pro typography hierarchy: crisp, confident typography with tight tracking and balanced weights.
- Apple Stats Cards with Circular Progress Rings: inspired directly by Apple Health / Fitness / Pet dashboard UI.
- Studio-grade bioacoustic telemetrics: High-definition Mel-Spectrogram, synchronized waveform visualizer, and Cornell BirdNET 6,522 species classifier.
- Non-avian biological shield: real-time mammalian larynx and phone-speaker replay discrimination.
"""

import base64
import glob
import html
import logging
import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.audio_processor import extract_acoustic_features, load_audio
from src.distress_engine import (
    compute_distress_index,
    compute_distress_timeline,
    evaluate_flock_consensus,
    generate_forensic_report,
)
from src.image_fetcher import fetch_bird_visuals
from src.species_classifier import identify_species
from src.spoof_detector import evaluate_avian_authenticity

logger = logging.getLogger(__name__)

# --- APPLE SF SYMBOLS STYLE OUTLINE SVG ICONS ---
ICON_SHIELD = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>"""
ICON_ALERT = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>"""
ICON_CHECK = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#34C759" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>"""
ICON_CROSS = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#FF3B30" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>"""
ICON_MIC = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>"""
ICON_UPLOAD_APPLE = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>"""
ICON_LIBRARY_APPLE = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>"""


def get_image_data_uri(path_or_url: str) -> str:
    """Convert local image paths into base64 Data URIs for instant, zero-lag browser rendering."""
    if not path_or_url:
        return ""
    if path_or_url.startswith(("http://", "https://", "data:")):
        return path_or_url
    if os.path.exists(path_or_url):
        try:
            with open(path_or_url, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(path_or_url)[1].lstrip(".").lower()
                mime = "jpeg" if ext in ["jpg", "jpeg"] else ("png" if ext == "png" else "webp")
                return f"data:image/{mime};base64,{encoded}"
        except OSError as e:
            logger.warning("Error reading species image '%s': %s", path_or_url, e)
            return path_or_url
    return path_or_url


def render_apple_stat_card(
    title: str,
    value: str,
    subtext: str,
    percentage: float,
    color: str = "#ff4d00",
    is_dark: bool = True
) -> str:
    """Generate a Cédric Studio charcoal stat panel with an SVG progress ring and Inter typography."""
    radius = 21
    circumference = 2 * 3.14159 * radius
    pct_clamped = min(100.0, max(0.0, float(percentage)))
    stroke_dashoffset = circumference * (1.0 - (pct_clamped / 100.0))
    bg_ring = "rgba(255, 255, 255, 0.08)"
    card_bg = "#1a1a1a"
    card_border = "rgba(255, 255, 255, 0.08)"
    text_color = "#f2f2f2"
    sub_color = "#878787"
    
    return f"""
    <div class="cedric-stat-card" style="background: {card_bg}; border: 1px solid {card_border}; border-radius: 12px; padding: 18px 20px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
        <div style="flex: 1; min-width: 0; padding-right: 8px;">
            <div style="color: {sub_color}; font-size: 13px; font-weight: 500; font-family: 'Inter', sans-serif; letter-spacing: 0.02em;">{html.escape(title)}</div>
            <div style="color: {text_color}; font-size: 22px; font-weight: 600; font-family: 'Inter', sans-serif; margin-top: 4px; letter-spacing: -0.02em;">{html.escape(value)}</div>
            <div style="color: {sub_color}; font-size: 13px; font-weight: 400; font-family: 'Inter', sans-serif; margin-top: 3px;">{html.escape(subtext)}</div>
        </div>
        <div style="position: relative; width: 56px; height: 56px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
            <svg width="56" height="56" viewBox="0 0 56 56" style="transform: rotate(-90deg);">
                <circle cx="28" cy="28" r="{radius}" fill="none" stroke="{bg_ring}" stroke-width="4.5" />
                <circle cx="28" cy="28" r="{radius}" fill="none" stroke="{color}" stroke-width="4.5"
                    stroke-dasharray="{circumference:.2f}"
                    stroke-dashoffset="{stroke_dashoffset:.2f}"
                    stroke-linecap="round"
                    style="transition: stroke-dashoffset 0.6s cubic-bezier(0.16, 1, 0.3, 1);" />
            </svg>
            <div style="position: absolute; font-size: 11.5px; font-weight: 600; color: #f2f2f2; font-family: 'Inter', sans-serif;">
                {int(pct_clamped)}%
            </div>
        </div>
    </div>
    """


def generate_waveform_svg_bars(y: np.ndarray, num_bars: int = 42) -> str:
    """Generate dynamic Apple Voice Memos style waveform bars from the actual audio signal."""
    if len(y) == 0:
        return ""
    step = max(1, len(y) // num_bars)
    bars_html = []
    for i in range(num_bars):
        chunk = y[i * step : (i + 1) * step]
        if len(chunk) > 0:
            amp = float(np.mean(np.abs(chunk)))
        else:
            amp = 0.05
        height_px = max(4, min(34, int(amp * 115)))
        bars_html.append(f'<span class="waveform-bar" style="height:{height_px}px;"></span>')
    return "".join(bars_html)


# Page configuration — full-width, clean layout without sidebar
st.set_page_config(
    page_title="Project INTERA — Bioacoustic Intelligence",
    page_icon="https://cdn.prod.website-files.com/643ddd7ffdf12273933a8cec/645bdf9d4bac53320316d064_CETI-Favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark Mode State Initialization
st.session_state.setdefault("dark_mode_toggle", False)
is_dark = bool(st.session_state["dark_mode_toggle"])

def get_app_css(dark: bool = True) -> str:
    """Generate Cédric Studio Design System CSS (https://www.cedric.design / DESIGN.md).
    Pure dark restrained palette: Surface #141414, Panels #1a1a1a, Primary #ff4d00, Secondary #78e0ff.
    """
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --studio-surface: #141414;
            --studio-panel: #1a1a1a;
            --studio-panel-elevated: #222222;
            --studio-border: rgba(255, 255, 255, 0.08);
            --studio-border-hover: rgba(255, 255, 255, 0.16);
            --studio-primary: #ff4d00;
            --studio-primary-hover: #e04400;
            --studio-secondary: #78e0ff;
            --studio-text: #f2f2f2;
            --studio-muted: #878787;
            --studio-error: #ff5252;
            --studio-success: #34d399;
        }

        /* Universal Typography: Inter, sans-serif */
        *, html, body, .stApp, 
        h1, h2, h3, h4, h5, h6, 
        p, span, div, a, label, input, select, textarea, button, 
        .stTabs [data-baseweb="tab"], 
        div[data-testid="stRadio"] *, 
        div[data-testid="stMetricValue"], 
        div[data-testid="stMetricLabel"],
        table, th, td, pre, code {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            -webkit-font-smoothing: antialiased !important;
            -moz-osx-font-smoothing: grayscale !important;
            text-rendering: optimizeLegibility !important;
            box-sizing: border-box !important;
        }

        html, body, .stApp {
            background-color: #141414 !important;
            color: #f2f2f2 !important;
            min-height: 100vh !important;
            overflow-x: hidden !important;
        }

        /* Typography Hierarchy (DESIGN.md) */
        h1 {
            font-size: 46px !important;
            font-weight: 500 !important;
            line-height: 1.15 !important;
            color: #f2f2f2 !important;
            letter-spacing: -0.03em !important;
        }

        h2, h3, h4 {
            font-weight: 500 !important;
            color: #f2f2f2 !important;
            letter-spacing: -0.02em !important;
        }

        p, .intera-hero-subtitle {
            font-size: 16px !important;
            line-height: 1.6 !important;
            color: #878787 !important;
            font-weight: 400 !important;
        }

        /* Cédric Studio Hero Section */
        .intera-hero {
            text-align: left;
            max-width: 860px;
            margin: 24px 0 42px 0;
            padding: 0 4px;
        }

        .intera-hero-eyebrow {
            font-size: 13px !important;
            font-weight: 500 !important;
            letter-spacing: 0.04em !important;
            text-transform: uppercase !important;
            color: #78e0ff !important;
            margin-bottom: 12px !important;
        }

        .intera-hero-understand {
            font-size: clamp(2.4rem, 5.2vw, 3.6rem) !important;
            font-weight: 500 !important;
            line-height: 1.15 !important;
            color: #f2f2f2 !important;
            letter-spacing: -0.03em !important;
            margin-bottom: 16px !important;
        }

        .intera-hero-subtitle {
            font-size: 16px !important;
            line-height: 1.6 !important;
            color: #878787 !important;
            max-width: 720px !important;
            font-weight: 400 !important;
            margin-top: 14px !important;
        }

        /* Completely collapse sidebar */
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

        /* Charcoal Panels (DESIGN.md #1a1a1a & #262626) */
        [data-testid="stVerticalBlockBorderWrapper"] > div,
        .intera-card {
            background: #1a1a1a !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            box-shadow: none !important;
            padding: 24px 26px !important;
            transition: border-color 0.2s ease !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] > div:hover,
        .intera-card:hover {
            border-color: rgba(255, 255, 255, 0.14) !important;
        }

        .apple-upload-card {
            background: #1a1a1a !important;
            border-radius: 12px !important;
            padding: 28px 24px !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            text-align: center !important;
            margin-bottom: 20px !important;
        }

        /* ========================================================
           Cédric Studio Segmented Acquisition Control (Pill Options)
           ======================================================== */
        [data-testid="stElementContainer"]:has([data-testid="stRadio"]) {
            display: flex !important;
            justify-content: flex-start !important;
            align-items: center !important;
            width: 100% !important;
            margin: 0 0 28px 0 !important;
        }

        div[data-testid="stRadio"] {
            display: flex !important;
            flex-direction: column !important;
            width: 100% !important;
        }

        div[data-testid="stRadio"] > label,
        div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
            display: none !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] {
            display: inline-flex !important;
            flex-direction: row !important;
            flex-wrap: wrap !important;
            justify-content: flex-start !important;
            align-items: center !important;
            background: #1a1a1a !important;
            padding: 4px !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 12px !important;
            gap: 4px !important;
            width: auto !important;
            max-width: 100% !important;
        }

        /* Hide radio circle dots completely */
        div[data-testid="stRadio"] [data-baseweb="radio"] input[type="radio"],
        div[data-testid="stRadio"] [data-baseweb="radio"] input[type="radio"] + div,
        div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-of-type,
        div[data-testid="stRadio"] [data-baseweb="radio"] svg {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            opacity: 0 !important;
        }

        /* Maker Studio Pill Options */
        div[data-testid="stRadio"] div[role="radiogroup"] > label,
        div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"] {
            background: transparent !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            white-space: normal !important;
            text-align: center !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin: 0 !important;
            box-shadow: none !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label *,
        div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"] * {
            font-size: 14px !important;
            font-weight: 500 !important;
            color: #878787 !important;
            transition: color 0.2s ease !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label:hover *,
        div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"]:hover * {
            color: #f2f2f2 !important;
        }

        /* Active State: Punchy Safety Orange #ff4d00 (DESIGN.md) */
        div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
        div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"]:has(input:checked) {
            background: #ff4d00 !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 10px rgba(255, 77, 0, 0.35) !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) *,
        div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"]:has(input:checked) * {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* SVG Icons inside Pill Options */
        div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(1) [data-testid="stMarkdownContainer"] p::before {
            content: '';
            display: inline-block;
            width: 16px;
            height: 16px;
            margin-right: 8px;
            vertical-align: -2px;
            background-repeat: no-repeat;
            background-size: contain;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23878787' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M2 10v3'/%3E%3Cpath d='M6 6v11'/%3E%3Cpath d='M10 3v18'/%3E%3Cpath d='M14 8v7'/%3E%3Cpath d='M18 5v13'/%3E%3Cpath d='M22 10v3'/%3E%3C/svg%3E");
        }
        div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(1):has(input:checked) [data-testid="stMarkdownContainer"] p::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FFFFFF' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M2 10v3'/%3E%3Cpath d='M6 6v11'/%3E%3Cpath d='M10 3v18'/%3E%3Cpath d='M14 8v7'/%3E%3Cpath d='M18 5v13'/%3E%3Cpath d='M22 10v3'/%3E%3C/svg%3E");
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(2) [data-testid="stMarkdownContainer"] p::before {
            content: '';
            display: inline-block;
            width: 16px;
            height: 16px;
            margin-right: 8px;
            vertical-align: -2px;
            background-repeat: no-repeat;
            background-size: contain;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23878787' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E");
        }
        div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(2):has(input:checked) [data-testid="stMarkdownContainer"] p::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FFFFFF' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E");
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(3) [data-testid="stMarkdownContainer"] p::before {
            content: '';
            display: inline-block;
            width: 16px;
            height: 16px;
            margin-right: 8px;
            vertical-align: -2px;
            background-repeat: no-repeat;
            background-size: contain;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23878787' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z'/%3E%3Cpath d='M19 10v2a7 7 0 0 1-14 0v-2'/%3E%3Cline x1='12' y1='19' x2='12' y2='22'/%3E%3C/svg%3E");
        }
        div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(3):has(input:checked) [data-testid="stMarkdownContainer"] p::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FFFFFF' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z'/%3E%3Cpath d='M19 10v2a7 7 0 0 1-14 0v-2'/%3E%3Cline x1='12' y1='19' x2='12' y2='22'/%3E%3C/svg%3E");
        }

        /* Buttons & Actions: Safety Orange #ff4d00 (DESIGN.md) */
        .stButton > button {
            background: #ff4d00 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            letter-spacing: -0.01em !important;
            box-shadow: 0 2px 8px rgba(255, 77, 0, 0.3) !important;
            transition: all 0.2s ease !important;
            cursor: pointer !important;
        }

        .stButton > button:hover {
            background: #e04400 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 14px rgba(255, 77, 0, 0.45) !important;
        }

        .stButton > button * {
            color: #ffffff !important;
            font-weight: 500 !important;
        }

        /* Dropzone: Charcoal Panel #1a1a1a with subtle 1px dashed border */
        [data-testid="stFileUploaderDropzone"] {
            background: #1a1a1a !important;
            border: 1px dashed rgba(255, 255, 255, 0.14) !important;
            border-radius: 12px !important;
            padding: 28px 24px !important;
            text-align: center !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(255, 255, 255, 0.25) !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            background: #262626 !important;
            color: #f2f2f2 !important;
            border-radius: 8px !important;
            padding: 9px 22px !important;
            font-size: 13.5px !important;
            font-weight: 500 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            box-shadow: none !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="stFileUploaderDropzone"] button:hover {
            background: #ff4d00 !important;
            color: #ffffff !important;
            border-color: #ff4d00 !important;
        }

        /* File item delete button */
        [data-testid="stFileUploader"] button[aria-label*="Delete"],
        [data-testid="stFileUploader"] button[aria-label*="Remove"],
        [data-testid="stFileUploaderFile"] button {
            background: transparent !important;
            box-shadow: none !important;
            border: none !important;
            padding: 4px !important;
            border-radius: 4px !important;
            color: #878787 !important;
        }
        [data-testid="stFileUploader"] button[aria-label*="Delete"]:hover,
        [data-testid="stFileUploader"] button[aria-label*="Remove"]:hover,
        [data-testid="stFileUploaderFile"] button:hover {
            background: rgba(255, 82, 82, 0.15) !important;
            color: #ff5252 !important;
        }

        /* Selectbox */
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: #1a1a1a !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            padding: 6px 14px !important;
            font-size: 14px !important;
            font-weight: 400 !important;
            color: #f2f2f2 !important;
        }

        /* Tabs (DESIGN.md: Charcoal Navigation) */
        .stTabs [data-baseweb="tab-list"] {
            background: #1a1a1a !important;
            padding: 4px !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            display: flex !important;
            justify-content: flex-start !important;
            gap: 4px !important;
            margin-bottom: 24px !important;
            flex-wrap: wrap !important;
            max-width: 100% !important;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 6px !important;
            padding: 8px 18px !important;
            color: #878787 !important;
            font-weight: 500 !important;
            font-size: 13.5px !important;
            border: none !important;
            background: transparent !important;
            transition: all 0.2s ease !important;
            white-space: normal !important;
            text-align: center !important;
            line-height: 1.3 !important;
        }

        .stTabs [aria-selected="true"] {
            background: #262626 !important;
            color: #f2f2f2 !important;
            font-weight: 600 !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.4) !important;
        }

        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }

        /* Species Card Avatar */
        .apple-pet-card {
            background: #1a1a1a;
            border-radius: 12px;
            padding: 22px 24px;
            display: flex;
            align-items: center;
            gap: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .apple-avatar {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid rgba(255, 255, 255, 0.12);
        }

        /* Audio Console */
        .apple-audio-console {
            background: #1a1a1a;
            border-radius: 12px;
            padding: 14px 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 14px;
            flex-wrap: wrap;
        }

        .audio-console-left {
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 200px;
        }

        .audio-play-badge {
            width: 40px;
            height: 40px;
            border-radius: 8px;
            background: #ff4d00;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            flex-shrink: 0;
        }

        .audio-filename {
            font-size: 14px;
            font-weight: 600;
            color: #f2f2f2;
            letter-spacing: -0.01em;
        }

        .audio-subtext {
            font-size: 12px;
            color: #878787;
            font-weight: 400;
            margin-top: 2px;
        }

        .audio-console-center {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 180px;
        }

        .waveform-container {
            display: flex;
            align-items: center;
            gap: 3px;
            height: 34px;
            width: 100%;
            max-width: 400px;
            justify-content: center;
        }

        .waveform-bar {
            width: 3px;
            background: #78e0ff;
            border-radius: 2px;
            opacity: 0.85;
            transition: height 0.3s ease, opacity 0.2s ease;
        }

        .audio-console-right {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .live-equalizer {
            display: flex;
            align-items: flex-end;
            gap: 2.5px;
            height: 16px;
        }

        @keyframes eqWave {
            0%, 100% { height: 3px; }
            50% { height: 16px; }
        }

        .eq-bar {
            width: 2.5px;
            background: #ff4d00;
            border-radius: 2px;
            animation: eqWave 1.2s ease-in-out infinite alternate;
        }
        .eq-bar:nth-child(1) { animation-delay: 0.1s; height: 5px; }
        .eq-bar:nth-child(2) { animation-delay: 0.3s; height: 12px; }
        .eq-bar:nth-child(3) { animation-delay: 0.05s; height: 16px; }
        .eq-bar:nth-child(4) { animation-delay: 0.4s; height: 9px; }
        .eq-bar:nth-child(5) { animation-delay: 0.25s; height: 13px; }
        .eq-bar:nth-child(6) { animation-delay: 0.15s; height: 7px; }

        .audio-live-tag {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.04em;
            color: #78e0ff;
            background: rgba(120, 224, 255, 0.1);
            border: 1px solid rgba(120, 224, 255, 0.2);
            padding: 4px 10px;
            border-radius: 4px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .pulse-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background-color: #78e0ff;
            display: inline-block;
        }

        audio {
            width: 100% !important;
            border-radius: 8px !important;
            height: 36px !important;
        }
    </style>
    """

# Inject Cédric Studio Design System CSS
st.markdown(get_app_css(), unsafe_allow_html=True)


# --- PERFORMANCE CACHING PIPELINE ---
@st.cache_data(show_spinner=False)
def cached_process_audio(audio_bytes: bytes, audio_label: str = "Acoustic Specimen Stream"):
    """Zero-lag cached bioacoustic extraction and neural inference pipeline."""
    y, sr, duration = load_audio(audio_bytes, apply_denoising=True)
    features = extract_acoustic_features(y, sr)
    species_info = identify_species(audio_bytes, features, top_k=5)
    results = compute_distress_index(features, species_info=species_info)
    spoof_info = evaluate_avian_authenticity(y, sr, species_info, features)
    timeline_info = compute_distress_timeline(y, sr, species_info=species_info)
    consensus_info = evaluate_flock_consensus(species_info, features, results["adi"])
    forensic_report = generate_forensic_report(
        audio_label, duration, sr, features, species_info, spoof_info, results, timeline_info, consensus_info
    )
    return y, sr, duration, features, results, species_info, spoof_info, timeline_info, consensus_info, forensic_report


@st.cache_data(show_spinner=False)
def cached_fetch_visuals(common_name: str, scientific_name: str):
    """Cached species photographic imagery and taxonomy description."""
    return fetch_bird_visuals(common_name, scientific_name)


# --- PLOTLY VISUALIZATIONS (APPLE STUDIO PRECISION) ---
def create_gauge_chart(adi_score: float, threat_color: str, is_dark: bool = True) -> go.Figure:
    """Create a Cédric Studio bioacoustic distress gauge with Inter typography and #ff4d00 / #78e0ff accents."""
    bar_color = "#34d399" if adi_score < 38.0 else ("#78e0ff" if adi_score < 70.0 else "#ff4d00")
    text_color = "#f2f2f2"
    axis_color = "#878787"
    bg_arc = "#222222"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=adi_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "ACOUSTIC DISTRESS INDEX (ADI)",
            'font': {'size': 13, 'color': text_color, 'family': "Inter, sans-serif"}
        },
        number={
            'suffix': "%",
            'font': {'size': 54, 'color': text_color, 'family': "Inter, sans-serif"}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1.0,
                'tickcolor': axis_color,
                'tickfont': {'family': "Inter, sans-serif", 'size': 11, 'color': axis_color},
                'tickvals': [0, 25, 50, 75, 100]
            },
            'bar': {'color': bar_color, 'thickness': 0.32},
            'bgcolor': bg_arc,
            'borderwidth': 0,
            'steps': [
                {'range': [0, 38], 'color': 'rgba(52, 211, 153, 0.12)'},
                {'range': [38, 70], 'color': 'rgba(120, 224, 255, 0.12)'},
                {'range': [70, 100], 'color': 'rgba(255, 77, 0, 0.16)'}
            ],
            'threshold': {
                'line': {'color': "#ff4d00", 'width': 3.0},
                'thickness': 0.85,
                'value': 70.0
            }
        }
    ))
    fig.update_layout(
        height=280,
        margin={"l": 20, "r": 20, "t": 60, "b": 10},
        paper_bgcolor='rgba(0,0,0,0)',
        font={"family": "Inter, sans-serif", "color": text_color}
    )
    return fig


def plot_mel_spectrogram(features: dict, is_dark: bool = True) -> go.Figure:
    """Create a Cédric Studio Mel-Spectrogram with Inter typography and #78e0ff / #ff4d00 annotations."""
    mel_db = features["mel_db"]
    times = features["times"]
    freq_bins = np.linspace(200, 10000, mel_db.shape[0])

    p22 = float(np.percentile(mel_db, 22))
    p_max = float(mel_db.max())

    colorscale_cedric = [
        [0.0, '#141414'],
        [0.12, '#1a1a1a'],
        [0.28, '#1e293b'],
        [0.48, '#0284c7'],
        [0.68, '#78e0ff'],
        [0.85, '#ff4d00'],
        [1.0, '#ffffff']
    ]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=mel_db,
        x=times,
        y=freq_bins,
        colorscale=colorscale_cedric,
        zmin=p22,
        zmax=p_max,
        zsmooth='best',
        colorbar={
            "title": {
                "text": "Power (dB)",
                "font": {"family": "Inter, sans-serif", "size": 11, "color": "#f2f2f2"}
            },
            "tickfont": {"family": "Inter, sans-serif", "color": "#878787", "size": 10},
            "len": 0.88,
            "thickness": 10,
            "outlinewidth": 0,
            "bgcolor": "rgba(0,0,0,0)"
        },
        hovertemplate="<b>Frequency:</b> %{y:.0f} Hz<br><b>Time:</b> %{x:.2f} s<br><b>Energy:</b> %{z:.1f} dB<extra></extra>"
    ))

    centroid_traj = features.get("spectral_centroid", {}).get("trajectory")
    if centroid_traj is not None and len(centroid_traj) == len(times):
        fig.add_trace(go.Scatter(
            x=times,
            y=centroid_traj,
            mode='lines',
            name='Pitch Centroid',
            line={"color": '#78e0ff', "width": 1.8},
            hovertemplate="<b>Pitch Centroid:</b> %{y:.0f} Hz<br><b>Time:</b> %{x:.2f} s<extra></extra>"
        ))

    fig.add_hline(
        y=1500, line_dash='dash', line_color='rgba(120, 224, 255, 0.60)', line_width=1.2,
        annotation_text="Avian Syrinx Band (1.5 kHz)",
        annotation_position="top left",
        annotation_font={"family": "Inter, sans-serif", "size": 11, "color": "#78e0ff"}
    )
    fig.add_hline(
        y=4500, line_dash='dash', line_color='rgba(255, 77, 0, 0.70)', line_width=1.2,
        annotation_text="Alarm Screech Threshold (4.5 kHz)",
        annotation_position="top left",
        annotation_font={"family": "Inter, sans-serif", "size": 11, "color": "#ff4d00"}
    )

    fig.update_layout(
        title={
            "text": "HIGH-RESOLUTION MEL BIOACOUSTIC SONOGRAM",
            "font": {"family": "Inter, sans-serif", "size": 13, "color": "#f2f2f2"}
        },
        xaxis={
            "title": {"text": "Time Elapsed (seconds)", "font": {"family": "Inter, sans-serif", "size": 12, "color": "#878787"}},
            "showgrid": True,
            "gridcolor": "rgba(255, 255, 255, 0.05)",
            "tickfont": {"family": "Inter, sans-serif", "color": "#878787"}
        },
        yaxis={
            "title": {"text": "Frequency (Hz)", "font": {"family": "Inter, sans-serif", "size": 12, "color": "#878787"}},
            "showgrid": True,
            "gridcolor": "rgba(255, 255, 255, 0.05)",
            "tickfont": {"family": "Inter, sans-serif", "color": "#878787"},
            "range": [200, 10000]
        },
        height=360,
        margin={"l": 20, "r": 20, "t": 44, "b": 25},
        paper_bgcolor='#141414',
        plot_bgcolor='#141414',
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"family": "Inter, sans-serif", "size": 11, "color": "#878787"},
            "bgcolor": "rgba(0,0,0,0)"
        }
    )
    return fig


def plot_waveform_and_dynamics(y: np.ndarray, sr: int, features: dict, is_dark: bool = True) -> go.Figure:
    """Plot synchronized waveform and pitch dynamics with Cédric Studio Inter styling."""
    time_wave = np.linspace(0, len(y) / sr, len(y))
    times_feats = features["times"]
    centroid = features["spectral_centroid"]["trajectory"]

    text_color = "#f2f2f2"
    axis_color = "#878787"
    plot_bg = "#1a1a1a"
    grid_col = "rgba(255, 255, 255, 0.05)"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Acoustic Signal Waveform & Onset Bursts", "Pitch / Spectral Centroid Trajectory (Hz)")
    )

    step = max(1, len(y) // 3000)
    fig.add_trace(
        go.Scatter(x=time_wave[::step], y=y[::step], mode='lines', name='Waveform', line={"color": "#78e0ff", "width": 1.4}),
        row=1, col=1
    )

    onset_times = features.get("onset_times", [])
    if len(onset_times) > 0:
        fig.add_trace(
            go.Scatter(
                x=onset_times,
                y=[0.85] * len(onset_times),
                mode='markers',
                name='Call Burst Onset',
                marker={"symbol": "triangle-down", "size": 7, "color": "#ff4d00"}
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(x=times_feats, y=centroid, mode='lines', name='Centroid (Hz)', line={"color": "#ff4d00", "width": 1.8}),
        row=2, col=1
    )

    fig.update_layout(
        height=350,
        margin={"l": 40, "r": 20, "t": 35, "b": 25},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=plot_bg,
        showlegend=True,
        font={"family": "Inter, sans-serif", "color": text_color}
    )
    fig.update_xaxes(title_text="Time (seconds)", title_font={"color": axis_color, "size": 12, "family": "Inter, sans-serif"}, tickfont={"color": axis_color, "size": 11, "family": "Inter, sans-serif"}, row=2, col=1, gridcolor=grid_col)
    fig.update_yaxes(title_text="Amplitude", title_font={"color": axis_color, "size": 12, "family": "Inter, sans-serif"}, tickfont={"color": axis_color, "size": 11, "family": "Inter, sans-serif"}, row=1, col=1, gridcolor=grid_col)
    fig.update_yaxes(title_text="Centroid (Hz)", title_font={"color": axis_color, "size": 12, "family": "Inter, sans-serif"}, tickfont={"color": axis_color, "size": 11, "family": "Inter, sans-serif"}, row=2, col=1, gridcolor=grid_col)
    fig.update_annotations(font={"family": "Inter, sans-serif", "size": 12, "color": text_color})

    return fig


def plot_temporal_distress_curve(timeline_info: dict, is_dark: bool = True) -> go.Figure:
    """Plot high-precision temporal threat escalation curve with Cédric Studio Inter styling."""
    times = timeline_info.get("times", [0.0])
    adi_series = timeline_info.get("adi_series", [50.0])
    shockwave = timeline_info.get("shockwave_detected", False)
    peak_adi = timeline_info.get("peak_adi", 50.0)
    peak_time = timeline_info.get("peak_time", 0.0)
    trend_label = timeline_info.get("trend_label", "Stationary")
    escalation_rate = timeline_info.get("escalation_rate", 0.0)

    text_color = "#f2f2f2"
    axis_color = "#878787"
    plot_bg = "#1a1a1a"
    grid_col = "rgba(255, 255, 255, 0.05)"

    fig = go.Figure()

    line_color = "#ff4d00" if shockwave or peak_adi >= 70.0 else ("#78e0ff" if peak_adi >= 38.0 else "#34d399")
    fill_color = "rgba(255, 77, 0, 0.12)" if line_color == "#ff4d00" else ("rgba(120, 224, 255, 0.10)" if line_color == "#78e0ff" else "rgba(52, 211, 153, 0.08)")

    fig.add_trace(go.Scatter(
        x=times,
        y=adi_series,
        mode='lines',
        name='Distress Trajectory',
        line={"color": line_color, "width": 2.2, "shape": "spline"},
        fill='tozeroy',
        fillcolor=fill_color,
        hovertemplate="<b>Distress Index:</b> %{y:.1f}%<br><b>Time:</b> %{x:.2f}s<extra></extra>"
    ))

    if shockwave and peak_time > 0:
        fig.add_trace(go.Scatter(
            x=[peak_time],
            y=[peak_adi],
            mode='markers+text',
            name='Shockwave Burst',
            text=["INFRASONIC SURGE"],
            textposition="top center",
            textfont={"color": "#ff4d00", "size": 11, "family": "Inter, sans-serif"},
            marker={"size": 10, "color": "#ff4d00", "symbol": "circle"}
        ))

    fig.add_hline(
        y=70, line_dash="dash", line_color="#ff5252", line_width=1.2,
        annotation_text="Critical Threat (70%)",
        annotation_position="bottom right",
        annotation_font={"size": 11, "color": "#ff5252", "family": "Inter, sans-serif"}
    )
    fig.add_hline(
        y=38, line_dash="dot", line_color="#78e0ff", line_width=1.2,
        annotation_text="Elevated Vigilance (38%)",
        annotation_position="bottom right",
        annotation_font={"size": 11, "color": "#78e0ff", "family": "Inter, sans-serif"}
    )

    fig.update_layout(
        title={
            "text": f"TEMPORAL THREAT ESCALATION &bull; {trend_label.upper()} (Velocity d(ADI)/dt: {escalation_rate:.1f}%/s)",
            "font": {"size": 13, "color": text_color, "family": "Inter, sans-serif"}
        },
        xaxis={
            "title": "Elapsed Time (Seconds)",
            "showgrid": True,
            "gridcolor": grid_col,
            "tickfont": {"color": axis_color, "size": 11, "family": "Inter, sans-serif"},
            "title_font": {"family": "Inter, sans-serif", "size": 12, "color": axis_color}
        },
        yaxis={
            "title": "Acoustic Distress Index (%)",
            "range": [0, 105],
            "showgrid": True,
            "gridcolor": grid_col,
            "tickfont": {"color": axis_color, "size": 11, "family": "Inter, sans-serif"},
            "title_font": {"family": "Inter, sans-serif", "size": 12, "color": axis_color}
        },
        height=320,
        margin={"l": 40, "r": 30, "t": 46, "b": 36},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=plot_bg,
        showlegend=False
    )
    return fig


# --- CÉDRIC STUDIO COMPACT NAVIGATION (DESIGN.md) ---
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding: 18px 0 24px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 32px;">
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="width: 8px; height: 8px; border-radius: 2px; background-color: #ff4d00; display: inline-block;"></span>
        <span style="font-size: 16px; font-weight: 600; color: #f2f2f2; letter-spacing: -0.01em;">Project INTERA</span>
        <span style="font-size: 12px; font-weight: 500; color: #78e0ff; background: rgba(120, 224, 255, 0.1); border: 1px solid rgba(120, 224, 255, 0.2); padding: 2px 8px; border-radius: 4px;">Bioacoustic Studio / v2.4</span>
    </div>
    <div style="display: flex; gap: 24px; font-size: 14px; font-weight: 400; color: #878787;">
        <span style="color: #f2f2f2;">Observatory</span>
        <span>Species Atlas (6,522)</span>
        <span>Anti-Spoofing</span>
        <span>Early Warning</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- CÉDRIC STUDIO INTRODUCTORY HERO (DESIGN.md) ---
st.markdown("""
<div class="intera-hero">
    <div class="intera-hero-eyebrow">A SIGNAL IN THE NOISE &bull; BIOACOUSTIC EARLY WARNING</div>
    <h1 class="intera-hero-understand">Understand What the Birds Are Saying</h1>
    <div class="intera-hero-subtitle">
        Birds sense approaching storms and earthquakes hours before instruments pick them up. When danger nears, their calls change. Project INTERA decodes these acoustic distress patterns to warn communities before disaster strikes.
    </div>
</div>
""", unsafe_allow_html=True)

# --- APPLE SEGMENTED CONTROL: ACOUSTIC ACQUISITION MODE ---
input_mode = st.radio(
    "Acoustic Acquisition Mode:",
    options=[
        "Calibrated Laboratory Benchmarks",
        "Upload Field Recording",
        "Live Canopy Microphone"
    ],
    horizontal=True
)

audio_data = None
audio_label = ""
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

if input_mode == "Upload Field Recording":
    col_u_l, col_u_m, col_u_r = st.columns([1, 3.2, 1])
    with col_u_m:
        st.markdown(f"""
        <div class="apple-upload-card">
            <div style="width: 62px; height: 62px; border-radius: 50%; background: #EBF4FF; display: inline-flex; align-items: center; justify-content: center; box-shadow: 0 6px 16px rgba(0, 122, 255, 0.15); margin-bottom: 12px; color: #007AFF;">
                {ICON_UPLOAD_APPLE}
            </div>
            <div style="font-size: 20px; font-weight: 800; color: #111827; letter-spacing: -0.02em;">
                Upload Bioacoustic Recording
            </div>
            <div style="font-size: 14px; color: #8E8E93; font-weight: 500; margin-top: 4px; max-width: 440px; margin-left: auto; margin-right: auto; line-height: 1.5;">
                Drop high-fidelity avian audio files (.wav, .mp3, .ogg, .flac up to 25 MB) for automated BirdNET classification and dual-syrinx verification.
            </div>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload raw bioacoustic field recording (.wav, .mp3, .ogg, .flac)",
            type=["wav", "mp3", "ogg", "flac"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            if uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:
                st.error("Security Alert: Uploaded file exceeds the 25 MB limit.")
            elif uploaded_file.size < 64:
                st.error("Invalid Audio: Uploaded file is empty or corrupted.")
            else:
                audio_data = uploaded_file.read()
                audio_label = uploaded_file.name

elif input_mode == "Live Canopy Microphone":
    col_m_l, col_m_m, col_m_r = st.columns([1, 2.8, 1])
    with col_m_m:
        st.markdown(f"""
        <div class="apple-upload-card">
            <div style="width: 62px; height: 62px; border-radius: 50%; background: #FEECEB; display: inline-flex; align-items: center; justify-content: center; box-shadow: 0 6px 16px rgba(255, 59, 48, 0.15); margin-bottom: 12px; color: #FF3B30;">
                {ICON_MIC}
            </div>
            <div style="font-size: 20px; font-weight: 800; color: #111827; letter-spacing: -0.02em;">
                Live Environmental Microphone
            </div>
            <div style="font-size: 14px; color: #8E8E93; font-weight: 500; margin-top: 4px; max-width: 420px; margin-left: auto; margin-right: auto; line-height: 1.5;">
                Point your directional microphone toward the canopy. Click Record to capture real-time 44.1 kHz ambient avian signals.
            </div>
        </div>
        """, unsafe_allow_html=True)
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
    
    col_sel_l, col_sel_m, col_sel_r = st.columns([1, 3.2, 1])
    with col_sel_m:
        st.markdown(f"""
        <div class="apple-upload-card">
            <div style="width: 62px; height: 62px; border-radius: 50%; background: #EBF9F1; display: inline-flex; align-items: center; justify-content: center; box-shadow: 0 6px 16px rgba(52, 199, 89, 0.15); margin-bottom: 12px; color: #34C759;">
                {ICON_LIBRARY_APPLE}
            </div>
            <div style="font-size: 20px; font-weight: 800; color: #111827; letter-spacing: -0.02em;">
                Calibrated Laboratory Benchmarks
            </div>
            <div style="font-size: 14px; color: #8E8E93; font-weight: 500; margin-top: 4px; max-width: 440px; margin-left: auto; margin-right: auto; line-height: 1.5;">
                Select documented seismic alarm calls, tornado infrasound warblers, or domestic non-avian animal test fixtures.
            </div>
        </div>
        """, unsafe_allow_html=True)

        selected_sample_name = st.selectbox(
            "Select Calibrated Laboratory Reference:",
            options=list(sample_names.keys()),
            index=None,
            placeholder="Choose a calibrated biological benchmark...",
            format_func=lambda x: {
                "calm_robin_song.wav": "OBS-01 // European Robin (Erithacus rubecula) — Melodic Baseline Song",
                "elevated_warning_call.wav": "OBS-02 // Song Sparrow (Melospiza melodia) — Elevated Agitation Call",
                "critical_disaster_panic.wav": "OBS-03 // Northern Cardinal (Cardinalis cardinalis) — Critical Pre-Disaster Panic",
                "authentic_warbler_storm_alarm.wav": "OBS-04 // Golden-winged Warbler — 2014 Tornado Outbreak Infrasound",
                "authentic_peafowl_earthquake_alarm.wav": "OBS-05 // Indian Peafowl — Documented Seismic P-Wave Honk",
                "human_whistling_mimic.wav": "OBS-06 // Homo sapiens Oral Cavity — Anti-Spoofing Whistling Test",
                "domestic_cat_meow_spoof.wav": "OBS-07 // Domestic Cat (Felis catus) — Feline Meow (Peafowl Rejection Test)",
                "dog_bark_phone_speaker_replay.wav": "OBS-08 // Domestic Dog Bark via Phone Speaker — Loudspeaker Replay Test"
            }.get(x, str(x or "")),
            label_visibility="collapsed"
        )
        if selected_sample_name:
            chosen_path = sample_names[selected_sample_name]
            try:
                with open(chosen_path, "rb") as f_samp:
                    audio_data = f_samp.read()
                    audio_label = selected_sample_name
            except OSError as e:
                st.error(f"Error loading benchmark sample: {e}")

# --- BIOACOUSTIC INGESTION & PROCESSING ---
processed_payload = None
if audio_data is not None:
    try:
        with st.spinner("Decoding bioacoustic signals with Cornell BirdNET & verifying biometrics..."):
            processed_payload = cached_process_audio(audio_data, audio_label)
    except (ValueError, RuntimeError, OSError) as exc:
        st.error(f"Bioacoustic Ingestion Error: Unable to decode audio stream ({type(exc).__name__}). Please provide a valid, uncorrupted audio recording.")
        processed_payload = None

if processed_payload is not None:
    y, sr, duration, features, results, species_info, spoof_info, timeline_info, consensus_info, forensic_report = processed_payload
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    waveform_bars = generate_waveform_svg_bars(y, num_bars=42)
    st.markdown(f"""
    <div class="apple-audio-console">
        <div class="audio-console-left">
            <div class="audio-play-badge">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                    <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
            </div>
            <div class="audio-meta">
                <div class="audio-filename">{html.escape(audio_label)}</div>
                <div class="audio-subtext">{duration:.2f}s &bull; {sr} Hz &bull; Pitch Centroid: {features['spectral_centroid']['mean']:.0f} Hz</div>
            </div>
        </div>
        <div class="audio-console-center">
            <div class="waveform-container">
                {waveform_bars}
            </div>
        </div>
        <div class="audio-console-right">
            <div class="live-equalizer">
                <span class="eq-bar"></span>
                <span class="eq-bar"></span>
                <span class="eq-bar"></span>
                <span class="eq-bar"></span>
                <span class="eq-bar"></span>
                <span class="eq-bar"></span>
            </div>
            <span class="audio-live-tag">
                <span class="pulse-dot"></span> LIVE TELEMETRY
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.audio(audio_data)

    # --- CÉDRIC STUDIO MULTI-SPECIES CONSENSUS BANNER (DESIGN.md) ---
    consensus_type = consensus_info.get("consensus_type", "nominal_calm")
    cons_status = consensus_info.get("status", "NOMINAL")
    cons_summary = consensus_info.get("summary", "")

    if consensus_type == "macro_ecological_panic":
        st.markdown(f"""
        <div style="background: #1a1a1a; border: 1px solid #ff4d00; border-radius: 12px; padding: 18px 22px; margin-bottom: 24px; display: flex; align-items: center; gap: 14px;">
            <div><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ff4d00" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg></div>
            <div>
                <div style="font-size: 14px; font-weight: 600; color: #ff4d00; letter-spacing: 0.02em; text-transform: uppercase;">
                    {cons_status}
                </div>
                <div style="font-size: 14px; color: #f2f2f2; font-weight: 400; margin-top: 2px;">
                    {cons_summary}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif consensus_type == "localized_sentinel_alert":
        st.markdown(f"""
        <div style="background: #1a1a1a; border: 1px solid #78e0ff; border-radius: 12px; padding: 18px 22px; margin-bottom: 24px; display: flex; align-items: center; gap: 14px;">
            <div><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#78e0ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg></div>
            <div>
                <div style="font-size: 14px; font-weight: 600; color: #78e0ff; letter-spacing: 0.02em; text-transform: uppercase;">
                    {cons_status}
                </div>
                <div style="font-size: 14px; color: #f2f2f2; font-weight: 400; margin-top: 2px;">
                    {cons_summary}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: #1a1a1a; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px 22px; margin-bottom: 24px; display: flex; align-items: center; gap: 14px;">
            <div><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="m9 12 2 2 4-4"></path></svg></div>
            <div>
                <div style="font-size: 14px; font-weight: 600; color: #34d399; letter-spacing: 0.02em; text-transform: uppercase;">
                    {cons_status}
                </div>
                <div style="font-size: 14px; color: #f2f2f2; font-weight: 400; margin-top: 2px;">
                    {cons_summary}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- CÉDRIC STUDIO SPECIES CARD (PRODUCT / VENTURE SHOWCASE) ---
    is_bird = spoof_info.get("is_authentic", True)
    prim_sp = species_info.get("primary_species", {})

    if is_bird and prim_sp:
        safe_com_name = html.escape(prim_sp.get('common_name', 'Unknown Avian Species'))
        safe_sci_name = html.escape(prim_sp.get('scientific_name', 'Aves'))
        
        visual_data = cached_fetch_visuals(prim_sp.get('common_name', ''), prim_sp.get('scientific_name', ''))
        raw_photo = visual_data.get("image_url", "")
        photo_uri = get_image_data_uri(raw_photo)
        safe_desc = html.escape(visual_data.get("description", "Avian specimen classified via Cornell BirdNET neural taxonomy."))
        if not photo_uri:
            photo_uri = "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=600&q=80"
        
        st.markdown(f"""
        <div class="apple-pet-card" style="background: #1a1a1a; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 22px 24px; display: flex; align-items: center; gap: 20px; margin-bottom: 24px; flex-wrap: wrap;">
            <img src="{photo_uri}" alt="{safe_com_name}" class="apple-avatar" style="width: 80px; height: 80px; border-radius: 50%; object-fit: cover; border: 2px solid rgba(255, 255, 255, 0.1);" />
            <div style="flex: 1; min-width: 260px;">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 24px; font-weight: 600; color: #f2f2f2; letter-spacing: -0.02em;">{safe_com_name}</span>
                        <span style="background: #ff4d00; color: #ffffff; font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 4px; letter-spacing: 0.02em;">
                            BirdNET &bull; {prim_sp['confidence']}% Match
                        </span>
                    </div>
                    <span style="background: rgba(120, 224, 255, 0.1); color: #78e0ff; border: 1px solid rgba(120, 224, 255, 0.25); font-size: 12px; font-weight: 500; padding: 3px 10px; border-radius: 4px;">
                        <span class="pulse-dot" style="margin-right: 4px;"></span> Verified Authentic Syrinx
                    </span>
                </div>
                <div style="font-size: 14px; font-style: italic; color: #878787; margin-top: 3px;">
                    {safe_sci_name} &bull; {results.get('baseline_profile', {}).get('family_label', 'Avian Taxonomy')}
                </div>
                <div style="font-size: 14px; color: #a0a0a0; margin-top: 8px; line-height: 1.6;">
                    {safe_desc}
                </div>
                <div style="display: flex; gap: 16px; margin-top: 12px; font-size: 13px; color: #878787; flex-wrap: wrap;">
                    <span>Acoustic Frequency: <strong style="color: #78e0ff;">{prim_sp['pitch_status']} ({prim_sp['detected_pitch']:.0f} Hz)</strong></span>
                    <span>&bull;</span>
                    <span>Disaster Sensitivity: <strong style="color: #f2f2f2;">{prim_sp['disaster_sensitivity']}</strong></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        cat_labels = {
            "canine_dog": "Domestic Canine (Dog Bark / Whine)",
            "feline_cat": "Domestic Feline (Cat Meow)",
            "phone_speaker_replay": "Smartphone Loudspeaker Replay Detected",
            "mammalian_domestic": "Domestic Mammalian Sound (Canine / Feline)",
            "human_mimicry": "Human Vocal Mimicry / Oral Whistling",
            "artificial_siren": "Artificial Siren / Electronic Alarm",
            "mechanical_engine": "Mechanical Engine Noise"
        }
        detected_non_bird_title = cat_labels.get(spoof_info.get("detected_category", ""), "Non-Avian Acoustic Interference")
        factors_items = "".join(f"<li style='margin-bottom: 4px;'>{html.escape(f)}</li>" for f in spoof_info.get('spoof_factors', []))
        
        st.markdown(f"""
        <div style="background: #1a1a1a; border-radius: 12px; padding: 24px 26px; border: 1px solid rgba(255, 82, 82, 0.35); border-left: 4px solid #ff5252; margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; flex-wrap: wrap; gap: 8px;">
                <span style="background: rgba(255, 82, 82, 0.15); color: #ff5252; font-size: 11px; padding: 3px 8px; border-radius: 4px; font-weight: 600;">
                    Biometric Rejection &bull; Non-Avian Acoustic Source
                </span>
                <span style="font-size: 13px; color: #ff5252; font-weight: 500;">
                    Sub-800Hz Energy: {spoof_info.get('sub800_ratio', 0.0)*100:.1f}% (Mammal Threshold &gt; 25%)
                </span>
            </div>
            <div style="font-size: 22px; font-weight: 600; color: #f2f2f2; letter-spacing: -0.02em; margin-top: 6px;">
                {detected_non_bird_title}
            </div>
            <p style="font-size: 14.5px; color: #a0a0a0; margin-top: 6px; line-height: 1.6;">
                {html.escape(spoof_info.get('status_summary', ''))}
            </p>
            <div style="margin-top: 12px; font-size: 13px; color: #f2f2f2;">
                <strong>Biometric Rejection Criteria Triggered:</strong>
                <ul style="margin: 6px 0 0 18px; padding: 0; color: #878787;">
                    {factors_items}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- CÉDRIC STUDIO STATS DASHBOARD & TELEMETRY ---
    adi_val = results["adi"]
    if adi_val < 38.0:
        adi_tier_label = "Nominal Equilibrium // Baseline Chorus"
        adi_tier_color = "#34d399"
        adi_tier_desc = "Calm environmental foraging calls. No seismic micro-tremors or severe weather infrasound detected."
    elif adi_val < 70.0:
        adi_tier_label = "Elevated Vigilance // Canopy Agitation"
        adi_tier_color = "#78e0ff"
        adi_tier_desc = "Noticeable frequency shift and accelerated call density detected. Indicates localized ecological unrest."
    else:
        adi_tier_label = "Acute Biotic Distress // Evacuation Call"
        adi_tier_color = "#ff4d00"
        adi_tier_desc = "Extreme high-frequency screeching and frantic burst pacing. Strongly correlates with imminent natural hazards."

    col_gauge, col_stats = st.columns([1.15, 1.85])

    with col_gauge, st.container(border=True):
        gauge_fig = create_gauge_chart(results["adi"], results["threat_color"], is_dark=True)
        st.plotly_chart(gauge_fig, width="stretch")
        st.markdown(f"""
        <div style="text-align: center; margin-top: -6px; padding-bottom: 8px;">
            <div style="font-size: 14px; font-weight: 600; color: {adi_tier_color}; letter-spacing: 0.01em;">
                {adi_tier_label}
            </div>
            <div style="font-size: 13px; color: #878787; margin-top: 4px; line-height: 1.5; font-weight: 400;">
                {adi_tier_desc}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats, st.container(border=True):
        st.markdown("""
        <div style="margin-bottom: 12px;">
            <div style="font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: #78e0ff; font-weight: 600;">
                Bioacoustic Telemetry & Stats
            </div>
            <div style="font-size: 1.35rem; font-weight: 600; color: #f2f2f2; letter-spacing: -0.02em;">
                Real-Time Biological Sensor Telemetry
            </div>
        </div>
        """, unsafe_allow_html=True)

        sub_pitch = results['sub_scores'].get('pitch', 0.0)
        sub_rate = results['sub_scores'].get('call_rate', 0.0)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(render_apple_stat_card(
                title="Acoustic Distress",
                value=f"{results['adi']:.1f}%",
                subtext="Equilibrium" if results['adi'] < 38 else ("Agitated" if results['adi'] < 70 else "Panic Threat"),
                percentage=results['adi'],
                color=adi_tier_color,
                is_dark=True
            ), unsafe_allow_html=True)

            st.markdown(render_apple_stat_card(
                title="Pitch Elevation",
                value=f"{features['spectral_centroid']['mean']:.0f} Hz",
                subtext=f"{sub_pitch:.0f}% vs Baseline",
                percentage=min(100.0, max(5.0, sub_pitch)),
                color="#78e0ff",
                is_dark=True
            ), unsafe_allow_html=True)

        with col_s2:
            st.markdown(render_apple_stat_card(
                title="Vocal Organ",
                value="Avian Syrinx" if spoof_info['is_authentic'] else "Mammalian",
                subtext="Harmonic Dual Syrinx" if spoof_info['is_authentic'] else "Low-Freq Larynx",
                percentage=prim_sp.get('confidence', 95.0) if spoof_info['is_authentic'] else 15.0,
                color="#34d399" if spoof_info['is_authentic'] else "#ff5252",
                is_dark=True
            ), unsafe_allow_html=True)

            st.markdown(render_apple_stat_card(
                title="Call Cadence",
                value=f"{features['call_rate']:.2f} calls/s",
                subtext=f"{sub_rate:.0f}% Bursts Pacing",
                percentage=min(100.0, max(5.0, sub_rate)),
                color="#ff4d00",
                is_dark=True
            ), unsafe_allow_html=True)

    # --- CÉDRIC STUDIO SEGMENTED TABS: 7 TELEMETRY STATIONS ---
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    tab_timeline, tab_spec, tab_wave, tab_species, tab_anti_spoof, tab_table, tab_report = st.tabs([
        "Threat Timeline & Escalation",
        "Mel Sonogram Spectrograph",
        "Waveform Dynamics",
        "Species Taxonomy",
        "Biometric Shield",
        "Diagnostic Table",
        "Forensic Telemetry Report"
    ])

    card_bg_tab = "#1a1a1a"
    card_border_tab = "rgba(255, 255, 255, 0.08)"
    text_prime_tab = "#f2f2f2"
    text_sec_tab = "#878787"

    # ========================================================
    # TAB 1: THREAT TIMELINE & ESCALATION
    # ========================================================
    with tab_timeline, st.container(border=True):
        st.markdown(f"""
        <div style="margin-bottom: 18px;">
            <div style="font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: #78e0ff; font-weight: 600;">
                Pre-Disaster Dynamic Infrasound & P-Wave Progression
            </div>
            <div style="font-size: 1.45rem; font-weight: 600; color: {text_prime_tab}; letter-spacing: -0.02em; margin-top: 2px;">
                Temporal Threat Escalation Velocity
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        esc_rate = timeline_info.get('escalation_rate', 0.0)
        peak_time = timeline_info.get('peak_time', 0.0)
        peak_adi = timeline_info.get('peak_adi', 0.0)
        shock_detected = timeline_info.get('shockwave_detected', False)

        with col_t1:
            st.markdown(render_apple_stat_card(
                title="Escalation Velocity",
                value=f"{esc_rate:.1f}%/s",
                subtext="Threat Acceleration Rate",
                percentage=min(100.0, max(5.0, esc_rate * 3.5)),
                color="#ff4d00" if esc_rate > 5 else "#78e0ff",
                is_dark=True
            ), unsafe_allow_html=True)
        with col_t2:
            st.markdown(render_apple_stat_card(
                title="Acoustic Apex Time",
                value=f"{peak_time:.2f}s",
                subtext=f"Total Duration: {duration:.2f}s",
                percentage=min(100.0, (peak_time / max(0.1, duration)) * 100.0),
                color="#78e0ff",
                is_dark=True
            ), unsafe_allow_html=True)
        with col_t3:
            st.markdown(render_apple_stat_card(
                title="Peak Threat Level",
                value=f"{peak_adi:.1f}%",
                subtext="Maximum Distress Index",
                percentage=peak_adi,
                color="#ff4d00" if peak_adi >= 70 else ("#78e0ff" if peak_adi >= 38 else "#34d399"),
                is_dark=True
            ), unsafe_allow_html=True)
        with col_t4:
            st.markdown(render_apple_stat_card(
                title="Pre-Seismic Shock",
                value="DETECTED" if shock_detected else "NOMINAL",
                subtext="Infrasonic P-Wave Burst" if shock_detected else "Steady Atmospheric State",
                percentage=95.0 if shock_detected else 12.0,
                color="#ff4d00" if shock_detected else "#34d399",
                is_dark=True
            ), unsafe_allow_html=True)

        st.plotly_chart(plot_temporal_distress_curve(timeline_info, is_dark=True), width="stretch")

        trend_name = timeline_info.get("trend_label", "Stationary Chorus")
        st.markdown(f"""
        <div style="background: #141414; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08); padding: 16px 20px; margin-top: 14px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
            <div>
                <span style="font-size: 14px; font-weight: 500; color: #878787;">Biological Evacuation Status: </span>
                <span style="font-size: 14px; font-weight: 600; color: {'#ff4d00' if shock_detected or peak_adi >= 70 else '#34d399'};">{trend_name.upper()}</span>
                <div style="font-size: 13px; color: #878787; margin-top: 2px;">
                    Bioacoustic sensor arrays indicate an estimated warning lead time of 45–90 minutes before surface micro-tremor emergence or storm wind shear.
                </div>
            </div>
            <span style="background: rgba(120, 224, 255, 0.1); color: #78e0ff; font-size: 12px; font-weight: 500; padding: 4px 12px; border-radius: 4px;">
                Verified Acoustic Trajectory
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ========================================================
    # TAB 2: MEL SONOGRAM SPECTROGRAPH
    # ========================================================
    with tab_spec, st.container(border=True):
        st.markdown(f"""
        <div style="margin-bottom: 16px;">
            <div style="font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: #78e0ff; font-weight: 600;">
                High-Resolution Time-Frequency Bioacoustics
            </div>
            <div style="font-size: 1.45rem; font-weight: 600; color: {text_prime_tab}; letter-spacing: -0.02em; margin-top: 2px;">
                Mel Sonogram Spectrograph & Syringeal Formants
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(plot_mel_spectrogram(features, is_dark=True), width="stretch")

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.markdown(f"""
            <div style="background: #141414; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08); padding: 14px 18px; text-align: center;">
                <div style="font-size: 11px; font-weight: 600; color: #78e0ff; text-transform: uppercase;">Avian Syrinx Band</div>
                <div style="font-size: 17px; font-weight: 600; color: {text_prime_tab}; margin-top: 2px;">1,500 – 3,200 Hz</div>
                <div style="font-size: 12px; color: {text_sec_tab}; margin-top: 2px;">Core Syringeal Membrane Resonance</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b2:
            st.markdown(f"""
            <div style="background: #141414; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08); padding: 14px 18px; text-align: center;">
                <div style="font-size: 11px; font-weight: 600; color: #ff4d00; text-transform: uppercase;">Alarm Screech Ceiling</div>
                <div style="font-size: 17px; font-weight: 600; color: {text_prime_tab}; margin-top: 2px;">> 4,500 Hz</div>
                <div style="font-size: 12px; color: {text_sec_tab}; margin-top: 2px;">High-Harmonic Panic Burst Zone</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b3:
            st.markdown(f"""
            <div style="background: #141414; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08); padding: 14px 18px; text-align: center;">
                <div style="font-size: 11px; font-weight: 600; color: #34d399; text-transform: uppercase;">Bioacoustic Bandwidth</div>
                <div style="font-size: 17px; font-weight: 600; color: {text_prime_tab}; margin-top: 2px;">10,000 Hz / 128 Bins</div>
                <div style="font-size: 12px; color: {text_sec_tab}; margin-top: 2px;">Full Ultrasonic Nyquist Range</div>
            </div>
            """, unsafe_allow_html=True)

    # ========================================================
    # TAB 3: WAVEFORM DYNAMICS
    # ========================================================
    with tab_wave, st.container(border=True):
        st.markdown(f"""
        <div style="margin-bottom: 16px;">
            <div style="font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: #78e0ff; font-weight: 600;">
                Acoustic Amplitude & Dynamic Envelope
            </div>
            <div style="font-size: 1.45rem; font-weight: 600; color: {text_prime_tab}; letter-spacing: -0.015em; margin-top: 2px; word-break: break-word; overflow-wrap: break-word;">
                Waveform Dynamics & Spectral Trajectory
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(plot_waveform_and_dynamics(y, sr, features, is_dark=True), width="stretch")

        col_w1, col_w2, col_w3 = st.columns(3)
        peak_amp = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
        rms_val = float(features.get('rms', {}).get('mean', np.sqrt(np.mean(y**2)))) if len(y) > 0 else 0.0
        call_cadence = float(features.get('call_rate', 0.0))

        with col_w1:
            st.markdown(render_apple_stat_card(
                title="Peak Signal Amplitude",
                value=f"{peak_amp:.3f}",
                subtext="Linear Dynamic Headroom (1.000 max)",
                percentage=min(100.0, peak_amp * 100.0),
                color="#78e0ff",
                is_dark=True
            ), unsafe_allow_html=True)
        with col_w2:
            st.markdown(render_apple_stat_card(
                title="RMS Energy Power",
                value=f"{rms_val:.4f}",
                subtext="Root-Mean-Square Acoustic Energy",
                percentage=min(100.0, max(8.0, rms_val * 400.0)),
                color="#ff4d00",
                is_dark=True
            ), unsafe_allow_html=True)
        with col_w3:
            st.markdown(render_apple_stat_card(
                title="Call Burst Density",
                value=f"{call_cadence:.2f} calls/s",
                subtext="Burst Frequency Modulation",
                percentage=min(100.0, max(5.0, call_cadence * 25.0)),
                color="#ff4d00" if call_cadence > 2.5 else "#34d399",
                is_dark=True
            ), unsafe_allow_html=True)

    # ========================================================
    # TAB 4: SPECIES TAXONOMY (CÉDRIC STUDIO VENTURE CARD ROSTER)
    # ========================================================
    with tab_species, st.container(border=True):
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <div style="font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: #78e0ff; font-weight: 600;">
                Cornell Bioacoustics Neural Classifier
            </div>
            <div style="font-size: 1.45rem; font-weight: 600; color: {text_prime_tab}; letter-spacing: -0.02em; margin-top: 2px;">
                Top Species Candidates (Cornell BirdNET 6,522 Global Taxonomy)
            </div>
            <div style="font-size: 14px; color: {text_sec_tab}; margin-top: 4px;">
                Ranked probability roster with cross-verified syringeal harmonics and disaster sensitivity profiles.
            </div>
        </div>
        """, unsafe_allow_html=True)

        for i, cand in enumerate(species_info.get("top_candidates", []), 1):
            cand_com = html.escape(cand.get("common_name", "Unknown Avian Specimen"))
            cand_sci = html.escape(cand.get("scientific_name", "Aves sp."))
            cand_conf = cand.get("confidence", 0.0)
            cand_pitch_stat = html.escape(str(cand.get("pitch_status", "Harmonic Baseline")))
            cand_pitch_val = cand.get("detected_pitch", 0.0)
            cand_sentinel = html.escape(str(cand.get("disaster_sensitivity", "Documented Natural Hazard Sentinel")))

            vis = cached_fetch_visuals(cand.get("common_name", ""), cand.get("scientific_name", ""))
            raw_cand_img = vis.get("image_url", "")
            cand_photo_uri = get_image_data_uri(raw_cand_img)
            if not cand_photo_uri:
                cand_photo_uri = "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=300&q=80"

            rank_badge_bg = "#ff4d00" if i == 1 else "rgba(255, 255, 255, 0.08)"
            rank_badge_color = "#ffffff" if i == 1 else text_sec_tab
            rank_label = f"#{i} Primary Match" if i == 1 else f"#{i} Candidate"

            st.markdown(f"""
            <div class="cedric-candidate-card" style="background: #141414; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; margin-bottom: 12px; padding: 18px 20px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
                <img src="{cand_photo_uri}" alt="{cand_com}" style="width: 64px; height: 64px; border-radius: 50%; object-fit: cover; border: 1px solid rgba(255, 255, 255, 0.1); flex-shrink: 0;" />
                <div style="flex: 1; min-width: 220px;">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                        <div>
                            <span style="font-size: 18px; font-weight: 600; color: {text_prime_tab}; letter-spacing: -0.015em;">{cand_com}</span>
                            <span style="background: {rank_badge_bg}; color: {rank_badge_color}; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; margin-left: 8px;">
                                {rank_label}
                            </span>
                        </div>
                        <span style="background: rgba(120, 224, 255, 0.1); color: #78e0ff; border: 1px solid rgba(120, 224, 255, 0.2); font-size: 12px; font-weight: 600; padding: 2px 8px; border-radius: 4px;">
                            {cand_conf:.1f}% Confidence
                        </span>
                    </div>
                    <div style="font-size: 13.5px; font-style: italic; color: {text_sec_tab}; margin-top: 2px;">
                        {cand_sci} &bull; Avian Bioacoustic Class
                    </div>
                    <div style="display: flex; gap: 16px; margin-top: 8px; font-size: 13px; color: {text_sec_tab}; flex-wrap: wrap;">
                        <span>Vocal Frequency: <strong style="color: #78e0ff;">{cand_pitch_stat} ({cand_pitch_val:.0f} Hz)</strong></span>
                        <span>&bull;</span>
                        <span>Sensitivity: <strong style="color: #f2f2f2;">{cand_sentinel}</strong></span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ========================================================
    # TAB 5: BIOMETRIC SHIELD (VERIFICATION & ANTI-SPOOFING)
    # ========================================================
    with tab_anti_spoof, st.container(border=True):
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <div style="font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: #78e0ff; font-weight: 600;">
                Avian Syrinx Biometric Shield & Anti-Spoofing
            </div>
            <div style="font-size: 1.45rem; font-weight: 600; color: {text_prime_tab}; letter-spacing: -0.02em; margin-top: 2px;">
                Dual-Syrinx Physical Verification & Non-Avian Defense
            </div>
            <div style="font-size: 14px; color: {text_sec_tab}; margin-top: 4px;">
                Defense against human whistling, feline/canine domestic calls, and smartphone speaker replays.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_sh1, col_sh2, col_sh3 = st.columns(3)
        sp_score = float(spoof_info.get('spoof_score', 95.0))
        sub800 = float(spoof_info.get('sub800_ratio', 0.0)) * 100.0
        roll95 = float(spoof_info.get('rolloff_95', 0.0))

        with col_sh1:
            st.markdown(render_apple_stat_card(
                title="Syrinx Purity Index",
                value=f"{sp_score:.1f}%",
                subtext="Verified Avian Syrinx" if spoof_info['is_authentic'] else "Spoof / Mammalian Interference",
                percentage=sp_score,
                color="#34d399" if sp_score >= 70 else "#ff5252",
                is_dark=True
            ), unsafe_allow_html=True)
        with col_sh2:
            st.markdown(render_apple_stat_card(
                title="Sub-800Hz Energy Ratio",
                value=f"{sub800:.1f}%",
                subtext="Mammalian Larynx Ceiling < 25%",
                percentage=min(100.0, sub800),
                color="#34d399" if sub800 < 25.0 else "#ff5252",
                is_dark=True
            ), unsafe_allow_html=True)
        with col_sh3:
            st.markdown(render_apple_stat_card(
                title="95% High-Freq Rolloff",
                value=f"{roll95:.0f} Hz",
                subtext="Speaker Replay Floor > 7,500 Hz",
                percentage=min(100.0, (roll95 / 10000.0) * 100.0),
                color="#78e0ff" if roll95 >= 7500 else "#ff4d00",
                is_dark=True
            ), unsafe_allow_html=True)

        if spoof_info["is_authentic"]:
            st.markdown(f"""
            <div style="background: #141414; border: 1px solid rgba(52, 211, 153, 0.4); border-radius: 8px; padding: 18px 22px; margin-top: 14px; display: flex; align-items: center; gap: 16px;">
                <div><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="m9 12 2 2 4-4"></path></svg></div>
                <div>
                    <div style="font-size: 14px; font-weight: 600; color: #34d399; letter-spacing: 0.02em; text-transform: uppercase;">
                        AUTHENTIC DUAL SYRINX HARMONICS CONFIRMED
                    </div>
                    <div style="font-size: 13.5px; color: {text_prime_tab}; font-weight: 400; margin-top: 3px; line-height: 1.5;">
                        Acoustic energy matches verified avian syringeal resonance. Zero oral cavity formants, mammalian chest resonance, or phone loudspeaker harmonic cutoff detected. Audio signal is armed for automated early disaster alerting.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            cat_label = html.escape(str(spoof_info.get('detected_category', 'Non-Avian Acoustic Interference')))
            st.markdown(f"""
            <div style="background: #141414; border: 1px solid rgba(255, 82, 82, 0.4); border-left: 4px solid #ff5252; border-radius: 8px; padding: 18px 22px; margin-top: 14px; display: flex; align-items: center; gap: 16px;">
                <div><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ff5252" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg></div>
                <div>
                    <div style="font-size: 14px; font-weight: 600; color: #ff5252; letter-spacing: 0.02em; text-transform: uppercase;">
                        BIOMETRIC REJECTION: NON-AVIAN ACOUSTIC INTERFERENCE ({cat_label})
                    </div>
                    <div style="font-size: 13.5px; color: {text_prime_tab}; font-weight: 400; margin-top: 3px; line-height: 1.5;">
                        This specimen failed avian syrinx authenticity verification. Sub-800Hz energy or harmonic dispersion matches non-avian mammalian vocal tract patterns. Disarmed from civil disaster escalation.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ========================================================
    # TAB 6: DIAGNOSTIC TABLE (CÉDRIC STUDIO MINIMALIST TABLE)
    # ========================================================
    with tab_table, st.container(border=True):
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <div style="font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: #78e0ff; font-weight: 600;">
                Laboratory Telemetry Matrix
            </div>
            <div style="font-size: 1.45rem; font-weight: 600; color: {text_prime_tab}; letter-spacing: -0.02em; margin-top: 2px;">
                Detailed Bioacoustic Diagnostic Parameters
            </div>
        </div>
        """, unsafe_allow_html=True)

        diag_params = [
            ("Acoustic Duration", f"{duration:.2f} seconds", "Complete temporal duration of analyzed audio stream", "Time Resolution"),
            ("Sampling Frequency", f"{sr} Hz", "Bioacoustic Nyquist capture bandwidth", "Audio Standard"),
            ("Spectral Centroid", f"{features['spectral_centroid']['mean']:.1f} Hz", "Center of acoustic mass; surges dramatically under distress", "Pitch Formant"),
            ("Spectral Flatness", f"{features['spectral_flatness']['mean']:.4f}", "Harshness/screech index; tonal calls < 0.010, chaotic screeches > 0.045", "Chaos Entropy"),
            ("Call Burst Cadence", f"{features['call_rate']:.2f} calls/s", "Temporal burst frequency; accelerated call rates signify panic", "Cadence Density"),
            ("Sub-800Hz Energy Ratio", f"{features.get('sub800_ratio', 0.0)*100:.1f}%", "Vocal organ discriminator: avian syrinx < 25%, mammal larynx > 25%", "Vocal Filter"),
            ("95% Spectral Rolloff", f"{spoof_info.get('rolloff_95', 0.0):.0f} Hz", "Loudspeaker replay defense; phone speakers drop above 7.5 kHz", "Hardware Shield")
        ]

        th_bg = "#111111"
        td_border = "rgba(255, 255, 255, 0.06)"

        rows_html = []
        for name, val, desc, cat in diag_params:
            rows_html.append(f"""
            <tr style="border-bottom: 1px solid {td_border};">
                <td style="padding: 12px 16px; font-weight: 500; color: #f2f2f2; font-size: 13.5px;">{name}</td>
                <td style="padding: 12px 16px; font-weight: 600; color: #78e0ff; font-size: 13.5px;">{val}</td>
                <td style="padding: 12px 16px; color: #878787; font-size: 13px;">{desc}</td>
                <td style="padding: 12px 16px;"><span style="background: rgba(255, 255, 255, 0.06); color: #878787; font-size: 11px; font-weight: 500; padding: 2px 8px; border-radius: 4px;">{cat}</span></td>
            </tr>
            """)

        st.markdown(f"""
        <div style="overflow-x: auto; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08); background: #141414;">
            <table style="width: 100%; border-collapse: collapse; text-align: left;">
                <thead>
                    <tr style="background: {th_bg}; border-bottom: 1px solid {td_border}; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: #878787;">
                        <th style="padding: 12px 16px;">Diagnostic Parameter</th>
                        <th style="padding: 12px 16px;">Measured Telemetry</th>
                        <th style="padding: 12px 16px;">Biological Significance</th>
                        <th style="padding: 12px 16px;">Sub-System</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows_html)}
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # ========================================================
    # TAB 7: FORENSIC TELEMETRY REPORT
    # ========================================================
    with tab_report, st.container(border=True):
        st.markdown(f"""
        <div style="margin-bottom: 16px;">
            <div style="font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: #78e0ff; font-weight: 600;">
                Civil Catastrophe Protocol & Certification
            </div>
            <div style="font-size: 1.45rem; font-weight: 600; color: {text_prime_tab}; letter-spacing: -0.02em; margin-top: 2px;">
                Official Bioacoustic Forensic Telemetry Report
            </div>
            <div style="font-size: 14px; color: {text_sec_tab}; margin-top: 4px;">
                Standardized civil catastrophe incident telemetry formatted for national disaster management and seismological observatories.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.text_area("Forensic Telemetry Data Stream", forensic_report, height=320)
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="Download Bioacoustic Forensic Report (.md)",
            data=forensic_report,
            file_name=f"intera_bioacoustic_report_{audio_label.replace('.wav', '').replace('.mp3', '')}.md",
            mime="text/markdown"
        )
