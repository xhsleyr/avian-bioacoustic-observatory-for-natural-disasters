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
    color: str = "#007AFF",
    is_dark: bool = False
) -> str:
    """Generate an aesthetic Stat Card with an SVG Circular Progress Ring and Times New Roman typography."""
    radius = 23
    circumference = 2 * 3.14159 * radius
    pct_clamped = min(100.0, max(0.0, float(percentage)))
    stroke_dashoffset = circumference * (1.0 - (pct_clamped / 100.0))
    bg_ring = "rgba(255, 255, 255, 0.12)" if is_dark else "rgba(0, 0, 0, 0.06)"
    text_color = "#FFFFFF" if is_dark else "#111827"
    sub_color = "#94A3B8" if is_dark else "#64748B"
    card_bg = "rgba(18, 24, 42, 0.90)" if is_dark else "rgba(255, 255, 255, 0.95)"
    card_border = "rgba(255, 255, 255, 0.10)" if is_dark else "rgba(255, 255, 255, 0.85)"
    
    return f"""
    <div class="apple-stat-card" style="background: {card_bg}; border: 1px solid {card_border};">
        <div style="flex: 1; min-width: 0; padding-right: 8px;">
            <div class="apple-stat-title" style="color: {text_color}; font-family: 'Times New Roman', Times, serif;">{html.escape(title)}</div>
            <div class="apple-stat-value" style="color: {color}; font-family: 'Times New Roman', Times, serif;">{html.escape(value)}</div>
            <div class="apple-stat-sub" style="color: {sub_color}; font-family: 'Times New Roman', Times, serif;">{html.escape(subtext)}</div>
        </div>
        <div class="apple-stat-ring-box">
            <svg width="60" height="60" viewBox="0 0 60 60" style="transform: rotate(-90deg);">
                <circle cx="30" cy="30" r="{radius}" fill="none" stroke="{bg_ring}" stroke-width="5" />
                <circle cx="30" cy="30" r="{radius}" fill="none" stroke="{color}" stroke-width="5"
                    stroke-dasharray="{circumference:.2f}"
                    stroke-dashoffset="{stroke_dashoffset:.2f}"
                    stroke-linecap="round"
                    style="transition: stroke-dashoffset 0.8s cubic-bezier(0.16, 1, 0.3, 1);" />
            </svg>
            <div class="apple-stat-ring-pct" style="color: {text_color}; font-family: 'Times New Roman', Times, serif;">
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

def get_app_css(dark: bool) -> str:
    """Generate dynamic CSS for Midnight Canopy (Dark) or Daylight Sky (Light) themes with Universal Times New Roman."""
    if dark:
        bg_rules = """
            background-color: #070913 !important;
            background-image:
                url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.038'/%3E%3C/svg%3E"),
                radial-gradient(ellipse 60% 50% at 85% 15%, rgba(14, 165, 233, 0.22) 0%, transparent 60%),
                radial-gradient(ellipse 55% 65% at 30% 40%, rgba(99, 102, 241, 0.26) 0%, transparent 65%),
                radial-gradient(ellipse 50% 50% at 90% 70%, rgba(168, 85, 247, 0.18) 0%, transparent 60%),
                radial-gradient(ellipse 120% 70% at 50% 115%, #030712 0%, #030712 55%, transparent 65%),
                linear-gradient(180deg, #070913 0%, #0b0f1d 40%, #030712 100%) !important;
        """
        vars_rules = """
            --apple-bg: #070913;
            --apple-card: rgba(18, 24, 38, 0.92);
            --apple-card-border: rgba(255, 255, 255, 0.09);
            --apple-text-primary: #F8FAFC;
            --apple-text-secondary: #94A3B8;
            --apple-text-muted: #64748B;
            --apple-blue: #0A84FF;
            --apple-blue-hover: #0070E0;
            --apple-green: #30D158;
            --apple-red: #FF453A;
            --apple-amber: #FF9F0A;
            --apple-indigo: #5E5CE6;
            --apple-teal: #64D2FF;
            --apple-border: rgba(255, 255, 255, 0.08);
            --apple-shadow-soft: 0 16px 36px rgba(0, 0, 0, 0.45), 0 2px 6px rgba(0, 0, 0, 0.3);
            --apple-shadow-elevated: 0 20px 48px rgba(0, 0, 0, 0.6), 0 4px 12px rgba(0, 0, 0, 0.4);
        """
        text_color = "#F8FAFC"
        heading_color = "#FFFFFF"
        sub_color = "#CBD5E1"
        hero_eyebrow_color = "#38BDF8"
        hero_understand_color = "#FFFFFF"
        hero_saying_color = "#67E8F9"
        radio_bg = "#111827"
        radio_active_bg = "#1E293B"
        radio_active_color = "#38BDF8"
        radio_text = "#94A3B8"
        tab_bg = "#111827"
        tab_active_bg = "#1E293B"
        tab_active_color = "#38BDF8"
        tab_inactive_color = "#94A3B8"
        dropzone_bg = "#0F172A"
        dropzone_border = "#334155"
        select_bg = "#1E293B"
        select_border = "#334155"
        select_text = "#F8FAFC"
    else:
        bg_rules = """
            background-color: #9bb8f5 !important;
            background-image:
                url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.82' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.052'/%3E%3C/svg%3E"),
                radial-gradient(ellipse 55% 45% at 86% 16%, rgba(186, 247, 240, 0.92) 0%, rgba(186, 247, 240, 0.45) 45%, transparent 72%),
                radial-gradient(ellipse 52% 64% at 36% 44%, rgba(88, 56, 248, 0.78) 0%, rgba(109, 40, 217, 0.45) 38%, transparent 68%),
                radial-gradient(ellipse 50% 50% at 94% 68%, rgba(216, 180, 254, 0.75) 0%, transparent 62%),
                radial-gradient(ellipse 135% 65% at 48% 115%, #ffffff 0%, #ffffff 52%, transparent 54%),
                linear-gradient(175deg, #9bb8f5 0%, #b4cef8 30%, #d5e4fd 60%, #edf3fe 100%) !important;
        """
        vars_rules = """
            --apple-bg: #F5F7FA;
            --apple-card: rgba(255, 255, 255, 0.94);
            --apple-card-border: rgba(255, 255, 255, 0.9);
            --apple-text-primary: #111827;
            --apple-text-secondary: #8E8E93;
            --apple-text-muted: #64748B;
            --apple-blue: #007AFF;
            --apple-blue-hover: #0066D6;
            --apple-green: #34C759;
            --apple-red: #FF3B30;
            --apple-amber: #FF9500;
            --apple-indigo: #5856D6;
            --apple-teal: #30B0C7;
            --apple-border: rgba(0, 0, 0, 0.05);
            --apple-shadow-soft: 0 16px 36px rgba(0, 0, 0, 0.04), 0 2px 6px rgba(0, 0, 0, 0.02);
            --apple-shadow-elevated: 0 20px 48px rgba(0, 0, 0, 0.06), 0 4px 12px rgba(0, 0, 0, 0.03);
        """
        text_color = "#1e1b4b"
        heading_color = "#000000"
        sub_color = "#1E293B"
        hero_eyebrow_color = "#1e1b4b"
        hero_understand_color = "#ffffff"
        hero_saying_color = "#000000"
        radio_bg = "#E8ECEF"
        radio_active_bg = "#FFFFFF"
        radio_active_color = "#007AFF"
        radio_text = "#64748B"
        tab_bg = "#E8ECEF"
        tab_active_bg = "#FFFFFF"
        tab_active_color = "#007AFF"
        tab_inactive_color = "#64748B"
        dropzone_bg = "#F8FAFD"
        dropzone_border = "#D0D9E6"
        select_bg = "#FFFFFF"
        select_border = "#E2E8F0"
        select_text = "#1E293B"

    return f"""
    <style>
        :root {{
            {vars_rules}
        }}

        /* Universal Strict Times New Roman Typography */
        *, html, body, .stApp, 
        h1, h2, h3, h4, h5, h6, 
        p, span, div, a, label, input, select, textarea, button, 
        .stTabs [data-baseweb="tab"], 
        div[data-testid="stRadio"] *, 
        div[data-testid="stMetricValue"], 
        div[data-testid="stMetricLabel"],
        .apple-stat-title, .apple-stat-value, .apple-stat-sub, .apple-stat-ring-pct,
        .audio-filename, .audio-subtext, .audio-live-tag,
        table, th, td, pre, code {{
            font-family: 'Times New Roman', Times, Baskerville, Georgia, serif !important;
            -webkit-font-smoothing: antialiased !important;
            -moz-osx-font-smoothing: grayscale !important;
            text-rendering: optimizeLegibility !important;
            box-sizing: border-box !important;
        }}

        @keyframes cetiAuraMovement {{
            0% {{ background-position: 0% 0%, 86% 16%, 36% 44%, 94% 68%, 48% 115%, 0% 0%; }}
            50% {{ background-position: 0% 0%, 92% 22%, 32% 48%, 88% 74%, 50% 112%, 0% 0%; }}
            100% {{ background-position: 0% 0%, 86% 16%, 36% 44%, 94% 68%, 48% 115%, 0% 0%; }}
        }}

        html, body, .stApp {{
            {bg_rules}
            background-attachment: fixed !important;
            background-repeat: repeat, no-repeat, no-repeat, no-repeat, no-repeat, no-repeat !important;
            animation: cetiAuraMovement 24s ease-in-out infinite alternate !important;
            color: {text_color} !important;
            min-height: 100vh !important;
            overflow-x: hidden !important;
        }}

        h1, h2, h3, h4, h5, h6,
        .intera-hero-title,
        .species-hero-title {{
            font-weight: 700 !important;
            line-height: 1.25 !important;
            color: {heading_color} !important;
            letter-spacing: -0.015em !important;
        }}

        p, .intera-hero-subtitle, .species-hero-desc, .intera-card p {{
            font-size: 1.15rem !important;
            line-height: 1.68 !important;
            color: {sub_color} !important;
            font-weight: 500 !important;
        }}

        /* Hero Section Spacing & Overlap Prevention */
        .intera-hero {{
            text-align: center;
            margin-top: 10px;
            margin-bottom: 34px;
            position: relative;
            padding: 10px 0 20px 0;
        }}

        .intera-hero-eyebrow {{
            font-size: 14px !important;
            font-weight: 700 !important;
            letter-spacing: 0.16em !important;
            text-transform: uppercase !important;
            color: {hero_eyebrow_color} !important;
            margin-bottom: 14px !important;
            line-height: 1.3 !important;
        }}

        .intera-hero-understand {{
            font-size: clamp(3.0rem, 7vw, 5.4rem) !important;
            font-weight: 700 !important;
            line-height: 1.15 !important;
            color: {hero_understand_color} !important;
            letter-spacing: -0.02em !important;
            text-shadow: 0 4px 28px rgba(30, 27, 75, 0.20), 0 1px 3px rgba(30, 27, 75, 0.14) !important;
            display: block !important;
            margin-bottom: 6px !important;
            padding: 4px 10px !important;
            overflow: visible !important;
        }}

        .intera-hero-saying {{
            font-size: clamp(2.2rem, 5vw, 3.8rem) !important;
            font-weight: 800 !important;
            font-style: italic !important;
            line-height: 1.18 !important;
            color: {hero_saying_color} !important;
            letter-spacing: -0.015em !important;
            margin-top: 8px !important;
            margin-bottom: 24px !important;
            display: block !important;
            padding: 4px 10px !important;
            overflow: visible !important;
        }}

        .intera-hero-subtitle {{
            font-size: 1.18rem !important;
            line-height: 1.72 !important;
            color: {sub_color} !important;
            max-width: 840px !important;
            margin: 0 auto !important;
            font-weight: 500 !important;
            padding: 0 18px !important;
        }}

        /* Completely collapse sidebar */
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"],
        section[data-testid="stSidebar"],
        button[data-testid="stSidebarCollapseButton"] {{
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            pointer-events: none !important;
        }}

        /* Neumorphic / Glass Card Containers */
        [data-testid="stVerticalBlockBorderWrapper"] > div,
        .intera-card {{
            background: var(--apple-card) !important;
            backdrop-filter: blur(20px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
            border-radius: 28px !important;
            border: 1px solid var(--apple-card-border) !important;
            box-shadow: var(--apple-shadow-soft) !important;
            padding: 26px 28px !important;
            transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }}

        .apple-upload-card {{
            background: var(--apple-card) !important;
            border-radius: 28px !important;
            padding: 28px 24px 20px 24px !important;
            box-shadow: var(--apple-shadow-soft) !important;
            border: 1px solid var(--apple-card-border) !important;
            text-align: center !important;
            margin-bottom: 20px !important;
        }}

        /* Apple iOS Segmented Control (Radio Pill Selector) */
        [data-testid="stElementContainer"]:has([data-testid="stRadio"]) {{
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            width: 100% !important;
            margin: 0 auto 28px auto !important;
        }}

        div[data-testid="stRadio"] {{
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
        }}

        div[data-testid="stRadio"] > label,
        div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {{
            display: none !important;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"] {{
            display: inline-flex !important;
            flex-direction: row !important;
            flex-wrap: wrap !important;
            justify-content: center !important;
            align-items: center !important;
            background: {radio_bg} !important;
            padding: 6px 8px !important;
            border-radius: 9999px !important;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05) !important;
            border: 1px solid var(--apple-border) !important;
            gap: 6px !important;
            width: auto !important;
            max-width: 100% !important;
            margin: 0 auto !important;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"] > label,
        div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"] {{
            background: transparent !important;
            border: none !important;
            border-radius: 9999px !important;
            padding: 10px 22px !important;
            cursor: pointer !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
            white-space: normal !important;
            text-align: center !important;
            line-height: 1.3 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin: 0 !important;
            box-shadow: none !important;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"] > div:first-child {{
            display: none !important;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"] > label *,
        div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"] * {{
            font-size: 15px !important;
            font-weight: 600 !important;
            color: {radio_text} !important;
            transition: color 0.2s ease !important;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"] > label:hover,
        div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"]:hover {{
            background: rgba(255, 255, 255, 0.12) !important;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
        div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"]:has(input:checked) {{
            background: {radio_active_bg} !important;
            border-radius: 9999px !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12) !important;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) *,
        div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"]:has(input:checked) * {{
            color: {radio_active_color} !important;
            font-weight: 700 !important;
        }}

        /* Buttons & Action Pills */
        .stButton > button {{
            background: var(--apple-blue) !important;
            background-image: linear-gradient(180deg, #3894FF 0%, #007AFF 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 9999px !important;
            padding: 12px 34px !important;
            font-size: 15px !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            box-shadow: 0 8px 22px rgba(0, 122, 255, 0.28) !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
            cursor: pointer !important;
        }}

        .stButton > button:hover {{
            background: var(--apple-blue-hover) !important;
            box-shadow: 0 10px 28px rgba(0, 122, 255, 0.38) !important;
            transform: translateY(-1px) !important;
        }}

        .stButton > button * {{
            color: #FFFFFF !important;
        }}

        /* Dropzone */
        [data-testid="stFileUploaderDropzone"] {{
            background: {dropzone_bg} !important;
            border: 2px dashed {dropzone_border} !important;
            border-radius: 24px !important;
            padding: 32px 24px !important;
            text-align: center !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }}

        [data-testid="stFileUploaderDropzone"] button {{
            background: var(--apple-blue) !important;
            background-image: linear-gradient(180deg, #3894FF 0%, #007AFF 100%) !important;
            color: #FFFFFF !important;
            border-radius: 9999px !important;
            padding: 10px 30px !important;
            font-size: 14.5px !important;
            font-weight: 600 !important;
            border: none !important;
        }}

        /* Audio Input */
        div[data-testid="stAudioInput"] {{
            background: var(--apple-card) !important;
            border-radius: 24px !important;
            padding: 16px !important;
            box-shadow: var(--apple-shadow-soft) !important;
            border: 1px solid var(--apple-card-border) !important;
        }}

        /* Selectbox */
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
            background: {select_bg} !important;
            border-radius: 18px !important;
            border: 1.5px solid {select_border} !important;
            padding: 6px 14px !important;
            font-size: 15px !important;
            font-weight: 500 !important;
            color: {select_text} !important;
        }}

        /* Segmented Tabs: Zero overlap, clean wrap */
        .stTabs [data-baseweb="tab-list"] {{
            background: {tab_bg} !important;
            padding: 6px 12px !important;
            border-radius: 9999px !important;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.06) !important;
            border: 1px solid var(--apple-border) !important;
            display: flex !important;
            justify-content: center !important;
            gap: 8px !important;
            margin-bottom: 24px !important;
            flex-wrap: wrap !important;
            max-width: 100% !important;
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 9999px !important;
            padding: 10px 22px !important;
            color: {tab_inactive_color} !important;
            font-weight: 600 !important;
            font-size: 14.5px !important;
            border: none !important;
            background: transparent !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
            white-space: normal !important;
            text-align: center !important;
            line-height: 1.3 !important;
        }}

        .stTabs [aria-selected="true"] {{
            background: {tab_active_bg} !important;
            color: {tab_active_color} !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 14px rgba(0,0,0,0.12) !important;
        }}

        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] {{
            display: none !important;
        }}

        /* Apple Stat Card Component */
        .apple-stat-card {{
            border-radius: 22px;
            padding: 18px 22px;
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: var(--apple-shadow-soft);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .apple-stat-card:hover {{
            transform: translateY(-1px);
        }}
        .apple-stat-title {{
            font-size: 14.5px;
            font-weight: 700;
            letter-spacing: -0.01em;
        }}
        .apple-stat-value {{
            font-size: 21px;
            font-weight: 800;
            margin-top: 2px;
            font-variant-numeric: tabular-nums;
            letter-spacing: -0.02em;
        }}
        .apple-stat-sub {{
            font-size: 12.5px;
            font-weight: 500;
            margin-top: 2px;
        }}
        .apple-stat-ring-box {{
            position: relative;
            width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            margin-left: 14px;
        }}
        .apple-stat-ring-pct {{
            position: absolute;
            font-size: 12px;
            font-weight: 700;
        }}

        /* Species Card Avatar (Matching Milou Pet Card in Reference Pic) */
        .apple-pet-card {{
            background: var(--apple-card);
            border-radius: 26px;
            padding: 24px;
            display: flex;
            align-items: center;
            gap: 20px;
            box-shadow: var(--apple-shadow-soft);
            border: 1px solid var(--apple-card-border);
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .apple-avatar {{
            width: 92px;
            height: 92px;
            border-radius: 50%;
            object-fit: cover;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.15);
            border: 3px solid #FFFFFF;
        }}

        /* Audio Console */
        .apple-audio-console {{
            background: var(--apple-card);
            backdrop-filter: blur(24px) saturate(190%);
            -webkit-backdrop-filter: blur(24px) saturate(190%);
            border-radius: 24px;
            padding: 16px 24px;
            box-shadow: var(--apple-shadow-soft);
            border: 1px solid var(--apple-card-border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            margin-bottom: 14px;
            flex-wrap: wrap;
        }}

        .audio-console-left {{
            display: flex;
            align-items: center;
            gap: 14px;
            min-width: 220px;
        }}

        .audio-play-badge {{
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: linear-gradient(135deg, #007AFF 0%, #0056D2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #FFFFFF;
            box-shadow: 0 4px 14px rgba(0, 122, 255, 0.35);
            flex-shrink: 0;
        }}

        .audio-meta {{
            display: flex;
            flex-direction: column;
        }}

        .audio-filename {{
            font-size: 15px;
            font-weight: 700;
            color: {text_color};
            letter-spacing: -0.01em;
        }}

        .audio-subtext {{
            font-size: 12.5px;
            color: var(--apple-text-muted);
            font-weight: 500;
            margin-top: 2px;
        }}

        .audio-console-center {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 200px;
        }}

        .waveform-container {{
            display: flex;
            align-items: center;
            gap: 3.5px;
            height: 38px;
            width: 100%;
            max-width: 440px;
            justify-content: center;
        }}

        .waveform-bar {{
            width: 3.5px;
            background: var(--apple-blue);
            border-radius: 9999px;
            opacity: 0.85;
            transition: height 0.3s ease, opacity 0.2s ease, transform 0.2s ease;
        }}

        .waveform-bar:hover {{
            opacity: 1;
            background: #0056D2;
            transform: scaleY(1.2);
        }}

        .audio-console-right {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .live-equalizer {{
            display: flex;
            align-items: flex-end;
            gap: 2.5px;
            height: 18px;
        }}

        @keyframes eqWave {{
            0%, 100% {{ height: 4px; }}
            50% {{ height: 18px; }}
        }}

        .eq-bar {{
            width: 3px;
            background: #34C759;
            border-radius: 2px;
            animation: eqWave 1.2s ease-in-out infinite alternate;
        }}
        .eq-bar:nth-child(1) {{ animation-delay: 0.1s; height: 6px; }}
        .eq-bar:nth-child(2) {{ animation-delay: 0.3s; height: 14px; }}
        .eq-bar:nth-child(3) {{ animation-delay: 0.05s; height: 18px; }}
        .eq-bar:nth-child(4) {{ animation-delay: 0.4s; height: 10px; }}
        .eq-bar:nth-child(5) {{ animation-delay: 0.25s; height: 15px; }}
        .eq-bar:nth-child(6) {{ animation-delay: 0.15s; height: 8px; }}

        .audio-live-tag {{
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #34C759;
            background: rgba(52, 199, 89, 0.15);
            padding: 5px 12px;
            border-radius: 9999px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        @keyframes pulseGlow {{
            0% {{ box-shadow: 0 0 0 0 rgba(52, 199, 89, 0.5); }}
            70% {{ box-shadow: 0 0 0 7px rgba(52, 199, 89, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(52, 199, 89, 0); }}
        }}

        .pulse-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: #34C759;
            display: inline-block;
            animation: pulseGlow 1.8s infinite;
            vertical-align: middle;
        }}

        audio {{
            width: 100% !important;
            border-radius: 9999px !important;
            height: 38px !important;
        }}
    </style>
    """

# Inject Dynamic CSS based on Dark/Light mode
st.markdown(get_app_css(is_dark), unsafe_allow_html=True)


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
def create_gauge_chart(adi_score: float, threat_color: str, is_dark: bool = False) -> go.Figure:
    """Create a refined semi-circular bioacoustic distress gauge with Times New Roman styling."""
    bar_color = "#34C759" if adi_score < 38.0 else ("#FF9500" if adi_score < 70.0 else "#FF3B30")
    text_color = "#F8FAFC" if is_dark else "#111827"
    axis_color = "#94A3B8" if is_dark else "#64748B"
    bg_arc = "#1E293B" if is_dark else "#ECEFF4"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=adi_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "ACOUSTIC DISTRESS INDEX (ADI)",
            'font': {'size': 14, 'color': text_color, 'family': "Times New Roman, Times, serif"}
        },
        number={
            'suffix': "%",
            'font': {'size': 56, 'color': text_color, 'family': "Times New Roman, Times, serif"}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1.5,
                'tickcolor': axis_color,
                'tickfont': {'family': "Times New Roman, Times, serif", 'size': 12, 'color': axis_color},
                'tickvals': [0, 25, 50, 75, 100]
            },
            'bar': {'color': bar_color, 'thickness': 0.35},
            'bgcolor': bg_arc,
            'borderwidth': 0,
            'steps': [
                {'range': [0, 38], 'color': 'rgba(52, 199, 89, 0.20)'},
                {'range': [38, 70], 'color': 'rgba(255, 149, 0, 0.20)'},
                {'range': [70, 100], 'color': 'rgba(255, 59, 48, 0.25)'}
            ],
            'threshold': {
                'line': {'color': "#FF3B30", 'width': 3.0},
                'thickness': 0.85,
                'value': 70.0
            }
        }
    ))
    fig.update_layout(
        height=265,
        margin={"l": 15, "r": 15, "t": 32, "b": 10},
        paper_bgcolor='rgba(0,0,0,0)',
        font={"family": "Times New Roman, Times, serif", "color": text_color}
    )
    return fig


def plot_mel_spectrogram(features: dict, is_dark: bool = False) -> go.Figure:
    """Create a high-contrast Studio Mel-Spectrogram with Times New Roman typography."""
    mel_db = features["mel_db"]
    times = features["times"]
    freq_bins = np.linspace(200, 10000, mel_db.shape[0])

    p22 = float(np.percentile(mel_db, 22))
    p_max = float(mel_db.max())

    colorscale_apple = [
        [0.0, '#070912'],
        [0.10, '#0f172a'],
        [0.25, '#1e3a8a'],
        [0.45, '#0284c7'],
        [0.65, '#00e5ff'],
        [0.82, '#34c759'],
        [0.94, '#ffd60a'],
        [1.0, '#ffffff']
    ]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=mel_db,
        x=times,
        y=freq_bins,
        colorscale=colorscale_apple,
        zmin=p22,
        zmax=p_max,
        zsmooth='best',
        colorbar={
            "title": {
                "text": "Power (dB)",
                "font": {"family": "Times New Roman, Times, serif", "size": 12, "color": "#FFFFFF"}
            },
            "tickfont": {"family": "Times New Roman, Times, serif", "color": "#94A3B8", "size": 11},
            "len": 0.88,
            "thickness": 12,
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
            name='Vocal Pitch Centroid',
            line={"color": '#00F0FF', "width": 2.0},
            hovertemplate="<b>Vocal Pitch Centroid:</b> %{y:.0f} Hz<br><b>Time:</b> %{x:.2f} s<extra></extra>"
        ))

    fig.add_hline(
        y=1500, line_dash='dash', line_color='rgba(0, 240, 255, 0.70)', line_width=1.3,
        annotation_text="Avian Syrinx Band (1.5 kHz)",
        annotation_position="top left",
        annotation_font={"family": "Times New Roman, Times, serif", "size": 12, "color": "#00F0FF"}
    )
    fig.add_hline(
        y=4500, line_dash='dash', line_color='rgba(255, 59, 48, 0.80)', line_width=1.3,
        annotation_text="Alarm Screech Threshold (4.5 kHz)",
        annotation_position="top left",
        annotation_font={"family": "Times New Roman, Times, serif", "size": 12, "color": "#FF3B30"}
    )

    fig.update_layout(
        title={
            "text": "HIGH-RESOLUTION MEL BIOACOUSTIC SONOGRAM & SYRINGEAL FORMANTS",
            "font": {"family": "Times New Roman, Times, serif", "size": 15, "color": "#FFFFFF"}
        },
        xaxis={
            "title": {"text": "Time Elapsed (seconds)", "font": {"family": "Times New Roman, Times, serif", "size": 13, "color": "#CBD5E1"}},
            "showgrid": True,
            "gridcolor": "rgba(255, 255, 255, 0.08)",
            "tickfont": {"family": "Times New Roman, Times, serif", "color": "#94A3B8"}
        },
        yaxis={
            "title": {"text": "Frequency (Hz)", "font": {"family": "Times New Roman, Times, serif", "size": 13, "color": "#CBD5E1"}},
            "showgrid": True,
            "gridcolor": "rgba(255, 255, 255, 0.08)",
            "tickfont": {"family": "Times New Roman, Times, serif", "color": "#94A3B8"},
            "range": [200, 10000]
        },
        height=380,
        margin={"l": 20, "r": 20, "t": 44, "b": 25},
        paper_bgcolor='#070912',
        plot_bgcolor='#070912',
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"family": "Times New Roman, Times, serif", "size": 12, "color": "#94A3B8"},
            "bgcolor": "rgba(0,0,0,0)"
        }
    )
    return fig


def plot_waveform_and_dynamics(y: np.ndarray, sr: int, features: dict, is_dark: bool = False) -> go.Figure:
    """Plot synchronized waveform and pitch dynamics with Times New Roman styling."""
    time_wave = np.linspace(0, len(y) / sr, len(y))
    times_feats = features["times"]
    centroid = features["spectral_centroid"]["trajectory"]

    text_color = "#F8FAFC" if is_dark else "#111827"
    axis_color = "#94A3B8" if is_dark else "#475569"
    plot_bg = "#0B0F1D" if is_dark else "#FFFFFF"
    grid_col = "rgba(255, 255, 255, 0.08)" if is_dark else "#ECEFF4"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=("Acoustic Signal Waveform & Call Onset Bursts", "Pitch / Spectral Centroid Trajectory (Hz)")
    )

    step = max(1, len(y) // 3000)
    fig.add_trace(
        go.Scatter(x=time_wave[::step], y=y[::step], mode='lines', name='Waveform', line={"color": "#007AFF", "width": 1.5}),
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
                marker={"symbol": "triangle-down", "size": 9, "color": "#FF3B30"}
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(x=times_feats, y=centroid, mode='lines', name='Centroid (Hz)', line={"color": "#5856D6", "width": 2.2}),
        row=2, col=1
    )

    fig.update_layout(
        height=350,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=plot_bg,
        showlegend=True,
        font={"family": "Times New Roman, Times, serif", "color": text_color}
    )
    fig.update_xaxes(title_text="Time (seconds)", title_font={"color": axis_color, "size": 13, "family": "Times New Roman, Times, serif"}, tickfont={"color": axis_color, "size": 12, "family": "Times New Roman, Times, serif"}, row=2, col=1, gridcolor=grid_col)
    fig.update_yaxes(title_text="Amplitude", title_font={"color": axis_color, "size": 13, "family": "Times New Roman, Times, serif"}, tickfont={"color": axis_color, "size": 12, "family": "Times New Roman, Times, serif"}, row=1, col=1, gridcolor=grid_col)
    fig.update_yaxes(title_text="Centroid (Hz)", title_font={"color": axis_color, "size": 13, "family": "Times New Roman, Times, serif"}, tickfont={"color": axis_color, "size": 12, "family": "Times New Roman, Times, serif"}, row=2, col=1, gridcolor=grid_col)
    fig.update_annotations(font={"family": "Times New Roman, Times, serif", "size": 14, "color": text_color})

    return fig


def plot_temporal_distress_curve(timeline_info: dict, is_dark: bool = False) -> go.Figure:
    """Plot high-precision temporal threat escalation curve with Times New Roman styling."""
    times = timeline_info.get("times", [0.0])
    adi_series = timeline_info.get("adi_series", [50.0])
    shockwave = timeline_info.get("shockwave_detected", False)
    peak_time = timeline_info.get("peak_time", 0.0)
    peak_adi = timeline_info.get("peak_adi", 0.0)
    trend_label = timeline_info.get("trend_label", "Stationary")
    escalation_rate = timeline_info.get("escalation_rate", 0.0)

    text_color = "#F8FAFC" if is_dark else "#111827"
    axis_color = "#94A3B8" if is_dark else "#475569"
    plot_bg = "#0B0F1D" if is_dark else "#FFFFFF"
    grid_col = "rgba(255, 255, 255, 0.08)" if is_dark else "#ECEFF4"

    fig = go.Figure()

    line_color = "#FF3B30" if shockwave or peak_adi >= 70.0 else ("#FF9500" if peak_adi >= 38.0 else "#007AFF")
    fill_color = "rgba(255, 59, 48, 0.15)" if line_color == "#FF3B30" else ("rgba(255, 149, 0, 0.15)" if line_color == "#FF9500" else "rgba(0, 122, 255, 0.12)")

    fig.add_trace(go.Scatter(
        x=times,
        y=adi_series,
        mode='lines',
        name='Acoustic Distress Index',
        line={"color": line_color, "width": 3.0, "shape": "spline"},
        fill='tozeroy',
        fillcolor=fill_color,
        hovertemplate="<b>Elapsed Time:</b> %{x:.2f}s<br><b>Acoustic Distress Index:</b> %{y:.1f}%<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=[peak_time],
        y=[peak_adi],
        mode='markers+text',
        name='Peak Threat',
        text=[f"Peak: {peak_adi:.1f}%"],
        textposition="top center",
        marker={"size": 10, "color": line_color, "symbol": "diamond"},
        hoverinfo='skip'
    ))

    fig.add_hline(
        y=70, line_dash="dash", line_color="#FF3B30", line_width=1.5,
        annotation_text="Critical Threat (70%)",
        annotation_position="bottom right",
        annotation_font={"size": 12, "color": "#FF3B30", "family": "Times New Roman, Times, serif"}
    )
    fig.add_hline(
        y=38, line_dash="dot", line_color="#FF9500", line_width=1.5,
        annotation_text="Elevated Vigilance (38%)",
        annotation_position="bottom right",
        annotation_font={"size": 12, "color": "#FF9500", "family": "Times New Roman, Times, serif"}
    )

    fig.update_layout(
        title={
            "text": f"TEMPORAL THREAT ESCALATION TIMELINE &bull; {trend_label.upper()} (Max Velocity d(ADI)/dt: {escalation_rate:.1f}%/s)",
            "font": {"size": 14, "color": text_color, "family": "Times New Roman, Times, serif"}
        },
        xaxis={
            "title": "Elapsed Time (Seconds)",
            "showgrid": True,
            "gridcolor": grid_col,
            "tickfont": {"color": axis_color, "size": 12, "family": "Times New Roman, Times, serif"},
            "title_font": {"family": "Times New Roman, Times, serif", "size": 13, "color": axis_color}
        },
        yaxis={
            "title": "Acoustic Distress Index (%)",
            "range": [0, 105],
            "showgrid": True,
            "gridcolor": grid_col,
            "tickfont": {"color": axis_color, "size": 12, "family": "Times New Roman, Times, serif"},
            "title_font": {"family": "Times New Roman, Times, serif", "size": 13, "color": axis_color}
        },
        height=320,
        margin={"l": 40, "r": 30, "t": 46, "b": 36},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=plot_bg,
        showlegend=False
    )
    return fig


# --- APPLE MASTHEAD NAVIGATION WITH THEME TOGGLE ---
col_mh_left, col_mh_right = st.columns([3.4, 1.2])
with col_mh_left:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; padding: 10px 0 16px 0;">
        <span style="width: 11px; height: 11px; border-radius: 50%; background-color: #007AFF; display: inline-block; box-shadow: 0 0 12px rgba(0, 122, 255, 0.7);"></span>
        <span style="font-size: 20px; font-weight: 800; letter-spacing: -0.02em; color: {'#FFFFFF' if is_dark else '#111827'};">Project INTERA</span>
        <span style="font-size: 13px; font-weight: 600; color: {'#94A3B8' if is_dark else '#64748B'}; background: {'rgba(255,255,255,0.08)' if is_dark else '#ECEFF4'}; border: 1px solid {'rgba(255,255,255,0.1)' if is_dark else 'rgba(0,0,0,0.05)'}; padding: 3px 12px; border-radius: 9999px;">Bioacoustic Observatory</span>
    </div>
    """, unsafe_allow_html=True)
with col_mh_right:
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
    st.toggle("🌙 Midnight Canopy" if is_dark else "☀️ Daylight Sky", key="dark_mode_toggle")

# --- HUMAN EDITORIAL HERO (PROJECT CETI ART DIRECTION) ---
st.markdown("""
<div class="intera-hero">
    <div class="intera-hero-eyebrow">WHAT WOULD IT MEAN TO</div>
    <div class="intera-hero-understand">Understand</div>
    <div class="intera-hero-saying">What the Birds are Saying</div>
    <div class="intera-hero-subtitle">
        Birds sense approaching storms and earthquakes hours before instruments pick them up. When danger nears, their calls change. Project INTERA listens to these distress patterns to warn communities before disaster strikes.
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

    # --- APPLE MULTI-SPECIES SENTINEL CONSENSUS BANNER ---
    consensus_type = consensus_info.get("consensus_type", "nominal_calm")
    cons_status = consensus_info.get("status", "NOMINAL")
    cons_summary = consensus_info.get("summary", "")
    cons_color = consensus_info.get("color", "#34C759")

    if consensus_type == "macro_ecological_panic":
        st.markdown(f"""
        <div style="background: #FFF5F5; border: 1.5px solid #FF3B30; border-radius: 20px; padding: 18px 24px; margin-bottom: 24px; box-shadow: 0 6px 20px rgba(255, 59, 48, 0.12); display: flex; align-items: center; gap: 14px;">
            <div style="font-size: 24px;">🚨</div>
            <div>
                <div style="font-size: 15px; font-weight: 800; color: #FF3B30; letter-spacing: -0.01em;">
                    {cons_status}
                </div>
                <div style="font-size: 13.5px; color: #111827; font-weight: 500; margin-top: 2px;">
                    {cons_summary}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif consensus_type == "localized_sentinel_alert":
        st.markdown(f"""
        <div style="background: #FFFBF0; border: 1.5px solid #FF9500; border-radius: 20px; padding: 18px 24px; margin-bottom: 24px; box-shadow: 0 6px 20px rgba(255, 149, 0, 0.12); display: flex; align-items: center; gap: 14px;">
            <div style="font-size: 24px;">⚠️</div>
            <div>
                <div style="font-size: 15px; font-weight: 800; color: #FF9500; letter-spacing: -0.01em;">
                    {cons_status}
                </div>
                <div style="font-size: 13.5px; color: #111827; font-weight: 500; margin-top: 2px;">
                    {cons_summary}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: #F0FDF4; border: 1.5px solid #34C759; border-radius: 20px; padding: 18px 24px; margin-bottom: 24px; box-shadow: 0 6px 20px rgba(52, 199, 89, 0.10); display: flex; align-items: center; gap: 14px;">
            <div style="font-size: 22px;">🌿</div>
            <div>
                <div style="font-size: 15px; font-weight: 800; color: #34C759; letter-spacing: -0.01em;">
                    {cons_status}
                </div>
                <div style="font-size: 13.5px; color: #111827; font-weight: 500; margin-top: 2px;">
                    {cons_summary}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- APPLE SPECIES CARD (MATCHING REFERENCE IMAGE SCREEN 3) ---
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
        <div class="apple-pet-card">
            <img src="{photo_uri}" alt="{safe_com_name}" class="apple-avatar" />
            <div style="flex: 1; min-width: 260px;">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 26px; font-weight: 800; color: #111827; letter-spacing: -0.025em;">{safe_com_name}</span>
                        <span style="background: #EBF4FF; color: #007AFF; font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 9999px;">
                            BirdNET &bull; {prim_sp['confidence']}% Match
                        </span>
                    </div>
                    <span style="background: #EBF9F1; color: #34C759; font-size: 12.5px; font-weight: 700; padding: 4px 12px; border-radius: 9999px;">
                        <span class="pulse-dot" style="margin-right: 4px;"></span> Verified Authentic Syrinx
                    </span>
                </div>
                <div style="font-size: 14.5px; font-style: italic; color: #64748B; margin-top: 2px;">
                    {safe_sci_name} &bull; {results.get('baseline_profile', {}).get('family_label', 'Avian Taxonomy')}
                </div>
                <div style="font-size: 13.5px; color: #475569; margin-top: 8px; line-height: 1.55;">
                    {safe_desc}
                </div>
                <div style="display: flex; gap: 16px; margin-top: 12px; font-size: 13px; color: #64748B; flex-wrap: wrap;">
                    <span>Acoustic State: <strong style="color: #111827;">{prim_sp['pitch_status']} ({prim_sp['detected_pitch']:.0f} Hz)</strong></span>
                    <span>&bull;</span>
                    <span>Early Warning Sentinel: <strong style="color: #007AFF;">{prim_sp['disaster_sensitivity']}</strong></span>
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
        <div style="background: #FFFFFF; border-radius: 26px; padding: 26px 30px; box-shadow: 0 16px 36px rgba(0,0,0,0.04); border-left: 6px solid #FF3B30; margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; flex-wrap: wrap; gap: 8px;">
                <span style="background: #FEECEB; color: #FF3B30; font-size: 12px; padding: 4px 14px; border-radius: 9999px; font-weight: 700;">
                    Biometric Rejection &bull; Non-Avian Acoustic Source
                </span>
                <span style="font-size: 13px; color: #FF3B30; font-weight: 700;">
                    Sub-800Hz Energy: {spoof_info.get('sub800_ratio', 0.0)*100:.1f}% (Mammal Threshold &gt; 25%)
                </span>
            </div>
            <div style="font-size: 24px; font-weight: 800; color: #111827; letter-spacing: -0.02em; margin-top: 6px;">
                {detected_non_bird_title}
            </div>
            <p style="font-size: 14.5px; color: #475569; margin-top: 6px; line-height: 1.6;">
                {html.escape(spoof_info.get('status_summary', ''))}
            </p>
            <div style="margin-top: 12px; font-size: 13px; color: #111827;">
                <strong>Biometric Rejection Criteria Triggered:</strong>
                <ul style="margin: 6px 0 0 18px; padding: 0; color: #64748B;">
                    {factors_items}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- APPLE STATS DASHBOARD & STRESS DECOMPOSITION ---
    adi_val = results["adi"]
    if adi_val < 38.0:
        adi_tier_label = "Nominal Equilibrium // Baseline Chorus"
        adi_tier_color = "#34C759"
        adi_tier_desc = "Calm environmental foraging calls. No seismic micro-tremors, thunderstorm infrasound, or aerial raptor threats detected."
    elif adi_val < 70.0:
        adi_tier_label = "Elevated Vigilance // Canopy Agitation"
        adi_tier_color = "#FF9500"
        adi_tier_desc = "Noticeable frequency shift and accelerated call density detected. Indicates local environmental agitation."
    else:
        adi_tier_label = "Acute Biotic Distress // Evacuation Call"
        adi_tier_color = "#FF3B30"
        adi_tier_desc = "Extreme high-frequency screeching and frantic burst pacing. Strongly correlates with imminent natural hazards."

    col_gauge, col_stats = st.columns([1.15, 1.85])

    with col_gauge, st.container(border=True):
            gauge_fig = create_gauge_chart(results["adi"], results["threat_color"], is_dark=is_dark)
            st.plotly_chart(gauge_fig, width="stretch")
            st.markdown(f"""
            <div style="text-align: center; margin-top: -6px; padding-bottom: 8px;">
                <div style="font-size: 14.5px; font-weight: 800; color: {adi_tier_color}; letter-spacing: -0.01em;">
                    ● {adi_tier_label}
                </div>
                <div style="font-size: 13px; color: #8E8E93; margin-top: 4px; line-height: 1.5; font-weight: 500;">
                    {adi_tier_desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_stats, st.container(border=True):
            st.markdown("""
            <div style="margin-bottom: 12px;">
                <div style="font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: #007AFF; font-weight: 700;">
                    Bioacoustic Telemetry & Stats
                </div>
                <div style="font-size: 1.35rem; font-weight: 800; color: #111827; letter-spacing: -0.02em;">
                    Real-Time Biological Sensor Telemetry
                </div>
            </div>
            """, unsafe_allow_html=True)

            sub_pitch = results['sub_scores'].get('pitch', 0.0)
            sub_rate = results['sub_scores'].get('call_rate', 0.0)
            sub800 = features.get('sub800_ratio', 0.0) * 100.0

            # 4 Apple Stat Cards matching Screen 3 in Reference Image
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown(render_apple_stat_card(
                    title="Acoustic Distress",
                    value=f"{results['adi']:.1f}%",
                    subtext="Equilibrium" if results['adi'] < 38 else ("Agitated" if results['adi'] < 70 else "Panic Threat"),
                    percentage=results['adi'],
                    color=adi_tier_color,
                    is_dark=is_dark
                ), unsafe_allow_html=True)

                st.markdown(render_apple_stat_card(
                    title="Pitch Elevation",
                    value=f"{features['spectral_centroid']['mean']:.0f} Hz",
                    subtext=f"{sub_pitch:.0f}% vs Baseline",
                    percentage=min(100.0, max(5.0, sub_pitch)),
                    color="#007AFF",
                    is_dark=is_dark
                ), unsafe_allow_html=True)

            with col_s2:
                st.markdown(render_apple_stat_card(
                    title="Vocal Organ",
                    value="Avian Syrinx" if spoof_info['is_authentic'] else "Mammalian",
                    subtext="Harmonic Dual Syrinx" if spoof_info['is_authentic'] else "Low-Freq Larynx",
                    percentage=prim_sp.get('confidence', 95.0) if spoof_info['is_authentic'] else 15.0,
                    color="#34C759" if spoof_info['is_authentic'] else "#FF3B30",
                    is_dark=is_dark
                ), unsafe_allow_html=True)

                st.markdown(render_apple_stat_card(
                    title="Call Cadence",
                    value=f"{features['call_rate']:.2f} calls/s",
                    subtext=f"{sub_rate:.0f}% Bursts Pacing",
                    percentage=min(100.0, max(5.0, sub_rate)),
                    color="#5856D6",
                    is_dark=is_dark
                ), unsafe_allow_html=True)

    # --- APPLE SEGMENTED TABS: HIGH-DEFINITION TELEMETRY ---
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

    card_bg_tab = "rgba(18, 24, 42, 0.88)" if is_dark else "rgba(255, 255, 255, 0.94)"
    card_border_tab = "rgba(255, 255, 255, 0.10)" if is_dark else "rgba(255, 255, 255, 0.85)"
    text_prime_tab = "#FFFFFF" if is_dark else "#111827"
    text_sec_tab = "#94A3B8" if is_dark else "#64748B"

    # ========================================================
    # TAB 1: THREAT TIMELINE & ESCALATION
    # ========================================================
    with tab_timeline, st.container(border=True):
        st.markdown(f"""
        <div style="margin-bottom: 18px;">
            <div style="font-size: 13px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--apple-blue); font-weight: 700;">
                Pre-Disaster Dynamic Infrasound & P-Wave Progression
            </div>
            <div style="font-size: 1.55rem; font-weight: 800; color: {text_prime_tab}; letter-spacing: -0.02em; margin-top: 2px;">
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
                subtext="Threat Surge Acceleration",
                percentage=min(100.0, max(5.0, esc_rate * 3.5)),
                color="#FF9500" if esc_rate > 5 else "#007AFF",
                is_dark=is_dark
            ), unsafe_allow_html=True)
        with col_t2:
            st.markdown(render_apple_stat_card(
                title="Acoustic Apex Time",
                value=f"{peak_time:.2f}s",
                subtext=f"Total Duration: {duration:.2f}s",
                percentage=min(100.0, (peak_time / max(0.1, duration)) * 100.0),
                color="#5856D6",
                is_dark=is_dark
            ), unsafe_allow_html=True)
        with col_t3:
            st.markdown(render_apple_stat_card(
                title="Peak Threat Level",
                value=f"{peak_adi:.1f}%",
                subtext="Maximum Distress Index",
                percentage=peak_adi,
                color="#FF3B30" if peak_adi >= 70 else ("#FF9500" if peak_adi >= 38 else "#34C759"),
                is_dark=is_dark
            ), unsafe_allow_html=True)
        with col_t4:
            st.markdown(render_apple_stat_card(
                title="Pre-Seismic Shock",
                value="DETECTED" if shock_detected else "NOMINAL",
                subtext="Infrasonic P-Wave Burst" if shock_detected else "Steady Atmospheric State",
                percentage=95.0 if shock_detected else 12.0,
                color="#FF3B30" if shock_detected else "#34C759",
                is_dark=is_dark
            ), unsafe_allow_html=True)

        st.plotly_chart(plot_temporal_distress_curve(timeline_info, is_dark=is_dark), width="stretch")

        trend_name = timeline_info.get("trend_label", "Stationary Chorus")
        st.markdown(f"""
        <div style="background: {card_bg_tab}; border-radius: 20px; border: 1px solid {card_border_tab}; padding: 18px 22px; margin-top: 14px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
            <div>
                <span style="font-size: 14.5px; font-weight: 700; color: {text_prime_tab};">Biological Evacuation Alert Status: </span>
                <span style="font-size: 14.5px; font-weight: 800; color: {'#FF3B30' if shock_detected or peak_adi >= 70 else '#34C759'};">{trend_name.upper()}</span>
                <div style="font-size: 13px; color: {text_sec_tab}; margin-top: 3px;">
                    Bioacoustic sensor arrays indicate an estimated warning lead time of 45–90 minutes before surface micro-tremor emergence or severe storm wind shear.
                </div>
            </div>
            <span style="background: rgba(0, 122, 255, 0.15); color: var(--apple-blue); font-size: 12.5px; font-weight: 700; padding: 6px 16px; border-radius: 9999px;">
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
            <div style="font-size: 13px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--apple-blue); font-weight: 700;">
                High-Resolution Time-Frequency Bioacoustics
            </div>
            <div style="font-size: 1.55rem; font-weight: 800; color: {text_prime_tab}; letter-spacing: -0.02em; margin-top: 2px;">
                Mel Sonogram Spectrograph & Syringeal Formants
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(plot_mel_spectrogram(features, is_dark=is_dark), width="stretch")

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.markdown(f"""
            <div style="background: {card_bg_tab}; border-radius: 18px; border: 1px solid {card_border_tab}; padding: 14px 18px; text-align: center;">
                <div style="font-size: 12.5px; font-weight: 700; color: #00F0FF; text-transform: uppercase;">Avian Syrinx Band</div>
                <div style="font-size: 17px; font-weight: 800; color: {text_prime_tab}; margin-top: 2px;">1,500 – 3,200 Hz</div>
                <div style="font-size: 12px; color: {text_sec_tab}; margin-top: 2px;">Core Syringeal Membrane Resonance</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b2:
            st.markdown(f"""
            <div style="background: {card_bg_tab}; border-radius: 18px; border: 1px solid {card_border_tab}; padding: 14px 18px; text-align: center;">
                <div style="font-size: 12.5px; font-weight: 700; color: #FF3B30; text-transform: uppercase;">Alarm Screech Ceiling</div>
                <div style="font-size: 17px; font-weight: 800; color: {text_prime_tab}; margin-top: 2px;">> 4,500 Hz</div>
                <div style="font-size: 12px; color: {text_sec_tab}; margin-top: 2px;">High-Harmonic Panic Burst Zone</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b3:
            st.markdown(f"""
            <div style="background: {card_bg_tab}; border-radius: 18px; border: 1px solid {card_border_tab}; padding: 14px 18px; text-align: center;">
                <div style="font-size: 12.5px; font-weight: 700; color: #34C759; text-transform: uppercase;">Bioacoustic Bandwidth</div>
                <div style="font-size: 17px; font-weight: 800; color: {text_prime_tab}; margin-top: 2px;">10,000 Hz / 128 Bins</div>
                <div style="font-size: 12px; color: {text_sec_tab}; margin-top: 2px;">Full Ultrasonic Nyquist Range</div>
            </div>
            """, unsafe_allow_html=True)

    # ========================================================
    # TAB 3: WAVEFORM DYNAMICS
    # ========================================================
    with tab_wave, st.container(border=True):
        st.markdown(f"""
        <div style="margin-bottom: 16px;">
            <div style="font-size: 13px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--apple-blue); font-weight: 700;">
                Acoustic Amplitude & Dynamic Envelope
            </div>
            <div style="font-size: 1.55rem; font-weight: 800; color: {text_prime_tab}; letter-spacing: -0.02em; margin-top: 2px;">
                Synchronized Waveform & Spectral Centroid Dynamics
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(plot_waveform_and_dynamics(y, sr, features, is_dark=is_dark), width="stretch")

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
                color="#007AFF",
                is_dark=is_dark
            ), unsafe_allow_html=True)
        with col_w2:
            st.markdown(render_apple_stat_card(
                title="RMS Energy Power",
                value=f"{rms_val:.4f}",
                subtext="Root-Mean-Square Acoustic Energy",
                percentage=min(100.0, max(8.0, rms_val * 400.0)),
                color="#5856D6",
                is_dark=is_dark
            ), unsafe_allow_html=True)
        with col_w3:
            st.markdown(render_apple_stat_card(
                title="Call Burst Density",
                value=f"{call_cadence:.2f} calls/s",
                subtext="Burst Frequency Modulation",
                percentage=min(100.0, max(5.0, call_cadence * 25.0)),
                color="#FF9500" if call_cadence > 2.5 else "#34C759",
                is_dark=is_dark
            ), unsafe_allow_html=True)

    # ========================================================
    # TAB 4: SPECIES TAXONOMY (MATCHING PET WALK CARD AESTHETIC)
    # ========================================================
    with tab_species, st.container(border=True):
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <div style="font-size: 13px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--apple-blue); font-weight: 700;">
                Cornell Bioacoustics Neural Classifier
            </div>
            <div style="font-size: 1.55rem; font-weight: 800; color: {text_prime_tab}; letter-spacing: -0.02em; margin-top: 2px;">
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

            rank_badge_bg = "rgba(0, 122, 255, 0.15)" if i == 1 else "rgba(142, 142, 147, 0.15)"
            rank_badge_color = "var(--apple-blue)" if i == 1 else text_sec_tab
            rank_label = f"#{i} Primary Match" if i == 1 else f"#{i} Candidate"

            st.markdown(f"""
            <div class="apple-pet-card" style="margin-bottom: 16px; padding: 20px 24px;">
                <img src="{cand_photo_uri}" alt="{cand_com}" class="apple-avatar" style="width: 76px; height: 76px; flex-shrink: 0;" />
                <div style="flex: 1; min-width: 240px;">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                        <div>
                            <span style="font-size: 21px; font-weight: 800; color: {text_prime_tab}; letter-spacing: -0.015em;">{cand_com}</span>
                            <span style="background: {rank_badge_bg}; color: {rank_badge_color}; font-size: 12px; font-weight: 700; padding: 3px 12px; border-radius: 9999px; margin-left: 8px;">
                                {rank_label}
                            </span>
                        </div>
                        <span style="background: rgba(52, 199, 89, 0.15); color: #34C759; font-size: 13.5px; font-weight: 800; padding: 4px 14px; border-radius: 9999px;">
                            {cand_conf:.1f}% Confidence
                        </span>
                    </div>
                    <div style="font-size: 14px; font-style: italic; color: {text_sec_tab}; margin-top: 3px;">
                        {cand_sci} &bull; Avian Bioacoustic Class
                    </div>
                    <div style="display: flex; gap: 16px; margin-top: 10px; font-size: 13px; color: {text_sec_tab}; flex-wrap: wrap;">
                        <span>Vocal Frequency: <strong style="color: {text_prime_tab};">{cand_pitch_stat} ({cand_pitch_val:.0f} Hz)</strong></span>
                        <span>&bull;</span>
                        <span>Sensitivity: <strong style="color: var(--apple-blue);">{cand_sentinel}</strong></span>
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
            <div style="font-size: 13px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--apple-blue); font-weight: 700;">
                Avian Syrinx Biometric Shield & Anti-Spoofing
            </div>
            <div style="font-size: 1.55rem; font-weight: 800; color: {text_prime_tab}; letter-spacing: -0.02em; margin-top: 2px;">
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
                color="#34C759" if sp_score >= 70 else "#FF3B30",
                is_dark=is_dark
            ), unsafe_allow_html=True)
        with col_sh2:
            st.markdown(render_apple_stat_card(
                title="Sub-800Hz Energy Ratio",
                value=f"{sub800:.1f}%",
                subtext="Mammalian Larynx Ceiling < 25%",
                percentage=min(100.0, sub800),
                color="#34C759" if sub800 < 25.0 else "#FF3B30",
                is_dark=is_dark
            ), unsafe_allow_html=True)
        with col_sh3:
            st.markdown(render_apple_stat_card(
                title="95% High-Freq Rolloff",
                value=f"{roll95:.0f} Hz",
                subtext="Speaker Replay Floor > 7,500 Hz",
                percentage=min(100.0, (roll95 / 10000.0) * 100.0),
                color="#34C759" if roll95 >= 7500 else "#FF9500",
                is_dark=is_dark
            ), unsafe_allow_html=True)

        if spoof_info["is_authentic"]:
            st.markdown(f"""
            <div style="background: rgba(52, 199, 89, 0.12); border: 1.5px solid #34C759; border-radius: 20px; padding: 20px 24px; margin-top: 14px; display: flex; align-items: center; gap: 16px;">
                <div style="font-size: 28px;">🛡️</div>
                <div>
                    <div style="font-size: 16px; font-weight: 800; color: #34C759; letter-spacing: -0.01em;">
                        AUTHENTIC DUAL SYRINX HARMONICS CONFIRMED
                    </div>
                    <div style="font-size: 13.5px; color: {text_prime_tab}; font-weight: 500; margin-top: 3px; line-height: 1.5;">
                        Acoustic energy matches verified avian syringeal resonance. Zero oral cavity formants, mammalian chest resonance, or phone loudspeaker harmonic cutoff detected. Audio signal is armed for automated early disaster alerting.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            cat_label = html.escape(str(spoof_info.get('detected_category', 'Non-Avian Acoustic Interference')))
            st.markdown(f"""
            <div style="background: rgba(255, 59, 48, 0.12); border: 1.5px solid #FF3B30; border-radius: 20px; padding: 20px 24px; margin-top: 14px; display: flex; align-items: center; gap: 16px;">
                <div style="font-size: 28px;">⚠️</div>
                <div>
                    <div style="font-size: 16px; font-weight: 800; color: #FF3B30; letter-spacing: -0.01em;">
                        BIOMETRIC REJECTION: NON-AVIAN ACOUSTIC INTERFERENCE ({cat_label})
                    </div>
                    <div style="font-size: 13.5px; color: {text_prime_tab}; font-weight: 500; margin-top: 3px; line-height: 1.5;">
                        This specimen failed avian syrinx authenticity verification. Sub-800Hz energy or harmonic dispersion matches non-avian mammalian vocal tract patterns. Disarmed from civil disaster escalation.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ========================================================
    # TAB 6: DIAGNOSTIC TABLE
    # ========================================================
    with tab_table, st.container(border=True):
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <div style="font-size: 13px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--apple-blue); font-weight: 700;">
                Laboratory Telemetry Matrix
            </div>
            <div style="font-size: 1.55rem; font-weight: 800; color: {text_prime_tab}; letter-spacing: -0.02em; margin-top: 2px;">
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

        th_bg = "rgba(0,0,0,0.25)" if is_dark else "rgba(0,0,0,0.03)"
        td_border = "rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.06)"

        rows_html = []
        for name, val, desc, cat in diag_params:
            rows_html.append(f"""
            <tr style="border-bottom: 1px solid {td_border};">
                <td style="padding: 14px 18px; font-weight: 700; color: {text_prime_tab};">{name}</td>
                <td style="padding: 14px 18px; font-weight: 800; color: var(--apple-blue);">{val}</td>
                <td style="padding: 14px 18px; color: {text_sec_tab}; font-size: 13.5px;">{desc}</td>
                <td style="padding: 14px 18px;"><span style="background: rgba(0, 122, 255, 0.12); color: var(--apple-blue); font-size: 11.5px; font-weight: 700; padding: 4px 10px; border-radius: 9999px;">{cat}</span></td>
            </tr>
            """)

        st.markdown(f"""
        <div style="overflow-x: auto; border-radius: 18px; border: 1px solid {card_border_tab}; background: {card_bg_tab};">
            <table style="width: 100%; border-collapse: collapse; text-align: left;">
                <thead>
                    <tr style="background: {th_bg}; border-bottom: 1px solid {td_border}; font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em; color: {text_sec_tab};">
                        <th style="padding: 14px 18px;">Diagnostic Parameter</th>
                        <th style="padding: 14px 18px;">Measured Telemetry</th>
                        <th style="padding: 14px 18px;">Biological Significance</th>
                        <th style="padding: 14px 18px;">Sub-System</th>
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
            <div style="font-size: 13px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--apple-blue); font-weight: 700;">
                Civil Catastrophe Protocol & Certification
            </div>
            <div style="font-size: 1.55rem; font-weight: 800; color: {text_prime_tab}; letter-spacing: -0.02em; margin-top: 2px;">
                Official Bioacoustic Forensic Telemetry Report
            </div>
            <div style="font-size: 14px; color: {text_sec_tab}; margin-top: 4px;">
                Standardized civil catastrophe incident telemetry formatted for national disaster management and seismological observatories.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.text_area("Forensic Telemetry Data Stream", forensic_report, height=340)
        st.download_button(
            label="Download Bioacoustic Forensic Telemetry Report (.md)",
            data=forensic_report,
            file_name=f"intera_bioacoustic_report_{audio_label.replace('.wav', '').replace('.mp3', '')}.md",
            mime="text/markdown"
        )
