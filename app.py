"""Project INTERA — Avian Bioacoustic Translation & Planetary Early Warning Intelligence.
Art-directed in the visual system of Project CETI with authentic human editorial craftsmanship.

Principles:
- Distinctive editorial presence: no generic AI template tropes, no redundant all-caps eyebrows, no cookie-cutter cards.
- Human typography hierarchy: elegant Times New Roman serif display titles, generous line-heights, and thoughtful scientific copy.
- Museum-grade biological plate for species identification with high-resolution imagery and verified ornithological metadata.
- Acoustic telemetry console: dark bioluminescent sonogram, live waveform visualizer, and calibrated Cornell BirdNET v2.4 classification.
- Robust security: input validation, 25MB file upload ceiling, SSRF defenses, and strict HTML escaping.
"""

import os
import glob
import html
import base64
import logging
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.audio_processor import load_audio, extract_acoustic_features
from src.distress_engine import compute_distress_index
from src.species_classifier import identify_species
from src.spoof_detector import evaluate_avian_authenticity
from src.image_fetcher import fetch_bird_visuals

logger = logging.getLogger(__name__)

# --- FIGMA / ICONS8 OUTLINE SVG ICONS ---
ICON_SHIELD = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>"""
ICON_ALERT = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>"""
ICON_CHECK = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14 recovery"/></svg>"""
ICON_CROSS = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#e63946" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px;"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>"""
ICON_MIC = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-3px; margin-right:6px;"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>"""
ICON_UPLOAD_FIGMA = """<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#4749fe" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;"><path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M12 12v9"/><path d="m16 16-4-4-4 4"/></svg>"""


def get_image_data_uri(path_or_url: str) -> str:
    """Convert local image paths into base64 Data URIs for instant, zero-lag browser rendering."""
    if not path_or_url:
        return ""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://") or path_or_url.startswith("data:"):
        return path_or_url
    if os.path.exists(path_or_url):
        try:
            with open(path_or_url, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(path_or_url)[1].lstrip(".").lower()
                mime = "jpeg" if ext in ["jpg", "jpeg"] else ("png" if ext == "png" else "webp")
                return f"data:image/{mime};base64,{encoded}"
        except Exception as e:
            logger.warning("Error reading species image '%s': %s", path_or_url, e)
            return path_or_url
    return path_or_url


# Page configuration — full-width, clean layout without sidebar
st.set_page_config(
    page_title="Project INTERA — Listening to the Sentinels",
    page_icon="https://cdn.prod.website-files.com/643ddd7ffdf12273933a8cec/645bdf9d4bac53320316d064_CETI-Favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Human Craft & Editorial Art Direction CSS
st.markdown("""
<style>
    :root {
        --intera-primary: #4d1bc7;
        --intera-accent: #4749fe;
        --intera-hover: #000000;
        --intera-bg: #ffffff;
        --intera-text-body: #000000;
        --intera-text-dark: #000000;
        --intera-text-muted: #000000;
        --intera-border: #e2e5ec;
        --intera-cyan: #befff7;
    }

    /* Universal Crisp Font Smoothing & Rendering */
    * {
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
        text-rendering: optimizeLegibility !important;
    }

    html, body, .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", system-ui, sans-serif !important;
        color: #000000 !important;
    }

    /* Human Editorial Display Serifs: Story Headings, Scientific Nomenclature */
    h1, h2, h3, h4, h5, h6,
    .intera-hero-title,
    .species-hero-title {
        font-family: -apple-system-ui-serif, 'Newsreader', 'Charter', 'Iowan Old Style', 'Palatino', Georgia, 'Times New Roman', serif !important;
        font-weight: 600 !important;
        line-height: 1.18 !important;
        color: #000000 !important;
        letter-spacing: -0.025em !important;
    }

    /* Human Editorial Narrative & Prose */
    p, .intera-hero-subtitle, .species-hero-desc, .intera-card p {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", -apple-system-ui-serif, Georgia, serif !important;
        font-size: 1.15rem !important;
        line-height: 1.68 !important;
        color: #000000 !important;
        font-weight: 450 !important;
    }

    /* macOS Native UI: Buttons, Tabs, Dropdowns, Badges, Metrics */
    .stButton > button,
    .stTabs [data-baseweb="tab"],
    div[data-testid="stRadio"],
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] div[role="radiogroup"] > label,
    div[data-testid="stSelectbox"],
    div[data-baseweb="select"],
    div[data-baseweb="popover"],
    .intera-pill,
    .intera-brand-title,
    .intera-brand-tag,
    .intera-nav,
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    .stressor-name, .stressor-val, .stressor-context,
    table, th, td {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Segoe UI", system-ui, sans-serif !important;
    }

    [data-testid="stMetricValue"], .stressor-val {
        font-variant-numeric: tabular-nums lining-nums !important;
    }

    /* Protect all icons and Material Symbols from font override to prevent ligature overlap */
    [class*="material-symbols"],
    [data-testid*="stIcon"],
    [data-testid="stFileUploaderDropzone"] span[translate="no"],
    .material-symbols-rounded,
    .material-symbols-outlined,
    svg, svg * {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", sans-serif !important;
    }

    /* Figma-Grade File Uploader Dropzone */
    [data-testid="stFileUploaderDropzone"] button span:first-child {
        display: none !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.88) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border: 1.5px dashed rgba(71, 73, 254, 0.35) !important;
        border-radius: 14px !important;
        padding: 26px 20px !important;
        text-align: center !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 16px rgba(30, 27, 75, 0.04) !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--intera-accent) !important;
        background: #ffffff !important;
        box-shadow: 0 8px 28px rgba(71, 73, 254, 0.15) !important;
        transform: translateY(-2px) !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background-color: var(--intera-accent) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 10px 28px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        border: 1px solid var(--intera-accent) !important;
        box-shadow: 0 4px 14px rgba(71, 73, 254, 0.28) !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #3b3de0 !important;
        border-color: #3b3de0 !important;
        box-shadow: 0 6px 18px rgba(71, 73, 254, 0.38) !important;
        transform: translateY(-1px) !important;
    }

    [data-testid="stFileUploaderDropzone"] button * {
        color: #ffffff !important;
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

    /* ========================================================
       Project CETI Atmospheric Sky Gradient Canvas
       (Authentic Periwinkle Sky, Cyan Aura, Violet Halo & Stipple Grain)
       ======================================================== */
    @keyframes cetiAuraMovement {
        0% {
            background-position:
                0% 0%,             /* Grain overlay */
                86% 16%,           /* Top-right turquoise seafoam glow */
                36% 44%,           /* Center-left ultraviolet halo */
                94% 68%,           /* Lower-right lilac haze */
                48% 115%,          /* Sculptural white bottom dune */
                0% 0%;             /* Base sky periwinkle gradient */
        }
        50% {
            background-position:
                0% 0%,
                92% 22%,
                32% 48%,
                88% 74%,
                50% 112%,
                0% 0%;
        }
        100% {
            background-position:
                0% 0%,
                86% 16%,
                36% 44%,
                94% 68%,
                48% 115%,
                0% 0%;
        }
    }

    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #9bb8f5 !important;
        background-image:
            /* 1. Project CETI Tactile Film Grain / Stipple Texture */
            url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.82' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.052'/%3E%3C/svg%3E"),
            /* 2. Top-Right Luminous Aqua / Turquoise Seafoam Glow */
            radial-gradient(ellipse 55% 45% at 86% 16%, rgba(186, 247, 240, 0.92) 0%, rgba(186, 247, 240, 0.45) 45%, transparent 72%),
            /* 3. Center-Left Glowing Ultraviolet Arc & Violet Halo */
            radial-gradient(ellipse 52% 64% at 36% 44%, rgba(88, 56, 248, 0.78) 0%, rgba(109, 40, 217, 0.45) 38%, transparent 68%),
            /* 4. Lower-Right Soft Lilac / Violet Haze */
            radial-gradient(ellipse 50% 50% at 94% 68%, rgba(216, 180, 254, 0.75) 0%, transparent 62%),
            /* 5. Sculptural Sweeping White Dune / Horizon Curve across bottom */
            radial-gradient(ellipse 135% 65% at 48% 115%, #ffffff 0%, #ffffff 52%, transparent 54%),
            /* 6. Base Sky Periwinkle & Azure Gradient */
            linear-gradient(175deg, #9bb8f5 0%, #b4cef8 30%, #d5e4fd 60%, #edf3fe 100%) !important;
        background-attachment: fixed !important;
        background-repeat: repeat, no-repeat, no-repeat, no-repeat, no-repeat, no-repeat !important;
        animation: cetiAuraMovement 24s ease-in-out infinite alternate !important;
        color: #1e1b4b !important;
    }

    /* Main Container Padding — Generous, Centered Editorial Canvas */
    .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 5rem !important;
        max-width: 1140px !important;
        margin: 0 auto !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
        display: none !important;
    }

    /* Global Headings */
    h1, h2, h3, h4, h5 {
        font-weight: 700 !important;
        line-height: 1.2 !important;
        color: var(--intera-text-dark) !important;
        letter-spacing: -0.015em !important;
    }

    /* Editorial Masthead */
    .intera-masthead {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 0 22px 0;
        border-bottom: 1px solid var(--intera-border);
        margin-bottom: 44px;
        flex-wrap: wrap;
        gap: 16px;
    }

    .intera-brand {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .intera-brand-title {
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--intera-text-dark);
    }

    .intera-brand-tag {
        font-size: 12px;
        letter-spacing: 0.08em;
        color: var(--intera-accent);
        font-style: italic;
    }

    .intera-nav {
        display: flex;
        gap: 28px;
        font-size: 13.5px;
        color: #000000 !important;
        font-weight: 600 !important;
        flex-wrap: wrap;
    }
    .intera-nav span {
        color: #000000 !important;
    }

    /* Project CETI Editorial Hero Section */
    .intera-hero {
        text-align: center;
        margin-bottom: 42px;
        position: relative;
    }

    .intera-hero-eyebrow {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", system-ui, sans-serif !important;
        font-size: 13.5px !important;
        font-weight: 700 !important;
        letter-spacing: 0.18em !important;
        text-transform: uppercase !important;
        color: #1e1b4b !important;
        margin-bottom: 8px !important;
    }

    .intera-hero-understand {
        font-family: -apple-system-ui-serif, 'Playfair Display', 'Bodoni MT', 'Didot', 'Times New Roman', serif !important;
        font-size: clamp(3.8rem, 8vw, 6.4rem) !important;
        font-weight: 700 !important;
        line-height: 0.98 !important;
        color: #ffffff !important;
        letter-spacing: -0.02em !important;
        text-shadow: 0 4px 28px rgba(30, 27, 75, 0.20), 0 1px 3px rgba(30, 27, 75, 0.14) !important;
        display: block !important;
    }

    .intera-hero-saying {
        font-family: -apple-system-ui-serif, 'Playfair Display', 'Bodoni MT', 'Times New Roman', serif !important;
        font-size: clamp(2.9rem, 6.2vw, 5.0rem) !important;
        font-weight: 800 !important;
        font-style: italic !important;
        line-height: 1.05 !important;
        color: #000000 !important;
        letter-spacing: -0.015em !important;
        margin-top: -6px !important;
        margin-bottom: 22px !important;
        display: block !important;
    }

    .intera-hero-subtitle {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", system-ui, sans-serif !important;
        font-size: 1.2rem !important;
        line-height: 1.68 !important;
        color: #000000 !important;
        max-width: 820px !important;
        margin: 0 auto !important;
        font-weight: 500 !important;
    }

    /* Perfectly Centered Mode Switcher */
    [data-testid="stElementContainer"]:has([data-testid="stRadio"]) {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        margin: 0 auto 20px auto !important;
    }

    div[data-testid="stRadio"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        width: 100% !important;
        margin: 0 auto !important;
    }

    div[data-testid="stRadio"] > label,
    div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
        font-size: 1.15rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: #1e1b4b !important;
        font-weight: 700 !important;
        margin-bottom: 14px !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        width: 100% !important;
    }

    div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] {
        text-align: center !important;
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
    }

    div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
        text-align: center !important;
        margin: 0 auto !important;
        width: 100% !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 14px !important;
        flex-wrap: wrap !important;
        width: 100% !important;
        margin: 0 auto !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label,
    div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"] {
        background-color: #ffffff !important;
        border: 1px solid var(--intera-border) !important;
        border-radius: 100px !important;
        padding: 11px 26px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label *,
    div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"] * {
        font-size: 16px !important;
        font-weight: 600 !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover,
    div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"]:hover {
        border-color: var(--intera-accent) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(71, 73, 254, 0.15) !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
    div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"]:has(input:checked) {
        background-color: var(--intera-accent) !important;
        color: #ffffff !important;
        border-color: var(--intera-accent) !important;
        box-shadow: 0 4px 16px rgba(71, 73, 254, 0.25) !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) *,
    div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"]:has(input:checked) * {
        color: #ffffff !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }

    /* ========================================================
       macOS Native Selectbox & Popover System
       ======================================================== */
    div[data-testid="stSelectbox"] {
        max-width: 680px !important;
        margin: 0 auto 28px auto !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] {
        font-size: 1.15rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: #1e1b4b !important;
        font-weight: 700 !important;
        text-align: center !important;
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
        margin-bottom: 12px !important;
    }

    div[data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] {
        text-align: center !important;
        width: 100% !important;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        width: 100% !important;
        border-radius: 10px !important;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.82) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border: 1px solid rgba(0, 0, 0, 0.14) !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
        padding: 8px 18px !important;
        min-height: 48px !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        cursor: pointer !important;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
        font-size: 16px !important;
    }

    /* macOS Selectbox Hover State */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
        background: #ffffff !important;
        border-color: rgba(71, 73, 254, 0.55) !important;
        box-shadow: 0 4px 18px rgba(71, 73, 254, 0.15), 0 1px 4px rgba(0, 0, 0, 0.08) !important;
        transform: translateY(-1px) !important;
    }

    /* macOS Selectbox Focus / Open State */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
        border-color: var(--intera-accent) !important;
        box-shadow: 0 0 0 3.5px rgba(71, 73, 254, 0.22), 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        background: #ffffff !important;
    }

    /* Chevron Icon in selectbox */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] svg {
        transition: transform 0.2s ease, stroke 0.2s ease !important;
        color: #000000 !important;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"]:hover svg {
        color: var(--intera-accent) !important;
    }

    /* macOS Floating Dropdown Menu (BaseWeb Popover) */
    div[data-baseweb="popover"] {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 16px 40px -6px rgba(0, 0, 0, 0.18), 0 0 0 1px rgba(0, 0, 0, 0.08) !important;
        z-index: 999999 !important;
    }

    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] [data-baseweb="menu"] {
        background: rgba(255, 255, 255, 0.94) !important;
        backdrop-filter: blur(24px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(190%) !important;
        border-radius: 12px !important;
        padding: 6px !important;
        border: none !important;
    }

    div[data-baseweb="popover"] li[role="option"],
    div[data-baseweb="popover"] [data-baseweb="menu"] li {
        border-radius: 7px !important;
        padding: 11px 16px !important;
        margin: 2px 0 !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        color: #1d1d1f !important;
        transition: background 0.15s ease, color 0.15s ease, transform 0.1s ease !important;
        cursor: pointer !important;
    }

    /* macOS Blue Highlight Hover on Menu Options */
    div[data-baseweb="popover"] li[role="option"]:hover,
    div[data-baseweb="popover"] [data-baseweb="menu"] li:hover {
        background: var(--intera-accent) !important;
        color: #ffffff !important;
    }

    div[data-baseweb="popover"] li[role="option"]:hover *,
    div[data-baseweb="popover"] [data-baseweb="menu"] li:hover * {
        color: #ffffff !important;
    }

    div[data-baseweb="popover"] li[aria-selected="true"],
    div[data-baseweb="popover"] [data-baseweb="menu"] li[aria-selected="true"] {
        background: rgba(71, 73, 254, 0.12) !important;
        font-weight: 700 !important;
        color: var(--intera-accent) !important;
    }

    div[data-baseweb="popover"] li[aria-selected="true"]:hover {
        background: var(--intera-accent) !important;
        color: #ffffff !important;
    }

    /* Architectural Content Cards (Frosted Glass Depth) */
    .intera-card,
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(255, 255, 255, 0.88) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border: 1px solid rgba(226, 229, 236, 0.85) !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 24px -3px rgba(77, 27, 199, 0.05) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease !important;
        margin-bottom: 24px !important;
    }

    .intera-card:hover,
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: var(--intera-accent) !important;
        box-shadow: 0 8px 32px -4px rgba(71, 73, 254, 0.08) !important;
    }

    .intera-card {
        padding: 32px 38px;
        box-sizing: border-box !important;
        max-width: 100% !important;
    }

    .intera-empty-state {
        text-align: center !important;
        padding: clamp(36px, 6vw, 68px) clamp(16px, 4vw, 36px) !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    .intera-empty-title {
        font-size: clamp(1.75rem, 4.2vw, 2.5rem) !important;
        font-weight: 700 !important;
        color: #000000 !important;
        margin-bottom: 14px !important;
        line-height: 1.22 !important;
        letter-spacing: -0.015em !important;
        word-break: break-word !important;
    }

    .intera-empty-desc {
        color: #000000 !important;
        font-size: clamp(1.02rem, 1.9vw, 1.18rem) !important;
        font-weight: 500 !important;
        max-width: 680px !important;
        margin: 0 auto 26px auto !important;
        line-height: 1.65 !important;
        word-break: break-word !important;
        padding: 0 4px !important;
    }

    /* Suppress empty containers */
    .intera-card:empty,
    [data-testid="stVerticalBlockBorderWrapper"]:empty,
    [data-testid="stVerticalBlockBorderWrapper"] > div:empty {
        display: none !important;
    }

    /* Museum Natural History Photo Frame */
    .photo-mat-frame {
        border: 1px solid #dcdfe6;
        padding: 10px;
        background: #ffffff;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
        border-radius: 3px;
    }

    /* Project CETI Atmospheric Gradient Banner */
    .intera-gradient-banner {
        background: linear-gradient(135deg, #befff7 0%, #4749fe 50%, #4d1bc7 100%);
        color: #ffffff;
        padding: 38px 42px;
        border-radius: 0px;
        margin-bottom: 26px;
        position: relative;
        overflow: hidden;
    }

    .intera-gradient-banner h2 {
        font-size: 2.35rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 10px;
        line-height: 1.2;
    }

    .intera-gradient-banner p {
        font-size: 1.18rem;
        line-height: 1.65;
        color: rgba(255, 255, 255, 0.98) !important;
        max-width: 860px;
        margin: 0;
    }

    /* Badges & Telemetry Tags */
    .intera-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        padding: 7px 18px;
        border-radius: 100px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        max-width: 100%;
        box-sizing: border-box;
        text-align: center;
        line-height: 1.45;
        overflow-wrap: break-word;
        word-break: normal;
    }

    .intera-pill-verified {
        background-color: rgba(71, 73, 254, 0.08);
        color: var(--intera-accent);
        border: 1px solid var(--intera-accent);
    }

    .intera-pill-warning {
        background-color: rgba(230, 57, 70, 0.08);
        color: #e63946;
        border: 1px solid #e63946;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--intera-accent) !important;
        color: #ffffff !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 11px 36px !important;
        border-radius: 0px !important;
        border: 1px solid var(--intera-accent) !important;
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
        border-bottom: 1px solid var(--intera-border);
        background: transparent;
        flex-wrap: wrap;
        justify-content: center;
        margin-bottom: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 13px !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border-radius: 0px !important;
        padding: 12px 6px !important;
        background: transparent !important;
        white-space: nowrap !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--intera-accent) !important;
        border-bottom: 2px solid var(--intera-accent) !important;
        font-weight: 700 !important;
    }

    /* Species Card Typography */
    .species-hero-title {
        font-size: clamp(2.4rem, 4.2vw, 3.4rem);
        font-weight: 700;
        color: var(--intera-text-dark);
        line-height: 1.15;
        margin: 8px 0 4px 0;
    }

    .species-hero-sci {
        font-size: 1.3rem;
        font-style: italic;
        color: var(--intera-accent);
        margin-bottom: 16px;
    }

    .species-hero-desc {
        font-size: 1.2rem;
        color: #000000;
        font-weight: 450;
        line-height: 1.7;
        margin-bottom: 18px;
    }

    /* Stressor Decomposition Architecture */
    .stressor-card {
        padding: 12px 0;
        border-bottom: 1px solid #edf0f5;
    }
    .stressor-card:last-child {
        border-bottom: none;
        padding-bottom: 0;
    }
    .stressor-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 4px;
    }
    .stressor-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #000000;
    }
    .stressor-val {
        font-size: 1.05rem;
        font-weight: 700;
        color: #4749fe;
    }
    .stressor-context {
        font-size: 0.92rem;
        color: #000000;
        font-weight: 500;
        line-height: 1.45;
        margin-top: 3px;
    }
    .stressor-bar-track {
        width: 100%;
        height: 7px;
        background-color: #f1f3f9;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 6px;
    }
    .stressor-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease-out;
    }

    /* Metrics Sizing */
    [data-testid="stMetricValue"] {
        font-size: 2.25rem !important;
        font-weight: 700 !important;
        color: var(--intera-text-dark) !important;
        line-height: 1.2 !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 13px !important;
        letter-spacing: 0.07em !important;
        text-transform: uppercase !important;
        color: #000000 !important;
        font-weight: 700 !important;
        line-height: 1.35 !important;
        font-weight: 700 !important;
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
        color: #000000 !important;
        font-weight: 700 !important;
        border-bottom: 2px solid var(--intera-border) !important;
        padding: 12px 16px !important;
        text-align: left !important;
        font-weight: 700 !important;
    }
    tbody tr td {
        border-bottom: 1px solid #f4f5f8 !important;
        padding: 13px 16px !important;
        color: #000000 !important;
        font-weight: 500 !important;
    }

    audio {
        width: 100%;
        margin-top: 8px;
    }

    /* ========================================================
       CROSS-DEVICE RESPONSIVE ARCHITECTURE & FLUID ENGINE
       (Smartphones, Foldables, Tablets, Desktops & 4K UHD TVs)
       ======================================================== */

    /* Universal fluid sizing, anti-overflow and box boundary control */
    *, *::before, *::after {
        box-sizing: border-box !important;
    }

    html, body, .stApp {
        overflow-x: hidden !important;
        width: 100% !important;
        max-width: 100vw !important;
    }

    img, video, canvas, svg {
        max-width: 100% !important;
        height: auto;
    }

    /* Keyboard & 10-Foot TV Remote Navigation Focus Indicators (WCAG 2.2 AA) */
    :focus-visible {
        outline: 3px solid var(--intera-accent) !important;
        outline-offset: 2.5px !important;
    }

    /* Plotly Chart Viewport Safety */
    [data-testid="stPlotlyChart"],
    .js-plotly-plot,
    .plot-container {
        width: 100% !important;
        max-width: 100% !important;
    }

    /* Touchscreen Targets: Ergonomic Tap Compliance (iOS / Android 44-48px) */
    @media (pointer: coarse) {
        button,
        [role="radio"],
        [role="tab"],
        div[data-baseweb="select"] > div,
        .stButton > button,
        [data-testid="stFileUploaderDropzone"] button {
            min-height: 48px !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label,
        div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"] {
            padding: 13px 22px !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label:active,
        .stButton > button:active {
            transform: scale(0.98) !important;
        }
    }

    /* --------------------------------------------------------
       DEVICE TIER 1: Ultra-Wide Screens & 4K UHD Smart TVs (>= 1800px)
       Command Center / 10-Foot Living Room Viewing Distance
       -------------------------------------------------------- */
    @media screen and (min-width: 1800px) {
        .block-container {
            max-width: 1680px !important;
            padding: 3.5rem 3.5rem 7.5rem 3.5rem !important;
        }

        .intera-hero-understand {
            font-size: clamp(5.2rem, 6.2vw, 7.8rem) !important;
        }

        .intera-hero-saying {
            font-size: clamp(4.0rem, 5vw, 6.0rem) !important;
        }

        .intera-hero-subtitle {
            font-size: 1.42rem !important;
            max-width: 1020px !important;
            line-height: 1.75 !important;
        }

        div[data-testid="stRadio"] > label,
        div[data-testid="stRadio"] [data-testid="stWidgetLabel"],
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] {
            font-size: 1.35rem !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label,
        div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"] {
            font-size: 18px !important;
            padding: 14px 34px !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label *,
        div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"] * {
            font-size: 18px !important;
        }

        div[data-testid="stSelectbox"] {
            max-width: 880px !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            min-height: 56px !important;
            font-size: 18px !important;
            padding: 12px 22px !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
            font-size: 18px !important;
        }

        .intera-card,
        [data-testid="stVerticalBlockBorderWrapper"] {
            padding: 42px 48px !important;
            border-radius: 12px !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 2.85rem !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 15px !important;
        }

        .species-hero-title {
            font-size: 3.6rem !important;
        }

        .species-hero-sci {
            font-size: 1.55rem !important;
        }

        .species-hero-desc {
            font-size: 1.35rem !important;
        }

        p, .species-hero-desc, .intera-card p {
            font-size: 1.25rem !important;
        }

        .stressor-name, .stressor-val {
            font-size: 1.25rem !important;
        }

        .stressor-context {
            font-size: 1.05rem !important;
        }

        table {
            font-size: 16px !important;
        }

        thead tr th {
            font-size: 15px !important;
            padding: 16px 20px !important;
        }

        tbody tr td {
            font-size: 16px !important;
            padding: 16px 20px !important;
        }
    }

    /* --------------------------------------------------------
       DEVICE TIER 2: Tablets & Small Laptops (768px - 1023px)
       iPads, Tablets, Foldables (Unfolded)
       -------------------------------------------------------- */
    @media screen and (min-width: 768px) and (max-width: 1023px) {
        .block-container {
            max-width: 95% !important;
            padding: 1.8rem 1.4rem 4.5rem 1.4rem !important;
        }

        .intera-card,
        [data-testid="stVerticalBlockBorderWrapper"] {
            padding: 26px 28px !important;
        }

        .intera-hero-understand {
            font-size: 4.6rem !important;
        }

        .intera-hero-saying {
            font-size: 3.6rem !important;
        }

        .intera-hero-subtitle {
            font-size: 1.15rem !important;
            padding: 0 16px !important;
        }

        div[data-testid="stSelectbox"] {
            max-width: 88% !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label {
            padding: 10px 20px !important;
            font-size: 15px !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label * {
            font-size: 15px !important;
        }
    }

    /* --------------------------------------------------------
       DEVICE TIER 3: Smartphones & Phablets (< 768px)
       iPhones, Pixels, Galaxy S, Mobile Landscape/Portrait
       -------------------------------------------------------- */
    @media screen and (max-width: 767px) {
        .block-container {
            max-width: 100% !important;
            padding: 1.1rem 0.85rem 3.5rem 0.85rem !important;
        }

        /* Streamlit Column Stacking: Natural Mobile Vertical Flow */
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 16px !important;
        }

        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
            flex: 1 1 100% !important;
            margin-bottom: 0px !important;
        }

        /* Masthead Re-centering */
        .intera-masthead {
            flex-direction: column !important;
            align-items: center !important;
            text-align: center !important;
            gap: 12px !important;
            padding: 10px 0 16px 0 !important;
            margin-bottom: 24px !important;
        }

        .intera-brand {
            flex-direction: column !important;
            gap: 4px !important;
            align-items: center !important;
        }

        .intera-nav {
            justify-content: center !important;
            gap: 14px !important;
            font-size: 12px !important;
            overflow-x: auto !important;
            max-width: 100% !important;
            white-space: nowrap !important;
        }

        /* Fluid Mobile Hero Typography */
        .intera-hero {
            margin-bottom: 26px !important;
        }

        .intera-hero-eyebrow {
            font-size: 11.5px !important;
            letter-spacing: 0.14em !important;
            margin-bottom: 6px !important;
        }

        .intera-hero-understand {
            font-size: clamp(2.4rem, 11.5vw, 3.8rem) !important;
            line-height: 1.05 !important;
        }

        .intera-hero-saying {
            font-size: clamp(1.95rem, 9.2vw, 3.1rem) !important;
            line-height: 1.08 !important;
            margin-top: -3px !important;
            margin-bottom: 16px !important;
        }

        .intera-hero-subtitle {
            font-size: 1.05rem !important;
            line-height: 1.55 !important;
            padding: 0 4px !important;
        }

        /* Segmented Channel Radio Controls: Full Width Touch Stacking */
        div[data-testid="stRadio"] > label,
        div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
            font-size: 1.05rem !important;
            margin-bottom: 10px !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] {
            flex-direction: column !important;
            width: 100% !important;
            gap: 8px !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label,
        div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"] {
            width: 100% !important;
            justify-content: center !important;
            text-align: center !important;
            padding: 12px 16px !important;
            font-size: 15px !important;
            box-sizing: border-box !important;
            white-space: normal !important;
        }

        /* Dropdown Full Width */
        div[data-testid="stSelectbox"] {
            max-width: 100% !important;
            width: 100% !important;
            margin-bottom: 20px !important;
        }

        div[data-testid="stSelectbox"] label,
        div[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] {
            font-size: 1.05rem !important;
            margin-bottom: 8px !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            min-height: 48px !important;
            padding: 8px 14px !important;
            font-size: 15px !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
            font-size: 15px !important;
        }

        /* Badges & Pills Responsive Wrapping */
        .intera-pill {
            white-space: normal !important;
            border-radius: 16px !important;
            padding: 8px 14px !important;
            font-size: 11px !important;
            line-height: 1.45 !important;
            max-width: 100% !important;
            display: inline-flex !important;
            text-align: center !important;
            box-sizing: border-box !important;
            word-break: normal !important;
            overflow-wrap: break-word !important;
        }

        /* Cards and Visual Banners */
        .intera-card,
        [data-testid="stVerticalBlockBorderWrapper"] {
            padding: 22px 18px !important;
            border-radius: 10px !important;
            margin-bottom: 18px !important;
        }

        .intera-gradient-banner {
            padding: 24px 20px !important;
            border-radius: 8px !important;
        }

        .intera-gradient-banner h2 {
            font-size: 1.75rem !important;
        }

        .intera-gradient-banner p {
            font-size: 1.02rem !important;
        }

        /* Tabs: Smooth Horizontal Swipe */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px !important;
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
            justify-content: flex-start !important;
            padding-bottom: 8px !important;
            -webkit-overflow-scrolling: touch !important;
            scrollbar-width: thin !important;
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 12px !important;
            padding: 10px 14px !important;
            flex-shrink: 0 !important;
        }

        /* Metrics & Species Cards */
        [data-testid="stMetricValue"] {
            font-size: 1.85rem !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 12px !important;
        }

        .species-hero-title {
            font-size: 2.1rem !important;
            line-height: 1.15 !important;
        }

        .species-hero-sci {
            font-size: 1.15rem !important;
            margin-bottom: 12px !important;
        }

        .species-hero-desc {
            font-size: 1.05rem !important;
            line-height: 1.6 !important;
        }

        /* File Uploader Dropzone on Mobile */
        [data-testid="stFileUploaderDropzone"] {
            padding: 18px 14px !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            width: 100% !important;
            padding: 12px 16px !important;
        }

        /* Table Smooth Scrollable Container */
        div[data-testid="stTable"],
        div:has(> table) {
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
            display: block !important;
            width: 100% !important;
        }

        table {
            min-width: 480px !important;
            font-size: 13px !important;
        }

        thead tr th {
            font-size: 12px !important;
            padding: 10px 12px !important;
        }

        tbody tr td {
            font-size: 13px !important;
            padding: 10px 12px !important;
        }
    }

    /* --------------------------------------------------------
       DEVICE TIER 4: Compact / Foldable Screens (<= 480px)
       iPhone SE, Galaxy Z Flip cover screen, Mini phones
       -------------------------------------------------------- */
    @media screen and (max-width: 480px) {
        .block-container {
            padding: 0.85rem 0.65rem 2.8rem 0.65rem !important;
        }

        .intera-card,
        [data-testid="stVerticalBlockBorderWrapper"] {
            padding: 18px 14px !important;
        }

        .intera-pill {
            font-size: 10.5px !important;
            padding: 7px 12px !important;
            border-radius: 14px !important;
            letter-spacing: 0.04em !important;
        }

        .intera-empty-state {
            padding: 24px 14px !important;
        }

        .intera-hero-understand {
            font-size: 2.35rem !important;
        }

        .intera-hero-saying {
            font-size: 1.95rem !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label {
            font-size: 14.5px !important;
            padding: 11px 12px !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label * {
            font-size: 14.5px !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            font-size: 14.5px !important;
            padding: 7px 12px !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
            font-size: 14.5px !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# --- PERFORMANCE CACHING PIPELINE ---
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


# --- PLOTLY VISUALIZATIONS (HUMAN EDITORIAL PRECISION) ---
def create_gauge_chart(adi_score: float, threat_color: str) -> go.Figure:
    """Create a refined semi-circular bioacoustic distress gauge indicator with solid black high-contrast typography."""
    bar_color = "#00876c" if adi_score < 38.0 else ("#d97706" if adi_score < 70.0 else "#e63946")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=adi_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "ACOUSTIC DISTRESS INDEX (ADI)",
            'font': {'size': 14, 'color': "#000000", 'family': "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', 'Segoe UI', Arial, sans-serif"}
        },
        number={
            'suffix': "%",
            'font': {'size': 56, 'color': "#000000", 'family': "-apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif"}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1.5,
                'tickcolor': "#000000",
                'tickfont': {'family': "-apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif", 'size': 12, 'color': "#000000"},
                'tickvals': [0, 25, 50, 75, 100]
            },
            'bar': {'color': bar_color, 'thickness': 0.35},
            'bgcolor': "#ffffff",
            'borderwidth': 1.5,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, 38], 'color': 'rgba(0, 196, 159, 0.22)'},
                {'range': [38, 70], 'color': 'rgba(244, 162, 97, 0.25)'},
                {'range': [70, 100], 'color': 'rgba(230, 57, 70, 0.28)'}
            ],
            'threshold': {
                'line': {'color': "#000000", 'width': 3.5},
                'thickness': 0.85,
                'value': 70.0
            }
        }
    ))
    fig.update_layout(
        height=265,
        margin=dict(l=15, r=15, t=32, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="-apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif", color="#000000")
    )
    return fig


def plot_mel_spectrogram(features: dict) -> go.Figure:
    """Create a high-contrast Bioluminescent Mel-Spectrogram with pitch contour overlay."""
    mel_db = features["mel_db"]
    times = features["times"]
    freq_bins = np.linspace(200, 10000, mel_db.shape[0])

    # Dynamic floor thresholding to eliminate murky background noise
    p22 = float(np.percentile(mel_db, 22))
    p_max = float(mel_db.max())

    colorscale_biolum = [
        [0.0, '#05070d'],   # Deep Obsidian Void
        [0.10, '#100b2b'],  # Midnight Amethyst
        [0.22, '#23126d'],  # Deep Electric Indigo
        [0.38, '#3c35d9'],  # Vivid Bio-Cobalt
        [0.55, '#0099cc'],  # Cerulean Ocean
        [0.72, '#00f0ff'],  # Luminescent Cyan
        [0.86, '#70e000'],  # Chartreuse Resonance
        [1.0, '#fffbe0']    # Pure White-Gold Vocal Peak
    ]

    fig = go.Figure()

    # 1. Main Bioacoustic Heatmap
    fig.add_trace(go.Heatmap(
        z=mel_db,
        x=times,
        y=freq_bins,
        colorscale=colorscale_biolum,
        zmin=p22,
        zmax=p_max,
        zsmooth='best',
        colorbar=dict(
            title=dict(
                text="Power (dB)",
                font=dict(family="Times New Roman, serif", size=12, color="#ffffff")
            ),
            tickfont=dict(family="Times New Roman, serif", color="#94a3b8", size=11),
            len=0.88,
            thickness=13,
            outlinewidth=0,
            bgcolor="rgba(0,0,0,0)"
        ),
        hovertemplate="<b>Freq:</b> %{y:.0f} Hz<br><b>Time:</b> %{x:.2f} s<br><b>Energy:</b> %{z:.1f} dB<extra></extra>"
    ))

    # 2. Harmonic Pitch Centroid Contour (Traced over the Spectrogram)
    centroid_traj = features.get("spectral_centroid", {}).get("trajectory")
    if centroid_traj is not None and len(centroid_traj) == len(times):
        fig.add_trace(go.Scatter(
            x=times,
            y=centroid_traj,
            mode='lines',
            name='Vocal Pitch Centroid',
            line=dict(color='#00e5ff', width=1.8),
            hovertemplate="<b>Vocal Pitch Centroid:</b> %{y:.0f} Hz<br><b>Time:</b> %{x:.2f} s<extra></extra>"
        ))

    # 3. Syrinx and Alarm Biological Reference Bands
    fig.add_hline(
        y=1500, line_dash='dash', line_color='rgba(0, 229, 255, 0.45)', line_width=1.2,
        annotation_text="Avian Syrinx Band (1.5 kHz)",
        annotation_position="top left",
        annotation_font=dict(family="Times New Roman, serif", size=11, color="#befff7")
    )
    fig.add_hline(
        y=4500, line_dash='dash', line_color='rgba(230, 57, 70, 0.55)', line_width=1.2,
        annotation_text="Alarm Screech Threshold (4.5 kHz)",
        annotation_position="top left",
        annotation_font=dict(family="Times New Roman, serif", size=11, color="#ff9b9b")
    )

    fig.update_layout(
        title=dict(
            text="High-Definition Mel Bioacoustic Sonogram & Harmonic Contour",
            font=dict(family="Times New Roman, serif", size=18, color="#ffffff")
        ),
        xaxis=dict(
            title=dict(text="Time Elapsed (seconds)", font=dict(family="Times New Roman, serif", size=13, color="#cbd5e1")),
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            tickfont=dict(family="Times New Roman, serif", color="#94a3b8")
        ),
        yaxis=dict(
            title=dict(text="Frequency (Hz)", font=dict(family="Times New Roman, serif", size=13, color="#cbd5e1")),
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            tickfont=dict(family="Times New Roman, serif", color="#94a3b8"),
            range=[200, 10000]
        ),
        height=380,
        margin=dict(l=20, r=20, t=44, b=25),
        paper_bgcolor='#070912',
        plot_bgcolor='#05070d',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="Times New Roman, serif", size=11, color="#94a3b8"),
            bgcolor="rgba(0,0,0,0)"
        )
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
        font=dict(family="-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', sans-serif", color="#000000")
    )
    fig.update_xaxes(title_text="Time (seconds)", title_font=dict(color="#000000", size=13), tickfont=dict(color="#000000", size=11), row=2, col=1, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(title_text="Amplitude", title_font=dict(color="#000000", size=13), tickfont=dict(color="#000000", size=11), row=1, col=1, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(title_text="Centroid (Hz)", title_font=dict(color="#000000", size=13), tickfont=dict(color="#000000", size=11), row=2, col=1, gridcolor="rgba(0,0,0,0.08)")
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(family="-apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif", size=14, color="#000000")

    return fig


# --- EDITORIAL MASTHEAD ---
st.markdown("""
<div class="intera-masthead">
    <div class="intera-brand">
        <span style="width: 9px; height: 9px; border-radius: 50%; background-color: #4749fe; display: inline-block;"></span>
        <span class="intera-brand-title">Project INTERA</span>
        <span class="intera-brand-tag">Bioacoustic Observatory</span>
    </div>
    <div class="intera-nav">
        <span style="color: #4749fe; font-weight: 700;">HOME</span>
        <span>RESEARCH</span>
        <span>SPECIES ATLAS (6,522)</span>
        <span>ANTI-SPOOFING</span>
        <span>EARLY WARNING</span>
    </div>
</div>
""", unsafe_allow_html=True)

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

# Mode selection
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
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-bottom: 16px; text-align: center;">
            <div style="width: 58px; height: 58px; border-radius: 16px; background: linear-gradient(135deg, rgba(71, 73, 254, 0.12) 0%, rgba(186, 247, 240, 0.45) 100%); border: 1.5px solid rgba(71, 73, 254, 0.22); display: flex; align-items: center; justify-content: center; box-shadow: 0 6px 20px rgba(71, 73, 254, 0.12); margin-bottom: 12px; transition: transform 0.2s ease;">
                {ICON_UPLOAD_FIGMA}
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #1e1b4b; letter-spacing: 0.05em; text-transform: uppercase;">
                Upload Bioacoustic Recording
            </div>
            <div style="font-size: 14px; color: #000000; font-weight: 500; margin-top: 4px; max-width: 440px; line-height: 1.5;">
                Drop high-fidelity avian audio files (.wav, .mp3, .ogg, .flac up to 25 MB) for automated BirdNET classification and syrinx verification.
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
        st.markdown(f"<div style='text-align: center; margin-bottom: 6px;'><strong>{ICON_MIC} Real-Time Environmental Audio Stream</strong></div>", unsafe_allow_html=True)
        st.caption("Point your Mac or directional microphone towards the open canopy. Click Record to sample ambient avian signals.")
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
                "human_whistling_mimic.wav": "OBS-06 // Homo sapiens Oral Cavity — Anti-Spoofing Whistling Test"
            }.get(x, x)
        )
    
    if selected_sample_name:
        with open(sample_names[selected_sample_name], "rb") as f:
            audio_data = f.read()
        audio_label = selected_sample_name

if audio_data is not None:
    # 0ms Instant Response via Cached Pipeline
    try:
        with st.spinner("Decoding bioacoustic signals with Cornell BirdNET & verifying biometrics..."):
            y, sr, duration, features, results, species_info, spoof_info = cached_process_audio(audio_data)
    except Exception as exc:
        st.error(f"Bioacoustic Ingestion Error: Unable to decode audio stream ({type(exc).__name__}). Please provide a valid, uncorrupted audio recording.")
        audio_data = None

if audio_data is not None:
    prim_sp = species_info["primary_species"]
    safe_audio_label = html.escape(str(audio_label))
    safe_com_name = html.escape(str(prim_sp['common_name']))
    safe_sci_name = html.escape(str(prim_sp['scientific_name']))
    safe_spoof_label = html.escape(str(spoof_info['status_label']))
    safe_spoof_summary = html.escape(str(spoof_info['status_summary']))

    # Acoustic Feed Header
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 32px; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
        <div>
            <div style="font-size: 13px; letter-spacing: 0.12em; text-transform: uppercase; color: #4749fe; margin-bottom: 4px; font-weight: 700;">
                Acoustic Specimen Telemetry
            </div>
            <div style="font-size: 2.1rem; font-weight: 700; color: #000000;">
                {safe_audio_label}
            </div>
        </div>
        <div style="font-size: 14px; color: #000000; font-weight: 600;">
            Duration: <strong>{duration:.2f}s</strong> &bull; Sampling Rate: <strong>{sr:,} Hz</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.audio(audio_data)

    # 1. Biometric Verification Assessment Card
    if spoof_info["is_authentic"]:
        st.markdown(f"""
        <div class="intera-card" style="border-left: 4px solid #4749fe; padding: 22px 30px; margin-top: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <span class="intera-pill intera-pill-verified">
                    {ICON_SHIELD} {safe_spoof_label}
                </span>
                <span style="font-size: 13.5px; color: #000000; font-weight: 600;">
                    Cornell BirdNET Confidence: <strong>{spoof_info['birdnet_conf']}%</strong> &bull; Syrinx Harmonics Verified
                </span>
            </div>
            <p style="margin: 10px 0 0 0; font-size: 1.1rem; color: #000000; font-weight: 500; line-height: 1.6;">
                {safe_spoof_summary}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="intera-card" style="border-left: 4px solid #e63946; padding: 22px 30px; margin-top: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <span class="intera-pill intera-pill-warning">
                    {ICON_ALERT} {safe_spoof_label}
                </span>
                <span style="font-size: 13.5px; color: #e63946; font-weight: 700;">
                    Adversarial Non-Avian Acoustic Mimicry
                </span>
            </div>
            <p style="margin: 10px 0 0 0; font-size: 1.1rem; color: #000000; font-weight: 500; line-height: 1.6;">
                {safe_spoof_summary}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 2. Museum Natural History Species Plate
    if spoof_info["is_authentic"]:
        visuals = cached_fetch_visuals(prim_sp['common_name'], prim_sp['scientific_name'])
        safe_desc = html.escape(str(visuals['description']))
        safe_source = html.escape(str(visuals['source']))
        photo_uri = get_image_data_uri(visuals.get("image_url", ""))
        if not photo_uri:
            photo_uri = "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=600&q=80"
        
        with st.container(border=True):
            col_photo, col_details = st.columns([1.15, 2.05])
            
            with col_photo:
                st.markdown(f"""
                <div class="photo-mat-frame">
                    <img src="{photo_uri}" alt="{safe_com_name}" style="width: 100%; height: 260px; display: block; object-fit: cover; border-radius: 2px;" />
                    <div style="font-size: 12.5px; color: #000000; margin-top: 10px; text-align: center; font-weight: 600; line-height: 1.4;">
                        Specimen Photographic Archive: {safe_com_name} ({safe_source})
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_details:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; flex-wrap: wrap; gap: 6px;">
                    <span style="background-color: #4749fe; color: #ffffff; font-size: 12px; padding: 4px 14px; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700;">
                        Cornell BirdNET &bull; {prim_sp['confidence']}% Acoustic Match
                    </span>
                    <span style="font-size: 13px; color: #000000; font-weight: 700;">6,522 Species Global Taxonomy</span>
                </div>
                <div class="species-hero-title">{safe_com_name}</div>
                <div class="species-hero-sci">{safe_sci_name}</div>
                <div class="species-hero-desc">
                    {safe_desc}
                </div>
                <div style="border-top: 1px solid #e2e5ec; padding-top: 16px; margin-top: 16px; font-size: 1.05rem;">
                    <p style="margin: 0 0 8px 0; color: #000000; font-weight: 500;">
                        <strong>Acoustic State:</strong> {prim_sp['pitch_status']} &bull; Pitch Centroid: <strong>{prim_sp['detected_pitch']:.0f} Hz</strong>
                    </p>
                    <p style="margin: 4px 0 0 0; color: #4749fe; font-weight: 700;">
                        <strong>Disaster Sensitivity Profile:</strong> <span style="font-weight: 500; color: #000000;">{prim_sp['disaster_sensitivity']}</span>
                    </p>
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
        factors_items = "".join(f"<li style='margin-bottom: 5px;'>{html.escape(f)}</li>" for f in spoof_info.get('spoof_factors', []))
        
        st.markdown(f"""
        <div class="intera-card" style="border-left: 4px solid #e63946; padding: 28px 34px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 6px;">
                <span style="background-color: #000000; color: #ffffff; font-size: 12px; padding: 4px 14px; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700;">
                    Biometric Rejection &bull; Non-Avian Acoustic Source
                </span>
                <span style="font-size: 13px; color: #e63946; font-weight: 700;">
                    Sub-1kHz Energy: {spoof_info.get('sub1k_ratio', 0.0)}% (Avian Baseline &lt; 8%)
                </span>
            </div>
            <div class="species-hero-title" style="margin-top: 8px; font-size: 2.1rem; color: #000000;">
                {detected_non_bird_title}
            </div>
            <p style="font-size: 1.15rem; color: #000000; font-weight: 500; margin-top: 10px; line-height: 1.65;">
                {html.escape(spoof_info['status_summary'])}
            </p>
            <div style="margin-top: 14px; border-top: 1px solid #e2e5ec; padding-top: 14px; font-size: 0.98rem; color: #000000;">
                <strong>Biometric Rejection Criteria Triggered:</strong>
                <ul style="margin: 6px 0 0 18px; padding: 0;">
                    {factors_items}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Early Warning Threat Assessment Banner
    if spoof_info["suppress_alarm"]:
        st.markdown(f"""
        <div class="intera-card" style="border-left: 4px solid #e63946; padding: 26px 34px;">
            <h3 style="font-size: 1.75rem; color: #e63946; margin: 0 0 8px 0;">
                {ICON_ALERT} Civil Disaster Early-Warning Trigger Safely Disarmed
            </h3>
            <p style="color: #000000; font-size: 1.12rem; margin: 0; font-weight: 500; line-height: 1.6;">
                Because domestic animal acoustics (dog bark, cat meow), smartphone loudspeaker playback, or non-avian ambient noise was detected, automated disaster triggers have been disarmed to prevent civil false panic.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="intera-gradient-banner">
            <h2>{results['alert_title']}</h2>
            <p>{results['alert_summary']}</p>
        </div>
        """, unsafe_allow_html=True)

    # 4. Quantitative Metrics Row (Redesigned ADI Gauge & Multi-Factor Stressor Decomposition)
    col_gauge, col_stats = st.columns([1, 1.25])

    adi_val = results["adi"]
    if adi_val < 38.0:
        adi_tier_label = "● Calm Baseline // Melodic Equilibrium"
        adi_tier_color = "#00876c"
        adi_tier_desc = "Vocalizations exhibit stable melodic contours, regular intervals, and tonal syrinx harmonic resonance."
    elif adi_val < 70.0:
        adi_tier_label = "▲ Elevated Vigilance // Canopy Agitation"
        adi_tier_color = "#d97706"
        adi_tier_desc = "Noticeable frequency shift and accelerated call density detected. Indicates local environmental agitation."
    else:
        adi_tier_label = "■ Acute Biotic Distress // Evacuation Call"
        adi_tier_color = "#e63946"
        adi_tier_desc = "Extreme high-frequency screeching and frantic burst pacing. Strongly correlates with imminent natural hazards."

    with col_gauge:
        with st.container(border=True):
            gauge_fig = create_gauge_chart(results["adi"], results["threat_color"])
            st.plotly_chart(gauge_fig, width="stretch")
            st.markdown(f"""
            <div style="text-align: center; margin-top: -4px; padding-bottom: 8px;">
                <div style="font-size: 14.5px; font-weight: 800; color: #000000; letter-spacing: 0.05em; text-transform: uppercase;">
                    <span style="color: {adi_tier_color}; margin-right: 4px;">●</span> {adi_tier_label}
                </div>
                <div style="font-size: 13.5px; color: #000000; margin-top: 5px; line-height: 1.5; font-weight: 500;">
                    {adi_tier_desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_stats:
        with st.container(border=True):
            sub_pitch = results['sub_scores'].get('pitch', 0.0)
            sub_rate = results['sub_scores'].get('call_rate', 0.0)
            sub_flat = results['sub_scores'].get('flatness', 0.0)
            sub_energy = results['sub_scores'].get('energy_volatility', results['sub_scores'].get('energy', 0.0))
            
            c_pitch = "#e63946" if sub_pitch > 65 else ("#f4a261" if sub_pitch > 35 else "#4749fe")
            c_rate = "#e63946" if sub_rate > 65 else ("#f4a261" if sub_rate > 35 else "#4749fe")
            c_flat = "#e63946" if sub_flat > 65 else ("#f4a261" if sub_flat > 35 else "#4749fe")
            c_energy = "#e63946" if sub_energy > 65 else ("#f4a261" if sub_energy > 35 else "#4749fe")

            st.markdown(f"""
            <div style="margin-bottom: 14px;">
                <div style="font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: #4749fe; font-weight: 700; margin-bottom: 2px;">
                    Bioacoustic Stress Decomposition
                </div>
                <div style="font-size: 1.45rem; font-weight: 700; color: #000000;">
                    Empirical Stressor Attribution & Pacing
                </div>
            </div>

            <!-- Stressor 1: Pitch -->
            <div class="stressor-card">
                <div class="stressor-header">
                    <span class="stressor-name">Pitch Elevation (Centroid)</span>
                    <span class="stressor-val" style="color: {c_pitch};">{features['spectral_centroid']['mean']:.0f} Hz &bull; {sub_pitch:.0f}%</span>
                </div>
                <div class="stressor-bar-track">
                    <div class="stressor-bar-fill" style="width: {min(100.0, max(4.0, sub_pitch))}%; background-color: {c_pitch};"></div>
                </div>
                <div class="stressor-context">Avian Syrinx Baseline: 1,500 – 2,800 Hz (Stress shriek onset &gt; 3,500 Hz)</div>
            </div>

            <!-- Stressor 2: Call Burst Rate -->
            <div class="stressor-card">
                <div class="stressor-header">
                    <span class="stressor-name">Temporal Burst Cadence</span>
                    <span class="stressor-val" style="color: {c_rate};">{features['call_rate']:.2f} calls/s &bull; {sub_rate:.0f}%</span>
                </div>
                <div class="stressor-bar-track">
                    <div class="stressor-bar-fill" style="width: {min(100.0, max(4.0, sub_rate))}%; background-color: {c_rate};"></div>
                </div>
                <div class="stressor-context">Equilibrium pacing: 0.3 – 1.2 calls/s (Agitated burst rate &gt; 2.5 calls/s)</div>
            </div>

            <!-- Stressor 3: Noise Harshness / Chaos -->
            <div class="stressor-card">
                <div class="stressor-header">
                    <span class="stressor-name">Spectral Flatness & Harsh Chaos</span>
                    <span class="stressor-val" style="color: {c_flat};">{features['spectral_flatness']['mean']:.4f} &bull; {sub_flat:.0f}%</span>
                </div>
                <div class="stressor-bar-track">
                    <div class="stressor-bar-fill" style="width: {min(100.0, max(4.0, sub_flat))}%; background-color: {c_flat};"></div>
                </div>
                <div class="stressor-context">Melodic purity: &lt; 0.010 (Screech noise turbulence &gt; 0.045)</div>
            </div>

            <!-- Stressor 4: Dual Syrinx Verification -->
            <div class="stressor-card">
                <div class="stressor-header">
                    <span class="stressor-name">Biometric Vocal Tract Organ</span>
                    <span class="stressor-val" style="color: {'#10b981' if spoof_info['is_authentic'] else '#e63946'};">{'Avian Syrinx' if spoof_info['is_authentic'] else spoof_info.get('detected_category', 'Mammalian').replace('_', ' ').title()}</span>
                </div>
                <div class="stressor-bar-track">
                    <div class="stressor-bar-fill" style="width: {100.0 if spoof_info['is_authentic'] else 20.0}%; background-color: {'#10b981' if spoof_info['is_authentic'] else '#e63946'};"></div>
                </div>
                <div class="stressor-context">Sub-600Hz breath turbulence: {spoof_info['breath_ratio']}% (Human threshold &gt; 28%)</div>
            </div>
            """, unsafe_allow_html=True)

    # 5. Scientific Telemetry Tabs
    tab_spec, tab_wave, tab_species, tab_anti_spoof, tab_table = st.tabs([
        "Mel-Spectrogram Heatmap",
        "Waveform & Onset Dynamics",
        "Species Taxonomy (6,522)",
        "Biometric Syrinx Verification",
        "Diagnostic Parameters"
    ])

    with tab_spec:
        with st.container(border=True):
            st.plotly_chart(plot_mel_spectrogram(features), width="stretch")

    with tab_wave:
        with st.container(border=True):
            st.plotly_chart(plot_waveform_and_dynamics(y, sr, features), width="stretch")

    with tab_species:
        with st.container(border=True):
            st.markdown("""<h4 style="font-size: 1.55rem; margin-bottom: 18px;">Top Species Candidates (Cornell BirdNET 6,522 Engine)</h4>""", unsafe_allow_html=True)
            for cand in species_info["top_candidates"]:
                col_info, col_bar = st.columns([2.5, 1])
                with col_info:
                    st.markdown(f"**{html.escape(cand['common_name'])}** (*{html.escape(cand['scientific_name'])}*)")
                    st.caption(f"Disaster Profile: {html.escape(cand['disaster_sensitivity'])}")
                with col_bar:
                    st.progress(min(1.0, cand['confidence'] / 100.0))
                    st.caption(f"Confidence: **{cand['confidence']}%**")
                st.divider()

    with tab_anti_spoof:
        with st.container(border=True):
            st.markdown("""<h4 style="font-size: 1.55rem; margin-bottom: 18px;">Biometric Verification & Anti-Spoofing Assessment</h4>""", unsafe_allow_html=True)
            
            status_html = f"{ICON_CHECK} <strong>Authentic Avian Syrinx Verified</strong>" if spoof_info['is_authentic'] else f"{ICON_CROSS} <strong>Rejected: {html.escape(spoof_info['status_label'])}</strong>"
            st.markdown(f"<div style='margin-bottom: 12px; font-size: 1.15rem;'>{status_html}</div>", unsafe_allow_html=True)
            st.write(f"**Non-Avian Risk Score:** `{spoof_info['spoof_score']} / 100`")
            st.write(f"**Sub-1,000 Hz Low-Frequency Energy:** `{spoof_info['sub1k_ratio']}%` (Avian Baseline &lt; 8%; Mammals &gt; 20%)")
            st.write(f"**Detected Category:** `{spoof_info.get('detected_category', 'unknown')}`")
            
            if spoof_info["spoof_factors"]:
                st.markdown("##### Detected Anti-Spoofing Flags:")
                for factor in spoof_info["spoof_factors"]:
                    st.warning(f"• {html.escape(factor)}")
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
    <div class="intera-card intera-empty-state">
        <div class="intera-empty-title">
            Awaiting Bioacoustic Feed
        </div>
        <p class="intera-empty-desc">
            Select a calibrated laboratory benchmark, upload an avian field recording, or initiate live microphonic acquisition to commence neural species identification and disaster threat forecasting.
        </p>
        <div>
            <span class="intera-pill intera-pill-verified">
                {ICON_SHIELD} 6,522 Species Active &bull; Biometric Anti-Spoofing Enabled
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
