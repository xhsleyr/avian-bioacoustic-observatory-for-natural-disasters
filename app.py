"""Avian Early Warning System (AEWS) - Web Application.

A bioacoustic intelligence platform that analyzes bird vocalizations to detect
anomalous acoustic distress patterns preceding natural disasters.
"""

import os
import glob
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.audio_processor import load_audio, extract_acoustic_features
from src.distress_engine import compute_distress_index

# Page configuration
st.set_page_config(
    page_title="Avian Bioacoustics Early Warning System",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .threat-banner {
        border-radius: 8px;
        padding: 18px 22px;
        color: white;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


def create_gauge_chart(adi_score: float, threat_color: str) -> go.Figure:
    """Create a responsive Plotly gauge indicator for the Acoustic Distress Index."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=adi_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Acoustic Distress Index (ADI)", 'font': {'size': 18, 'color': "#1E293B"}},
        number={'suffix': "%", 'font': {'size': 38, 'color': threat_color, 'weight': "bold"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': threat_color, 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, 38], 'color': 'rgba(42, 157, 143, 0.2)'},
                {'range': [38, 70], 'color': 'rgba(244, 162, 97, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(230, 57, 70, 0.25)'}
            ],
            'threshold': {
                'line': {'color': "#E63946", 'width': 3},
                'thickness': 0.8,
                'value': 70.0
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="sans-serif")
    )
    return fig


def plot_mel_spectrogram(features: dict) -> go.Figure:
    """Create an interactive Mel-Spectrogram heatmap."""
    mel_db = features["mel_db"]
    times = features["times"]
    
    # 128 mel frequency bins mapped roughly across 200 Hz to 10000 Hz
    freq_bins = np.linspace(200, 10000, mel_db.shape[0])

    fig = go.Figure(data=go.Heatmap(
        z=mel_db,
        x=times,
        y=freq_bins,
        colorscale="Viridis",
        colorbar=dict(title="dB", len=0.8)
    ))
    fig.update_layout(
        title="Interactive Mel-Spectrogram (Frequency vs Time)",
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white"
    )
    return fig


def plot_waveform_and_dynamics(y: np.ndarray, sr: int, features: dict) -> go.Figure:
    """Plot synchronized waveform, RMS energy, and centroid trajectory."""
    time_wave = np.linspace(0, len(y) / sr, len(y))
    times_feats = features["times"]
    rms = features["rms_energy"]["trajectory"]
    centroid = features["spectral_centroid"]["trajectory"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Raw Acoustic Waveform & Call Onset Bursts", "Pitch / Spectral Centroid Trajectory (Hz)")
    )

    # Waveform
    # Downsample for faster web rendering if very long
    step = max(1, len(y) // 3000)
    fig.add_trace(
        go.Scatter(x=time_wave[::step], y=y[::step], mode='lines', name='Waveform', line=dict(color="#3B82F6", width=1)),
        row=1, col=1
    )

    # Overlay Onset markers
    onsets = features.get("onset_times", [])
    if len(onsets) > 0:
        fig.add_trace(
            go.Scatter(
                x=onsets,
                y=[0.85] * len(onsets),
                mode='markers',
                name='Call Bursts',
                marker=dict(symbol="triangle-down", size=9, color="#E63946")
            ),
            row=1, col=1
        )

    # Spectral Centroid (Pitch)
    fig.add_trace(
        go.Scatter(x=times_feats, y=centroid, mode='lines', name='Centroid (Hz)', line=dict(color="#F4A261", width=2)),
        row=2, col=1
    )

    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly_white",
        showlegend=True
    )
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Centroid (Hz)", row=2, col=1)

    return fig


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bird.png", width=64)
    st.title("AEWS Bioacoustics")
    st.markdown("**Avian Early Warning System**")
    st.caption("Final Year Engineering Project Prototype")
    st.divider()

    st.markdown("### 🔬 How it Works")
    st.markdown("""
    Birds exhibit significant vocal anomalies before natural events:
    * **Seismic P-waves:** Induce high-frequency alarm screams.
    * **Storm Infrasound:** Causes elevated call repetition and flock panic.
    * **Pressure Drops:** Triggers continuous agitation chatter.
    """)

    st.divider()
    st.markdown("### ⚙️ System Status")
    st.success("Core Engine: Local (Apple M1)")
    st.info("Analysis Window: 60s max")
    st.caption("Built with Python, Streamlit, Librosa & PyTorch")


# --- MAIN INTERFACE ---
st.markdown('<div class="main-title">🦅 Avian Vocal Bioacoustics for Natural Disaster Early Warning</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload or select bird vocalization audio to analyze acoustic panic signatures and predict imminent environmental disruptions.</div>', unsafe_allow_html=True)

# Select Input Mode
input_mode = st.radio(
    "Select Audio Source:",
    options=["📂 Upload Audio File", "🧪 Preset Benchmark Samples (Instant Demo)"],
    horizontal=True
)

audio_data = None
audio_label = ""

if input_mode == "📂 Upload Audio File":
    uploaded_file = st.file_uploader(
        "Choose an audio file (.wav, .mp3, .ogg, .flac)",
        type=["wav", "mp3", "ogg", "flac"]
    )
    if uploaded_file is not None:
        audio_data = uploaded_file.read()
        audio_label = uploaded_file.name
else:
    sample_files = sorted(glob.glob("data/samples/*.wav"))
    sample_names = {
        os.path.basename(p): p for p in sample_files
    }
    
    col_sel, _ = st.columns([2, 1])
    with col_sel:
        selected_sample_name = st.selectbox(
            "Select Benchmark Sample to Test:",
            options=list(sample_names.keys()),
            format_func=lambda x: {
                "calm_robin_song.wav": "🟢 Calm Robin Whistle (Baseline Normal)",
                "elevated_warning_call.wav": "🟡 Elevated Warning Call (Agitation Alert)",
                "critical_disaster_panic.wav": "🔴 Critical Disaster Panic Screech (Pre-Tremor Panic)"
            }.get(x, x)
        )
    
    if selected_sample_name:
        with open(sample_names[selected_sample_name], "rb") as f:
            audio_data = f.read()
        audio_label = selected_sample_name

if audio_data is not None:
    st.divider()
    st.subheader(f"Audio Source: `{audio_label}`")
    st.audio(audio_data)

    with st.spinner("Analyzing bioacoustic signal & extracting distress indices..."):
        y, sr, duration = load_audio(audio_data)
        features = extract_acoustic_features(y, sr)
        results = compute_distress_index(features)

    # 1. Early Warning Status Card
    threat_bg = results["threat_color"]
    st.markdown(f"""
    <div style="background-color: {threat_bg}; border-radius: 8px; padding: 18px 22px; color: white; margin-top: 15px; margin-bottom: 25px;">
        <h3 style="margin: 0; color: white; font-size: 1.4rem;">{results['alert_title']}</h3>
        <p style="margin-top: 8px; margin-bottom: 0; font-size: 1rem; opacity: 0.95;">{results['alert_summary']}</p>
    </div>
    """, unsafe_allow_html=True)

    # 2. Key Metrics Row
    col_gauge, col_stats = st.columns([1, 1.2])

    with col_gauge:
        gauge_fig = create_gauge_chart(results["adi"], results["threat_color"])
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col_stats:
        st.markdown("#### Diagnostic Bioacoustic Descriptors")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Pitch Center (Centroid)", f"{features['spectral_centroid']['mean']:.0f} Hz", delta=f"{results['sub_scores']['pitch']}% stress")
            st.metric("Call Burst Rate", f"{features['call_rate']:.2f} calls/sec", delta=f"{results['sub_scores']['call_rate']}% density")
        with c2:
            st.metric("Noise Harshness (Flatness)", f"{features['spectral_flatness']['mean']:.4f}", delta=f"{results['sub_scores']['flatness']}% chaos")
            st.metric("Duration Analyzed", f"{features['duration']:.2f} sec", delta=f"{features['total_onsets']} bursts")

    st.divider()

    # 3. Interactive Visualizers
    tab_spec, tab_wave, tab_table = st.tabs([
        "📊 Mel-Spectrogram Heatmap",
        "📈 Waveform & Call Onsets",
        "📋 Diagnostic Factor Breakdown"
    ])

    with tab_spec:
        st.plotly_chart(plot_mel_spectrogram(features), use_container_width=True)

    with tab_wave:
        st.plotly_chart(plot_waveform_and_dynamics(y, sr, features), use_container_width=True)

    with tab_table:
        st.markdown("#### Detailed Acoustic Diagnostic Metrics")
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
    st.info("👆 Please upload an audio file or select a preset benchmark sample to begin bioacoustic analysis.")
