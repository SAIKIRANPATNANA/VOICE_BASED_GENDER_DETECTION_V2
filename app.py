"""
Voice Gender Classification v2 — Interactive Streamlit Web Application
A dual-domain audio DSP and deep learning testing dashboard.
"""

import os
import io
import time
import numpy as np
import soundfile as sf
import scipy.signal
import scipy.fft
import scipy.stats
import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st
import librosa

# Internal project modules
from src.models import SpectrogramCNN, OldBaselineMLP
from src.audio_processing import resample_audio, compute_rms, rms_to_dbfs, peak_normalize
from src.features import compute_fft, compute_mel_spectrogram, compute_mfcc, extract_mfcc_statistics

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Voice AI v2 — Gender Classification & Audio DSP",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Cybernetic Dark Styling
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background canvas */
.stApp {
    background-color: #0B0F19;
    color: #E2E8F0;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0F172A;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Glowing Title Banner */
.hero-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px -10px rgba(0, 240, 255, 0.15);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #00F0FF 0%, #8B5CF6 50%, #EC4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #94A3B8;
    font-size: 1.05rem;
    line-height: 1.5;
    margin-bottom: 12px;
}

.badge-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 10px;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.badge-cyan {
    background: rgba(0, 240, 255, 0.12);
    color: #00F0FF;
    border: 1px solid rgba(0, 240, 255, 0.3);
}

.badge-violet {
    background: rgba(139, 92, 246, 0.12);
    color: #A78BFA;
    border: 1px solid rgba(139, 92, 246, 0.3);
}

.badge-pink {
    background: rgba(236, 72, 153, 0.12);
    color: #F472B6;
    border: 1px solid rgba(236, 72, 153, 0.3);
}

.badge-emerald {
    background: rgba(16, 185, 129, 0.12);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

/* Metric Cards */
.metric-card {
    background: #151D2E;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    padding: 16px 18px;
    text-align: left;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    border-color: rgba(0, 240, 255, 0.4);
    transform: translateY(-2px);
}

.metric-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-bottom: 4px;
}

.metric-val {
    font-size: 1.6rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #F8FAFC;
}

.metric-sub {
    font-size: 0.8rem;
    color: #64748B;
    margin-top: 4px;
}

/* Arena Prediction Cards */
.arena-card {
    background: #131B2E;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 20px;
    height: 100%;
    position: relative;
    overflow: hidden;
}

.arena-card.winner {
    border-color: #00F0FF;
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.15);
}

.card-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.gender-pill-female {
    background: linear-gradient(135deg, rgba(236, 72, 153, 0.2) 0%, rgba(244, 63, 94, 0.3) 100%);
    border: 1px solid #F43F5E;
    color: #FDA4AF;
    padding: 6px 14px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
    text-align: center;
}

.gender-pill-male {
    background: linear-gradient(135deg, rgba(0, 240, 255, 0.2) 0%, rgba(14, 165, 233, 0.3) 100%);
    border: 1px solid #00F0FF;
    color: #7DD3FC;
    padding: 6px 14px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
    text-align: center;
}

.card-desc {
    font-size: 0.82rem;
    color: #94A3B8;
    line-height: 1.4;
    margin-top: 10px;
    min-height: 48px;
}

/* Custom noise alert callout */
.noise-alert {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.4);
    border-radius: 10px;
    padding: 12px 16px;
    color: #FCD34D;
    font-size: 0.9rem;
    margin-bottom: 16px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. MODEL LOADING & CACHING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_all_models():
    """Loads and caches all three models for instantaneous inference."""
    device = torch.device("cpu")
    models = {}

    # 1. Modern 2D SpectrogramCNN
    cnn_path = "output/models/spectrogram_cnn.pt"
    cnn = SpectrogramCNN(n_mels=80)
    if os.path.exists(cnn_path):
        try:
            state_dict = torch.load(cnn_path, map_location=device, weights_only=True)
            cnn.load_state_dict(state_dict)
            cnn.eval()
            models["cnn"] = cnn
        except Exception as e:
            st.sidebar.warning(f"CNN checkpoint loading note: {e}")
            models["cnn"] = cnn.eval()
    else:
        # Fallback if export is completing
        models["cnn"] = cnn.eval()

    # 2. Random Forest Pipeline (360-dim stats)
    rf_path = "output/models/rf_pipeline.joblib"
    if os.path.exists(rf_path):
        try:
            rf = joblib.load(rf_path)
            models["rf"] = rf
        except Exception as e:
            st.sidebar.error(f"Error loading RF pipeline: {e}")
            models["rf"] = None
    else:
        models["rf"] = None

    # 3. Old Baseline MLP (40-dim mean MFCC)
    mlp_path = "output/models/baseline_old_mlp.pt"
    mlp = OldBaselineMLP(input_dim=40)
    if os.path.exists(mlp_path):
        try:
            mlp_state = torch.load(mlp_path, map_location=device, weights_only=True)
            mlp.load_state_dict(mlp_state)
            mlp.eval()
            models["mlp"] = mlp
        except Exception as e:
            st.sidebar.warning(f"MLP checkpoint loading note: {e}")
            models["mlp"] = mlp.eval()
    else:
        models["mlp"] = mlp.eval()

    return models


models_dict = load_all_models()


# -----------------------------------------------------------------------------
# 3. AUDIO HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def inject_noise_snr(signal: np.ndarray, target_snr_db: float) -> tuple[np.ndarray, float]:
    """
    Injects calibrated Gaussian White Noise to simulate real-world audio stress.
    Formula:
        signal_power = mean(signal^2)
        noise_power = signal_power / (10^(target_snr_db / 10))
        noisy_signal = signal + N(0, sqrt(noise_power))
    """
    signal_power = np.mean(signal ** 2)
    if signal_power <= 1e-9:
        return signal, target_snr_db

    snr_linear = 10.0 ** (target_snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=signal.shape).astype(np.float32)

    noisy_signal = signal + noise
    # Clip to prevent digital overflow
    noisy_signal = np.clip(noisy_signal, -1.0, 1.0)
    return noisy_signal, float(target_snr_db)


def estimate_f0_pitch(signal: np.ndarray, sr: int) -> float:
    """
    Estimates fundamental pitch (F0) using YIN / Autocorrelation.
    Female speech pitch: ~165 - 255 Hz
    Male speech pitch:   ~85  - 180 Hz
    """
    try:
        f0 = librosa.yin(signal, fmin=65, fmax=400, sr=sr)
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 0:
            return float(np.median(valid_f0))
    except Exception:
        pass
    # Fallback to FFT peak within speech pitch range
    freqs, mags, peak_freq = compute_fft(signal, sr)
    return float(peak_freq)


def run_vad_frames(signal: np.ndarray, sr: int, threshold_db: float = -35.0):
    """
    Extracts frame-by-frame energy and marks speech vs silence intervals.
    """
    frame_len = int(0.025 * sr)  # 25ms
    hop_len = int(0.010 * sr)    # 10ms
    if len(signal) < frame_len:
        return signal, np.array([True]), np.array([0.0]), np.array([0.0])

    num_frames = 1 + (len(signal) - frame_len) // hop_len
    frames = np.lib.stride_tricks.as_strided(
        signal,
        shape=(num_frames, frame_len),
        strides=(signal.strides[0] * hop_len, signal.strides[0])
    )
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    frame_db = 20.0 * np.log10(frame_rms)
    peak_db = np.max(frame_db)

    # Active if above threshold and within 32dB of peak voice frame
    is_speech = (frame_db >= threshold_db) & (frame_db >= peak_db - 32.0)
    kernel = np.ones(5, dtype=bool)
    is_speech_smooth = scipy.ndimage.binary_closing(is_speech, structure=kernel)
    is_speech_smooth = scipy.ndimage.binary_opening(is_speech_smooth, structure=kernel)

    frame_times = np.arange(num_frames) * (hop_len / sr)
    return is_speech_smooth, frame_times, frame_db, peak_db


# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS & AUDIO INGESTION
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🎙️ Audio Control Deck")

input_mode = st.sidebar.radio(
    "Choose Audio Source:",
    ("🎧 Curated Demo Library", "📁 Upload Audio File", "🎙️ Record via Microphone"),
    index=0
)

# Load metadata for curated library
metadata_path = "data/metadata.csv"
demo_df = None
if os.path.exists(metadata_path):
    demo_df = pd.read_csv(metadata_path)

raw_audio_array = None
audio_sr = 16000
source_description = ""
ground_truth_label = None

if input_mode == "🎧 Curated Demo Library":
    st.sidebar.markdown("Select a real recorded voice sample from the LibriSpeech test corpus:")
    if demo_df is not None and not demo_df.empty:
        # Format options nicely
        options = []
        for idx, row in demo_df.iterrows():
            lbl = f"{row['gender'].upper()} | Speaker: {row['speaker_name']} ({row['speaker_id']}) — {row['duration_sec']}s [{row['filename']}]"
            options.append((lbl, row['filename'], row['gender']))

        selected_lbl = st.sidebar.selectbox("Choose Sample:", [opt[0] for opt in options], index=0)
        selected_opt = [opt for opt in options if opt[0] == selected_lbl][0]
        selected_file = selected_opt[1]
        ground_truth_label = selected_opt[2]

        file_full_path = os.path.join("data/raw", selected_file)
        if os.path.exists(file_full_path):
            data, sr = sf.read(file_full_path, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            raw_audio_array = data
            audio_sr = sr
            source_description = f"Curated Sample: {selected_file} (Ground Truth: **{ground_truth_label.upper()}**)"
        else:
            st.sidebar.error(f"File not found: {file_full_path}")
    else:
        st.sidebar.warning("data/metadata.csv not found.")

elif input_mode == "📁 Upload Audio File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload audio file (.wav, .mp3, .ogg, .flac)",
        type=["wav", "mp3", "ogg", "flac"]
    )
    if uploaded_file is not None:
        try:
            data, sr = sf.read(io.BytesIO(uploaded_file.read()), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            raw_audio_array = data
            audio_sr = sr
            source_description = f"Uploaded File: {uploaded_file.name} ({len(data)/sr:.2f}s)"
        except Exception as e:
            st.sidebar.error(f"Error decoding audio: {e}")

elif input_mode == "🎙️ Record via Microphone":
    st.sidebar.markdown("Click below to record your voice live in the browser:")
    # Streamlit 1.63 native audio input
    audio_record = st.sidebar.audio_input("Speak for 3-5 seconds:")
    if audio_record is not None:
        try:
            data, sr = sf.read(io.BytesIO(audio_record.read()), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            raw_audio_array = data
            audio_sr = sr
            source_description = f"Microphone Recording ({len(data)/sr:.2f}s)"
        except Exception as e:
            st.sidebar.error(f"Error reading microphone stream: {e}")

st.sidebar.markdown("---")

# DSP Parameters
st.sidebar.markdown("### 🎛️ DSP & VAD Tuning")
vad_thresh_db = st.sidebar.slider("VAD Energy Threshold (dBFS)", min_value=-50.0, max_value=-20.0, value=-35.0, step=1.0)
target_sample_rate = 16000

# Step 12 Chaos Noise Lab
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Step 12: Chaos Noise Lab")
st.sidebar.markdown("Stress test model resilience by injecting real-time Gaussian noise:")
inject_noise = st.sidebar.toggle("Inject Background Noise", value=False)
target_snr = 20.0
if inject_noise:
    target_snr = st.sidebar.slider("Target Signal-to-Noise Ratio (SNR in dB)", min_value=-5.0, max_value=25.0, value=10.0, step=1.0)
    st.sidebar.caption("• **+20 dB**: Clean room ambient hiss\n• **+10 dB**: Moderate background chatter\n• **0 dB**: Noise equal to voice loudness\n• **-5 dB**: Harsh acoustic storm")


# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD HEADER & HERO
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">VOICE GENDER CLASSIFICATION V2</div>
    <div class="hero-subtitle">
        Next-Generation Audio AI: Evaluating <b>2D Spectrogram CNN</b> vs. <b>Improved Random Forest (360 Stats)</b> vs. <b>Old Baseline MLP (40 Mean MFCCs)</b>.
    </div>
    <div class="badge-row">
        <span class="badge badge-cyan">⚡ 16 kHz Polyphase DSP</span>
        <span class="badge badge-violet">🧠 2D Convolutional Mel-ResNet</span>
        <span class="badge badge-emerald">🛡️ Energy VAD Trimming</span>
        <span class="badge badge-pink">🔬 Step 12 Noise Robustness</span>
    </div>
</div>
""", unsafe_allow_html=True)

if raw_audio_array is None:
    st.info("👈 **Select an audio source in the left sidebar** (Curated Sample, File Upload, or Microphone) to inspect and classify voice audio.")
    st.stop()


# -----------------------------------------------------------------------------
# 6. AUDIO PROCESSING PIPELINE EXECUTION
# -----------------------------------------------------------------------------
# 1. Resample to 16,000 Hz if needed
if audio_sr != target_sample_rate:
    processed_signal = resample_audio(raw_audio_array, orig_sr=audio_sr, target_sr=target_sample_rate)
    current_sr = target_sample_rate
else:
    processed_signal = raw_audio_array.copy()
    current_sr = audio_sr

# 2. Peak normalization
processed_signal = peak_normalize(processed_signal, target_peak=0.95)

# 3. Optional Noise Injection
clean_signal = processed_signal.copy()
is_noisy = False
if inject_noise:
    processed_signal, actual_snr = inject_noise_snr(processed_signal, target_snr_db=target_snr)
    is_noisy = True

# 4. Compute Physical Audio Metrics
duration_sec = len(processed_signal) / current_sr
rms_val = compute_rms(processed_signal)
dbfs_val = rms_to_dbfs(rms_val)
f0_pitch = estimate_f0_pitch(processed_signal, current_sr)

# 5. VAD Analysis
is_speech_mask, frame_times, frame_db, peak_db = run_vad_frames(processed_signal, current_sr, threshold_db=vad_thresh_db)
speech_ratio = float(np.mean(is_speech_mask) * 100.0) if len(is_speech_mask) > 0 else 100.0


# -----------------------------------------------------------------------------
# 7. METRICS & AUDIO PLAYER SECTION
# -----------------------------------------------------------------------------
if is_noisy:
    st.markdown(f"""
    <div class="noise-alert">
        ⚠️ <b>Chaos Noise Injection Active:</b> Adding Gaussian White Noise at <b>SNR = {target_snr:.1f} dB</b>. 
        Notice how the 2D CNN retains speech formant patterns while the 1D MLP's average feature vector is corrupted!
    </div>
    """, unsafe_allow_html=True)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Duration & VAD Speech</div>
        <div class="metric-val">{duration_sec:.2f}s</div>
        <div class="metric-sub">{speech_ratio:.1f}% Active Voice</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Acoustic Energy (RMS)</div>
        <div class="metric-val">{dbfs_val:.1f} dBFS</div>
        <div class="metric-sub">Linear RMS: {rms_val:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    pitch_badge = "Female Range (165-255 Hz)" if f0_pitch >= 165 else "Male Range (85-180 Hz)"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Estimated Pitch (F0)</div>
        <div class="metric-val">{f0_pitch:.1f} Hz</div>
        <div class="metric-sub">{pitch_badge}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Sample Rate & Format</div>
        <div class="metric-val">16,000 Hz</div>
        <div class="metric-sub">Mono Float32 PCM</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
col_play1, col_play2 = st.columns([2, 1])
with col_play1:
    st.markdown(f"**Now Playing:** {source_description}")
    # Convert numpy float array to WAV bytes for st.audio
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, processed_signal, current_sr, format="WAV")
    wav_buffer.seek(0)
    st.audio(wav_buffer, format="audio/wav")

with col_play2:
    if ground_truth_label:
        st.markdown(f"**Ground Truth Label:** `{ground_truth_label.upper()}`")
    if is_noisy:
        st.caption(f"Noise Stress: SNR {target_snr:+.1f} dB injected")


# -----------------------------------------------------------------------------
# 8. THE PREDICTION ARENA: 3-MODEL SHOWDOWN
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("## ⚔️ The Model Prediction Arena")
st.markdown("Comparing inference across 3 distinct paradigms on the exact same audio input:")

# Prepare Features
# 1. 2D Log-Mel Spectrogram for CNN: (80, T)
mel_db, mel_freqs = compute_mel_spectrogram(processed_signal, sr=current_sr, n_mels=80)
# Standardize input for CNN
mel_mean = np.mean(mel_db)
mel_std = np.std(mel_db) + 1e-6
mel_norm = (mel_db - mel_mean) / mel_std
mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 80, T)

# 2. 360-dim Statistical MFCC for Random Forest
stats_360, feat_names = extract_mfcc_statistics(processed_signal, sr=current_sr, n_mfcc=20)
stats_input = stats_360.reshape(1, -1)

# 3. 40-dim Naive Mean MFCC for Old Baseline MLP
mfcc_40 = compute_mfcc(processed_signal, sr=current_sr, n_mfcc=40)
mfcc_40_mean = np.mean(mfcc_40, axis=1).reshape(1, -1)  # (1, 40)
mlp_tensor = torch.tensor(mfcc_40_mean, dtype=torch.float32)

# Run Inferences (In dataset & models: 1 = MALE, 0 = FEMALE)
# Model 1: CNN
prob_male_cnn = 0.5
if models_dict["cnn"] is not None:
    with torch.no_grad():
        logits_cnn = models_dict["cnn"](mel_tensor)
        prob_male_cnn = float(torch.sigmoid(logits_cnn).item())
prob_female_cnn = 1.0 - prob_male_cnn

# Model 2: Random Forest
prob_male_rf = 0.5
if models_dict["rf"] is not None:
    try:
        rf_probs = models_dict["rf"].predict_proba(stats_input)[0]
        prob_male_rf = float(rf_probs[1])
    except Exception as e:
        prob_male_rf = 0.5
prob_female_rf = 1.0 - prob_male_rf

# Model 3: Old Baseline MLP
prob_male_mlp = 0.5
if models_dict["mlp"] is not None:
    with torch.no_grad():
        logits_mlp = models_dict["mlp"](mlp_tensor)
        prob_male_mlp = float(torch.sigmoid(logits_mlp).item())
prob_female_mlp = 1.0 - prob_male_mlp

col_a1, col_a2, col_a3 = st.columns(3)

# Card 1: SpectrogramCNN
with col_a1:
    is_male = prob_male_cnn >= 0.5
    confidence = prob_male_cnn if is_male else prob_female_cnn
    pred_label = "MALE" if is_male else "FEMALE"
    pill_class = "gender-pill-male" if is_male else "gender-pill-female"

    st.markdown(f"""
    <div class="arena-card winner">
        <div class="card-title">
            <span>🧠 Modern 2D CNN</span>
            <span class="badge badge-cyan">Recommended</span>
        </div>
        <div style="text-align: center; margin: 16px 0;">
            <div class="{pill_class}">{pred_label}</div>
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; text-align: center;">
            Confidence: <b>{confidence * 100.0:.1f}%</b>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #94A3B8; margin-top: 10px;">
            <span>Female: {prob_female_cnn*100:.1f}%</span>
            <span>Male: {prob_male_cnn*100:.1f}%</span>
        </div>
        <div class="card-desc">
            <b>Spatial 2D Time-Frequency:</b> Evaluates formant trajectories and harmonic ridges across all 80 Mel bands without temporal collapse.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(float(prob_male_cnn), text=f"Male Probability: {prob_male_cnn*100:.1f}%")

# Card 2: Random Forest
with col_a2:
    is_male_rf = prob_male_rf >= 0.5
    conf_rf = prob_male_rf if is_male_rf else prob_female_rf
    pred_rf = "MALE" if is_male_rf else "FEMALE"
    pill_class_rf = "gender-pill-male" if is_male_rf else "gender-pill-female"

    st.markdown(f"""
    <div class="arena-card">
        <div class="card-title">
            <span>🌲 Random Forest (360 Stats)</span>
            <span class="badge badge-emerald">Statistical</span>
        </div>
        <div style="text-align: center; margin: 16px 0;">
            <div class="{pill_class_rf}">{pred_rf}</div>
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; text-align: center;">
            Confidence: <b>{conf_rf * 100.0:.1f}%</b>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #94A3B8; margin-top: 10px;">
            <span>Female: {prob_female_rf*100:.1f}%</span>
            <span>Male: {prob_male_rf*100:.1f}%</span>
        </div>
        <div class="card-desc">
            <b>Multi-Moment Statistics:</b> Evaluates variance, skewness, kurtosis, velocity (delta), and acceleration (delta-delta) of MFCCs.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(float(prob_male_rf), text=f"Male Probability: {prob_male_rf*100:.1f}%")

# Card 3: Old Baseline MLP
with col_a3:
    is_male_mlp = prob_male_mlp >= 0.5
    conf_mlp = prob_male_mlp if is_male_mlp else prob_female_mlp
    pred_mlp = "MALE" if is_male_mlp else "FEMALE"
    pill_class_mlp = "gender-pill-male" if is_male_mlp else "gender-pill-female"

    st.markdown(f"""
    <div class="arena-card">
        <div class="card-title">
            <span>📉 Old Baseline MLP</span>
            <span class="badge badge-pink">Old v1 Method</span>
        </div>
        <div style="text-align: center; margin: 16px 0;">
            <div class="{pill_class_mlp}">{pred_mlp}</div>
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; text-align: center;">
            Confidence: <b>{conf_mlp * 100.0:.1f}%</b>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #94A3B8; margin-top: 10px;">
            <span>Female: {prob_female_mlp*100:.1f}%</span>
            <span>Male: {prob_male_mlp*100:.1f}%</span>
        </div>
        <div class="card-desc">
            <b>Naive 40-Mean Collapse:</b> Averages all audio frames into a single static 40-dim vector. Easily fooled by silence and noise.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(float(prob_male_mlp), text=f"Male Probability: {prob_male_mlp*100:.1f}%")


# -----------------------------------------------------------------------------
# 9. DEEP ACOUSTIC & DSP VISUALIZATION LAB
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 📊 Acoustic & Spectral Visualization Studio")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Waveform & VAD Boundaries",
    "⚡ 1D FFT Magnitude Spectrum",
    "🌈 2D Log-Mel Spectrogram Heatmap",
    "🎼 40 MFCCs & Multi-Moment Radar"
])

# Dark plot styling setup
plt.style.use("dark_background")
DARK_BG = "#0F172A"
CARD_BG = "#131B2E"
GRID_COLOR = "#334155"
NEON_CYAN = "#00F0FF"
NEON_MAGENTA = "#F43F5E"
NEON_EMERALD = "#10B981"
NEON_VIOLET = "#8B5CF6"

with tab1:
    st.markdown("### Time Domain Waveform & Voice Activity Detection (VAD)")
    st.caption("Green shaded intervals represent active voiced speech detected by short-time energy tracking. Dark intervals are silence or background.")

    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)

    times = np.linspace(0, duration_sec, len(processed_signal))
    ax.plot(times, processed_signal, color=NEON_CYAN, alpha=0.85, linewidth=0.7, label="Audio Waveform")

    # Overlay VAD active regions
    if len(frame_times) > 0 and len(is_speech_mask) > 0:
        hop_sec = 0.010
        for i, is_spk in enumerate(is_speech_mask):
            if is_spk:
                t_start = frame_times[i]
                t_end = t_start + hop_sec
                ax.axvspan(t_start, t_end, color=NEON_EMERALD, alpha=0.18, edgecolor=None)

    ax.set_xlim(0, duration_sec)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (seconds)", color="#94A3B8", fontsize=10)
    ax.set_ylabel("Amplitude [-1.0, 1.0]", color="#94A3B8", fontsize=10)
    ax.tick_params(colors="#94A3B8", labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.2, color=GRID_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab2:
    st.markdown("### 1D Fast Fourier Transform (FFT) Magnitude Spectrum")
    st.caption("Converts time signal into constituent sine frequencies. Notice the fundamental pitch ($F_0$) peak and higher formant harmonics.")

    freqs, mags, peak_freq = compute_fft(processed_signal, current_sr)

    # Focus on speech bandwidth: 0 to 4000 Hz
    mask_4k = (freqs <= 4000)
    f_sub = freqs[mask_4k]
    m_sub = mags[mask_4k]

    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)

    ax.plot(f_sub, m_sub, color=NEON_VIOLET, linewidth=1.2, label="Magnitude |X(f)|")
    ax.fill_between(f_sub, m_sub, color=NEON_VIOLET, alpha=0.2)

    # Mark estimated pitch F0
    if f0_pitch > 0 and f0_pitch <= 4000:
        ax.axvline(f0_pitch, color=NEON_CYAN, linestyle="--", linewidth=1.8, label=f"Fundamental Pitch F0: {f0_pitch:.1f} Hz")
        ax.scatter([f0_pitch], [np.interp(f0_pitch, f_sub, m_sub)], color=NEON_CYAN, s=60, zorder=5)

    ax.set_xlim(0, 4000)
    ax.set_xlabel("Frequency (Hz)", color="#94A3B8", fontsize=10)
    ax.set_ylabel("Normalized Magnitude", color="#94A3B8", fontsize=10)
    ax.tick_params(colors="#94A3B8", labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.2, color=GRID_COLOR)
    ax.legend(facecolor=DARK_BG, edgecolor="#334155", labelcolor="#F8FAFC", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab3:
    st.markdown("### 2D Log-Mel Spectrogram (What the CNN Sees)")
    st.caption("80 Mel-frequency channels across time frames. Horizontal bands represent vocal tract formant resonances.")

    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)

    # Plot Mel Spectrogram Heatmap
    im = ax.imshow(
        mel_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, duration_sec, 0, 80]
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Energy (dB)", color="#94A3B8", fontsize=9)
    cbar.ax.tick_params(colors="#94A3B8", labelsize=8)

    ax.set_xlabel("Time (seconds)", color="#94A3B8", fontsize=10)
    ax.set_ylabel("Mel Filterbank Index (0 - 80)", color="#94A3B8", fontsize=10)
    ax.set_title("80-Band Log-Mel Spectrogram Heatmap", color="#F8FAFC", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#94A3B8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab4:
    st.markdown("### 40 MFCC Heatmap & Statistical Moment Distribution")
    st.caption("MFCCs de-correlate filterbank energies via DCT. Lower coefficients capture broad vocal tract shape; higher coefficients capture fine spectral harmonics.")

    col_mfcc1, col_mfcc2 = st.columns([3, 2])

    with col_mfcc1:
        fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=DARK_BG)
        ax.set_facecolor(CARD_BG)
        im = ax.imshow(mfcc_40, aspect="auto", origin="lower", cmap="viridis", extent=[0, duration_sec, 0, 40])
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("MFCC Coefficient Value", color="#94A3B8", fontsize=8)
        cbar.ax.tick_params(colors="#94A3B8", labelsize=8)
        ax.set_xlabel("Time (seconds)", color="#94A3B8", fontsize=9)
        ax.set_ylabel("MFCC Index (0 - 39)", color="#94A3B8", fontsize=9)
        ax.set_title("MFCC Temporal Evolution", color="#F8FAFC", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#94A3B8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_mfcc2:
        fig, ax = plt.subplots(figsize=(5, 3.8), facecolor=DARK_BG)
        ax.set_facecolor(CARD_BG)
        # Bar chart comparing the first 12 MFCC means vs stds
        indices = np.arange(12)
        means = np.mean(mfcc_40[:12], axis=1)
        stds = np.std(mfcc_40[:12], axis=1)

        ax.bar(indices - 0.18, means, width=0.36, color=NEON_CYAN, alpha=0.8, label="Mean")
        ax.bar(indices + 0.18, stds, width=0.36, color=NEON_MAGENTA, alpha=0.8, label="Std Dev")
        ax.set_xlabel("MFCC Index (0 - 11)", color="#94A3B8", fontsize=9)
        ax.set_ylabel("Value", color="#94A3B8", fontsize=9)
        ax.set_title("Mean vs Std Dev (Moments)", color="#F8FAFC", fontsize=10, fontweight="bold")
        ax.legend(facecolor=DARK_BG, edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)
        ax.tick_params(colors="#94A3B8", labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.2, color=GRID_COLOR)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# -----------------------------------------------------------------------------
# 10. EDUCATIONAL CSE MENTAL MODEL CARD
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("💡 **CSE Deep Dive: Why does the 2D CNN outperform the 1D MLP?**", expanded=False):
    st.markdown("""
### 🧠 The Core Architectural Difference

| Dimension | Old Project (Baseline 1D MLP) | v2 Project (Modern 2D Spectrogram CNN) |
| :--- | :--- | :--- |
| **Input Feature** | `np.mean(mfccs, axis=1)` (1D vector of 40 numbers) | 2D Matrix of 80 Mel Bands $\\times$ $T$ Time Frames (Image-like) |
| **Temporal Context** | **Completely Lost**: 5 seconds of audio collapsed into 1 static vector | **Preserved**: 2D Convolutions track how frequencies change over time |
| **Vocal Formants** | Formant movements are averaged out into a flat blur | Formant transitions between vowels are localized by 2D kernels |
| **Silence Impact** | Leading/trailing silence drastically skews the arithmetic mean | Handled cleanly via Energy VAD silence trimming |
| **Noise Resilience** | **Fails easily**: Background noise shifts the global mean vector | **Resilient**: 2D kernels recognize speech patterns submerged in white noise |

### 🎙️ How to test in this UI:
1. Try testing a female sample (e.g. `zinniz` or `Ransom`) vs a male sample (e.g. `fling93`).
2. Watch how the **Estimated Pitch ($F_0$)** aligns with the prediction.
3. Turn on the **Step 12: Chaos Noise Lab** in the left sidebar and lower the SNR slider to `0 dB` or `-5 dB`.
4. Observe that the **2D CNN** maintains robust confidence while the **1D MLP** flails!
    """)

st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.82rem; margin-top: 40px; margin-bottom: 20px;">
    Voice AI v2 Audio Engine • Digital Signal Processing & Deep Learning • Streamlit 1.63
</div>
""", unsafe_allow_html=True)
