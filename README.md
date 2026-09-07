# 🎙️ Voice Gender Classification v2 (Modern Audio AI Pipeline)

A modern, educational, and modular Audio AI project modernizing the original internship proof-of-concept (`VoiceBasedGenderDetection-NullClassInternshipTask`).

Instead of treating audio as a black-box tabular problem (taking the temporal mean of MFCCs and feeding it into an MLP), this project teaches digital signal processing (DSP), acoustic psychoacoustics, spectral representations, multi-moment feature engineering, 2D Convolutional Neural Networks on spectrograms, speaker leakage prevention, and noise robustness analysis.

---

## 📁 Repository Structure

```
AUDIO AI/
├── VoiceBasedGenderDetection-NullClassInternshipTask/      # Old project (preserved as baseline)
│   ├── main.ipynb
│   ├── main.py
│   └── voice_gender.csv                                   # 5,993 pre-extracted 40-MFCC vectors
│
└── VoiceGenderClassificationV2/                           # Modern Educational Project (v2)
    ├── data/
    │   ├── raw/                                           # 120 curated multi-speaker speech WAV files (6 Male, 6 Female)
    │   ├── processed/                                     # Cached feature arrays (.npz)
    │   └── metadata.csv                                   # Clip metadata (speaker ID, gender, duration, sample rate)
    ├── docs/
    │   └── audio_fundamentals.md                          # Complete Audio AI Theory & Concepts Handbook
    ├── notebooks/                                         # Google Colab-ready Jupyter Notebooks
    │   ├── 01_audio_basics_and_preprocessing.ipynb        # Steps 1–3: Basics, Waveforms, VAD
    │   ├── 02_spectral_representations_fft_stft_mel.ipynb # Steps 4–6: FFT, STFT, Mel Spectrograms
    │   ├── 03_mfcc_and_feature_engineering.ipynb          # Steps 7–9: MFCCs, Old MLP, 360-Dim Stats ML
    │   ├── 04_cnn_spectrogram_classification.ipynb        # Step 10: 2D Spectrogram CNN vs 1D MLP
    │   ├── 05_speaker_split_and_noise_robustness.ipynb     # Steps 11–12: Speaker Leakage & Noise (SNR)
    │   └── Voice_Gender_Classification_v2_Master.ipynb    # All 12 Steps in ONE Master Colab Notebook!
    ├── src/                                               # Modular Python Library
    │   ├── __init__.py
    │   ├── audio_processing.py                            # Loading, RMS, Peak Norm, Resampling, Energy VAD
    │   ├── features.py                                    # FFT, STFT, Mel Spectrogram, MFCC + Stats
    │   ├── models.py                                      # OldBaselineMLP, ClassicalML, SpectrogramCNN
    │   └── evaluation.py                                  # Metrics suite, Group Splitting, SNR Noise Harness
    ├── output/
    │   ├── figures/                                       # High-resolution educational plots
    │   │   ├── 01_waveform_and_vad.png                    # Waveforms, RMS & VAD speech boundaries
    │   │   ├── 02_spectral_representations.png            # FFT, STFT Trade-off, Mel Specs, MFCC Heatmaps
    │   │   └── 04_mfcc_feature_importance.png             # Top discriminative acoustic moments (Gini)
    │   └── models/                                        # Checkpoints & evaluation metric logs
    └── requirements.txt                                   # Python dependencies
```

---

## 🚀 How to Run on Google Colab

Every notebook in `notebooks/` is fully self-contained and pre-configured for **Google Colab**:
1. Open [Google Colab](https://colab.research.google.com).
2. Click **Upload** and upload any notebook from `notebooks/`, or upload `Voice_Gender_Classification_v2_Master.ipynb`.
3. Select **Runtime $\to$ Change runtime type $\to$ T4 GPU** (or free CPU).
4. Run each cell sequentially! Dependencies and audio synthesis/data fallbacks are automatically handled if running outside the local repository.

---

## 📚 Concepts Covered Step-by-Step

| Step | Concept | Software / CSE Mental Model |
|---|---|---|
| **1** | **Audio Basics** | Audio as a 1D float array, Sample Rate as Video FPS ($f_s$), Nyquist limit as the video wagon-wheel glitch, RMS average energy, and Peak MinMax normalization ($0.95$). |
| **2** | **Waveforms** | Time-domain 1D array $x[n]$, instantaneous pressure values, and vocal cord clock cycle ($T_0$). |
| **3** | **Preprocessing & VAD** | Stereo to mono channel averaging, 1D array resampling (image resize equivalent), and Voice Activity Detection as string `.strip()` for audio. |
| **4** | **FFT** | Decompiling a mixed audio thread into discrete frequency notes; detecting human pitch ($F_0$) via `argmax` in the $80-300\text{ Hz}$ range. |
| **5** | **STFT** | LeetCode Sliding Window algorithm slicing 1D audio into a 2D grayscale image matrix (Spectrogram); window size trade-off (profiler sampling granularity). |
| **6** | **Mel Spectrogram** | Non-linear perceptual bucketing; matrix multiplication ($\mathbf{W}_{\text{mel}} \times \mathbf{S}$) compressing 513 linear rows into 80 human-perceptual rows. |
| **7** | **MFCC** | Lossy JPEG compression for voice; using DCT to isolate smooth vocal tract formants ($C_1-C_{13}$) from mic noise ($C_{14}+$); why ECE folks coined the word "Cepstrum". |
| **8** | **Old Baseline Reproduction** | Faithful reproduction of the original 40 MFCC mean $\to$ 4-layer Dense MLP on `voice_gender.csv` ($90.66\%$ test accuracy). |
| **9** | **Statistical Feature Engineering** | Overcoming "Temporal Collapse" (the movie color averaging bug) using 360 multi-moment features (Mean, Std, Min, Max, Skew, Kurtosis, Delta velocity) + Random Forest feature importances. |
| **10** | **2D Spectrogram CNN** | Treating Log-Mel Spectrograms as 2D images `(1, 80, Time)`, applying 2D convolutions with Global Average Pooling (GAP) without temporal flattening. |
| **11** | **Speaker Leakage Prevention** | Preventing train/test data contamination (data snooping) across multi-utterance speakers using strict `GroupShuffleSplit` (`GroupKFold`). |
| **12** | **Noise Robustness Benchmarking** | Chaos / packet noise testing by injecting Gaussian white noise across calibrated SNRs ($+20\text{ dB}$ down to $-5\text{ dB}$); why 2D CNN edge detectors beat 1D MLPs. |

## 🌐 Interactive Streamlit Web Application

Test and evaluate your trained models interactively through a vibrant, dark-themed cybernetic dashboard:

```bash
# From the project root:
PYTHONPATH=. .venv/bin/streamlit run app.py
```

### ✨ Features Included:
- **Audio Inputs**:
  1. 🎙️ **Microphone (Live Browser Recording)**: Speak directly into the browser via native `st.audio_input`.
  2. 📁 **Custom File Upload**: Drag-and-drop any `.wav`, `.mp3`, `.ogg`, or `.flac` voice recording.
  3. 🎧 **Curated Demo Library**: Select real multi-speaker voice samples (Female & Male) with ground-truth verification.
- **Acoustic Scorecard**: Real-time duration, VAD speech %, RMS energy (dBFS), and Fundamental Pitch ($F_0$) estimation.
- **The Prediction Arena**: Side-by-side inference comparing:
  - 🧠 **Modern 2D CNN** (80-band Log-Mel Spectrogram)
  - 🌲 **Random Forest Pipeline** (360-dim multi-moment stats)
  - 📉 **Old Baseline MLP** (40-dim mean MFCC)
- **Spectral Visualization Studio**:
  - Time Waveform with green VAD active speech overlay.
  - 1D FFT Magnitude Spectrum with pitch $F_0$ callout.
  - 2D Log-Mel Spectrogram (80 Mel channels) heatmap.
  - 40 MFCC temporal heatmap & statistical moment breakdown.
- **🧪 Step 12 Chaos Noise Lab**: Real-time slider to inject Gaussian white noise (from $+20\text{ dB}$ down to $-5\text{ dB}$ SNR) to visually and empirically experience why the 2D CNN withstands acoustic noise while the 1D MLP collapses!

---

## 📖 Theory & Reference Documentation

Read [`docs/audio_fundamentals.md`](file:///home/user/snap/Downloads/AUDIO%20AI%20/VoiceGenderClassificationV2/docs/audio_fundamentals.md) for the complete **Audio AI Handbook for Software Engineers (CSE Guide)** with diagrams, memory layouts, Python code snippets, and intuitive analogies.

