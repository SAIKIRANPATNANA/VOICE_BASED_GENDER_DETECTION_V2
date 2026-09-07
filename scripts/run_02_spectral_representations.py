"""
Step 4, 5, 6, 7: Spectral Representations (FFT, STFT, Mel Spectrogram, MFCC)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.audio_processing import load_audio, peak_normalize, energy_vad
from src.features import (
    compute_fft,
    compute_stft,
    compute_mel_spectrogram,
    compute_mfcc
)

def main():
    print("=" * 80)
    print("AUDIO AI V2: STEP 4, 5, 6, 7 - SPECTRAL ANALYSIS (FFT, STFT, MEL, MFCC)")
    print("=" * 80)
    
    metadata_path = "VoiceGenderClassificationV2/data/metadata.csv"
    df = pd.read_csv(metadata_path)
    
    female_row = df[df["gender"] == "female"].iloc[0]
    male_row = df[df["gender"] == "male"].iloc[0]
    
    female_path = os.path.join("VoiceGenderClassificationV2/data/raw", female_row["filename"])
    male_path = os.path.join("VoiceGenderClassificationV2/data/raw", male_row["filename"])
    
    # Load and normalize
    sig_f, sr_f = load_audio(female_path)
    sig_m, sr_m = load_audio(male_path)
    
    sig_f = peak_normalize(sig_f, 0.95)
    sig_m = peak_normalize(sig_m, 0.95)
    
    # Trim with VAD
    sig_f, _, _ = energy_vad(sig_f, sr_f)
    sig_m, _, _ = energy_vad(sig_m, sr_m)
    
    # [STEP 4] FAST FOURIER TRANSFORM (FFT)
    print("\n[STEP 4] FAST FOURIER TRANSFORM (FFT): TIME -> FREQUENCY")
    print("-" * 60)
    # Extract 100ms stable vowel chunk for clean pitch analysis
    chunk_f = sig_f[int(1.0*sr_f):int(1.1*sr_f)]
    chunk_m = sig_m[int(1.0*sr_m):int(1.1*sr_m)]
    
    freqs_f, mags_f, peak_f0_f = compute_fft(chunk_f, sr_f)
    freqs_m, mags_m, peak_f0_m = compute_fft(chunk_m, sr_m)
    
    print(f"* Female Vowel Segment:")
    print(f"  - Length: {len(chunk_f)} samples (100 ms at {sr_f} Hz)")
    print(f"  - FFT Bins: {len(freqs_f)} complex frequency bins (0 to {sr_f/2} Hz)")
    print(f"  - Dominant Pitch (F0 proxy): {peak_f0_f:.1f} Hz (Typical Female: 165 - 255 Hz)")
    
    print(f"* Male Vowel Segment:")
    print(f"  - Length: {len(chunk_m)} samples (100 ms at {sr_m} Hz)")
    print(f"  - FFT Bins: {len(freqs_m)} complex frequency bins (0 to {sr_m/2} Hz)")
    print(f"  - Dominant Pitch (F0 proxy): {peak_f0_m:.1f} Hz (Typical Male: 85 - 180 Hz)")
    
    # [STEP 5] SHORT-TIME FOURIER TRANSFORM (STFT)
    print("\n[STEP 5] SHORT-TIME FOURIER TRANSFORM (STFT): TIME-FREQUENCY RESOLUTION")
    print("-" * 60)
    n_fft_narrow = 2048 # High frequency resolution
    n_fft_wide = 256    # High time resolution
    hop_length = 256
    
    stft_narrow_f, f_bins_n, t_bins_n = compute_stft(sig_f, sr_f, n_fft=n_fft_narrow, hop_length=hop_length)
    stft_wide_f, f_bins_w, t_bins_w = compute_stft(sig_f, sr_f, n_fft=n_fft_wide, hop_length=hop_length)
    
    print(f"* Narrow-band STFT (N_fft = {n_fft_narrow}, Hop = {hop_length}):")
    print(f"  - Shape: {stft_narrow_f.shape} (Freq Bins x Time Frames)")
    print(f"  - Frequency Resolution: {sr_f / n_fft_narrow:.2f} Hz per bin (Sharp harmonic lines)")
    print(f"  - Time Window: {n_fft_narrow / sr_f * 1000:.1f} ms")
    
    print(f"* Wide-band STFT (N_fft = {n_fft_wide}, Hop = {hop_length}):")
    print(f"  - Shape: {stft_wide_f.shape} (Freq Bins x Time Frames)")
    print(f"  - Frequency Resolution: {sr_f / n_fft_wide:.2f} Hz per bin (Coarse frequency)")
    print(f"  - Time Window: {n_fft_wide / sr_f * 1000:.1f} ms (Sharp glottal pulse timing)")
    
    # [STEP 6] MEL SPECTROGRAM
    print("\n[STEP 6] MEL SPECTROGRAM (PSYCHOACOUSTIC FILTER BANKS)")
    print("-" * 60)
    n_mels = 80
    mel_f, mel_freqs = compute_mel_spectrogram(sig_f, sr_f, n_mels=n_mels)
    mel_m, _ = compute_mel_spectrogram(sig_m, sr_m, n_mels=n_mels)
    
    print(f"* Log-Mel Spectrogram Dimensions:")
    print(f"  - Female Mel Shape: {mel_f.shape} ({n_mels} Mel frequency channels x {mel_f.shape[1]} Time frames)")
    print(f"  - Male Mel Shape:   {mel_m.shape} ({n_mels} Mel frequency channels x {mel_m.shape[1]} Time frames)")
    print(f"  - Lowest Mel Center Frequency:  {mel_freqs[0]:.1f} Hz")
    print(f"  - 40th Mel Center Frequency:    {mel_freqs[39]:.1f} Hz")
    print(f"  - Highest Mel Center Frequency: {mel_freqs[-1]:.1f} Hz")
    print("  -> Mel scale clusters filters tightly in 50-1000 Hz where voice pitch & formants live!")

    # [STEP 7] MFCC PIPELINE
    print("\n[STEP 7] MEL-FREQUENCY CEPSTRAL COEFFICIENTS (MFCC)")
    print("-" * 60)
    n_mfcc = 20
    mfcc_f = compute_mfcc(sig_f, sr_f, n_mfcc=n_mfcc)
    mfcc_m = compute_mfcc(sig_m, sr_m, n_mfcc=n_mfcc)
    
    print(f"* MFCC Matrix Shape: {mfcc_f.shape} ({n_mfcc} Cepstral Coefficients x {mfcc_f.shape[1]} Time Frames)")
    print(f"  - Coefficient C0: Mean = {np.mean(mfcc_f[0]):.2f} (Frame energy/loudness)")
    print(f"  - Coefficients C1-C13 (Vocal tract resonances): Captured per frame")
    print(f"  - Coefficients C14-C20 (Fine ripples): Captured per frame")

    # GENERATE EDUCATIONAL FIGURE
    print("\n" + "=" * 60)
    print("GENERATING COMPREHENSIVE SPECTRAL VISUALIZATION FIGURE")
    print("-" * 60)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.25)
    
    # Row 1: FFT Spectrum Comparison (Male vs Female)
    ax_fft_f = fig.add_subplot(gs[0, 0])
    ax_fft_m = fig.add_subplot(gs[0, 1])
    
    # Zoom in to 0 - 1500 Hz for speech
    plot_mask_f = freqs_f <= 1500
    plot_mask_m = freqs_m <= 1500
    
    ax_fft_f.plot(freqs_f[plot_mask_f], mags_f[plot_mask_f], color="#E63946", lw=1.2)
    ax_fft_f.axvline(peak_f0_f, color="darkred", linestyle="--", lw=1.5, label=f"F0: {peak_f0_f:.1f} Hz")
    ax_fft_f.set_title(f"Female FFT Magnitude Spectrum (Vowel)\nFundamental Pitch F0 ≈ {peak_f0_f:.1f} Hz", fontweight="bold", fontsize=10)
    ax_fft_f.set_xlabel("Frequency (Hz)")
    ax_fft_f.set_ylabel("Magnitude")
    ax_fft_f.legend(loc="upper right", fontsize=9)
    ax_fft_f.grid(True, alpha=0.3)
    
    ax_fft_m.plot(freqs_m[plot_mask_m], mags_m[plot_mask_m], color="#1D3557", lw=1.2)
    ax_fft_m.axvline(peak_f0_m, color="navy", linestyle="--", lw=1.5, label=f"F0: {peak_f0_m:.1f} Hz")
    ax_fft_m.set_title(f"Male FFT Magnitude Spectrum (Vowel)\nFundamental Pitch F0 ≈ {peak_f0_m:.1f} Hz", fontweight="bold", fontsize=10)
    ax_fft_m.set_xlabel("Frequency (Hz)")
    ax_fft_m.set_ylabel("Magnitude")
    ax_fft_m.legend(loc="upper right", fontsize=9)
    ax_fft_m.grid(True, alpha=0.3)
    
    # Row 2: STFT Uncertainty Tradeoff (Narrow-band vs Wide-band)
    ax_stft_n = fig.add_subplot(gs[1, 0])
    ax_stft_w = fig.add_subplot(gs[1, 1])
    
    im_n = ax_stft_n.imshow(stft_narrow_f, aspect="auto", origin="lower", cmap="magma",
                            extent=[0, t_bins_n[-1], 0, sr_f / 2])
    ax_stft_n.set_ylim([0, 4000])
    ax_stft_n.set_title("STFT: Narrow-Band (N_fft=2048)\nHigh Frequency Resolution (Horizontal Harmonics)", fontweight="bold", fontsize=10)
    ax_stft_n.set_ylabel("Frequency (Hz)")
    plt.colorbar(im_n, ax=ax_stft_n, format="%+2.0f dB")
    
    im_w = ax_stft_w.imshow(stft_wide_f, aspect="auto", origin="lower", cmap="magma",
                            extent=[0, t_bins_w[-1], 0, sr_f / 2])
    ax_stft_w.set_ylim([0, 4000])
    ax_stft_w.set_title("STFT: Wide-Band (N_fft=256)\nHigh Time Resolution (Vertical Glottal Pulses)", fontweight="bold", fontsize=10)
    ax_stft_w.set_ylabel("Frequency (Hz)")
    plt.colorbar(im_w, ax=ax_stft_w, format="%+2.0f dB")
    
    # Row 3: Log-Mel Spectrogram (Female vs Male)
    ax_mel_f = fig.add_subplot(gs[2, 0])
    ax_mel_m = fig.add_subplot(gs[2, 1])
    
    im_mf = ax_mel_f.imshow(mel_f, aspect="auto", origin="lower", cmap="viridis")
    ax_mel_f.set_title("Female 80-Band Log-Mel Spectrogram\n(Higher Formant Bands)", fontweight="bold", fontsize=10)
    ax_mel_f.set_ylabel("Mel Frequency Bands")
    plt.colorbar(im_mf, ax=ax_mel_f, format="%+2.0f dB")
    
    im_mm = ax_mel_m.imshow(mel_m, aspect="auto", origin="lower", cmap="viridis")
    ax_mel_m.set_title("Male 80-Band Log-Mel Spectrogram\n(Lower Formant Bands)", fontweight="bold", fontsize=10)
    ax_mel_m.set_ylabel("Mel Frequency Bands")
    plt.colorbar(im_mm, ax=ax_mel_m, format="%+2.0f dB")
    
    # Row 4: MFCC Heatmap (Female vs Male)
    ax_mfcc_f = fig.add_subplot(gs[3, 0])
    ax_mfcc_m = fig.add_subplot(gs[3, 1])
    
    im_cf = ax_mfcc_f.imshow(mfcc_f, aspect="auto", origin="lower", cmap="coolwarm")
    ax_mfcc_f.set_title("Female MFCCs (20 Coefficients over Time)\nLow Quefrencies (1-13) Capture Vocal Tract Shape", fontweight="bold", fontsize=10)
    ax_mfcc_f.set_xlabel("Time Frames")
    ax_mfcc_f.set_ylabel("MFCC Index")
    plt.colorbar(im_cf, ax=ax_mfcc_f)
    
    im_cm = ax_mfcc_m.imshow(mfcc_m, aspect="auto", origin="lower", cmap="coolwarm")
    ax_mfcc_m.set_title("Male MFCCs (20 Coefficients over Time)\nDistinct Resonances Across Quefrency Axis", fontweight="bold", fontsize=10)
    ax_mfcc_m.set_xlabel("Time Frames")
    ax_mfcc_m.set_ylabel("MFCC Index")
    plt.colorbar(im_cm, ax=ax_mfcc_m)
    
    out_fig = "VoiceGenderClassificationV2/output/figures/02_spectral_representations.png"
    plt.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to: {out_fig}")
    print("\nStep 4, 5, 6, 7 completed successfully!")

if __name__ == "__main__":
    main()
