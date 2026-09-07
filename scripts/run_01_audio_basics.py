"""
Step 1, 2, 3: Audio Basics, Waveforms, Resampling & VAD
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.audio_processing import (
    get_audio_metadata,
    load_audio,
    compute_rms,
    rms_to_dbfs,
    peak_normalize,
    resample_audio,
    energy_vad
)

def main():
    print("=" * 80)
    print("AUDIO AI V2: STEP 1, 2, 3 - AUDIO BASICS, WAVEFORMS, RESAMPLING & VAD")
    print("=" * 80)
    
    metadata_path = "VoiceGenderClassificationV2/data/metadata.csv"
    df = pd.read_csv(metadata_path)
    
    # Pick one representative female sample and one male sample
    female_row = df[df["gender"] == "female"].iloc[0]
    male_row = df[df["gender"] == "male"].iloc[0]
    
    female_path = os.path.join("VoiceGenderClassificationV2/data/raw", female_row["filename"])
    male_path = os.path.join("VoiceGenderClassificationV2/data/raw", male_row["filename"])
    
    print("\n[STEP 1] AUDIO BASICS & PHYSICAL METADATA")
    print("-" * 60)
    for title, path, speaker, gender in [
        ("Female Speaker Sample", female_path, female_row["speaker_name"], female_row["speaker_id"]),
        ("Male Speaker Sample", male_path, male_row["speaker_name"], male_row["speaker_id"])
    ]:
        meta = get_audio_metadata(path)
        print(f"\n* {title} ({speaker}, ID: {gender}):")
        print(f"  - File Path:      {meta['file_path']}")
        print(f"  - Format/Subtype: {meta['format']} / {meta['subtype']}")
        print(f"  - Channels:       {meta['channels']} (1 = Mono, 2 = Stereo)")
        print(f"  - Sample Rate:    {meta['sample_rate']} Hz (Samples per second)")
        print(f"  - Duration:       {meta['duration_sec']:.3f} seconds")
        print(f"  - Total Samples:  {meta['total_samples']} samples (Calculated: duration * sr)")
        
        # Load audio into memory
        sig, sr = load_audio(path)
        peak = np.max(np.abs(sig))
        rms = compute_rms(sig)
        dbfs = rms_to_dbfs(rms)
        
        print(f"  - Raw Shape:      {sig.shape} (1D numpy float32 array)")
        print(f"  - Amplitude Range:[{np.min(sig):.4f}, {np.max(sig):.4f}]")
        print(f"  - Peak Amplitude: {peak:.4f}")
        print(f"  - RMS Energy:     {rms:.4f} (Effective continuous acoustic power)")
        print(f"  - Loudness:       {dbfs:.2f} dBFS (0 dBFS = digital saturation)")
        
        # Peak normalization demonstration
        norm_sig = peak_normalize(sig, target_peak=0.95)
        new_peak = np.max(np.abs(norm_sig))
        new_rms = compute_rms(norm_sig)
        new_dbfs = rms_to_dbfs(new_rms)
        print(f"  - After Peak Norm:Peak = {new_peak:.2f}, RMS = {new_rms:.4f}, Loudness = {new_dbfs:.2f} dBFS")

    print("\n" + "=" * 60)
    print("[STEP 3] RESAMPLING & VOICE ACTIVITY DETECTION (VAD)")
    print("-" * 60)
    
    # Load raw audio
    sig_f, sr_f = load_audio(female_path)
    sig_m, sr_m = load_audio(male_path)
    
    # Demonstrate Resampling (16 kHz -> 22.05 kHz like old project)
    target_sr = 22050
    resampled_f = resample_audio(sig_f, orig_sr=sr_f, target_sr=target_sr)
    print(f"* Resampling Female Audio from {sr_f} Hz -> {target_sr} Hz:")
    print(f"  - Original Length:  {len(sig_f)} samples ({len(sig_f)/sr_f:.3f} s)")
    print(f"  - Resampled Length: {len(resampled_f)} samples ({len(resampled_f)/target_sr:.3f} s)")
    print(f"  - Ratio:            {len(resampled_f)/len(sig_f):.4f} (matches {target_sr}/{sr_f} = {target_sr/sr_f:.4f})")
    
    # Apply VAD & Silence Trimming
    trimmed_f, mask_f, vad_info_f = energy_vad(sig_f, sr_f)
    trimmed_m, mask_m, vad_info_m = energy_vad(sig_m, sr_m)
    
    print("\n* VAD Silence Trimming Results:")
    print(f"  - Female: Original = {vad_info_f['orig_duration_sec']:.2f}s -> Trimmed = {vad_info_f['trimmed_duration_sec']:.2f}s (Silence Removed: {vad_info_f['silence_ratio']*100:.1f}%)")
    print(f"  - Male:   Original = {vad_info_m['orig_duration_sec']:.2f}s -> Trimmed = {vad_info_m['trimmed_duration_sec']:.2f}s (Silence Removed: {vad_info_m['silence_ratio']*100:.1f}%)")

    # [STEP 2] VISUALIZATION: Waveform, RMS, and VAD Boundaries
    print("\n" + "=" * 60)
    print("[STEP 2] GENERATING EDUCATIONAL VISUALIZATION")
    print("-" * 60)
    
    os.makedirs("VoiceGenderClassificationV2/output/figures", exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex="col")
    plt.subplots_adjust(hspace=0.35, wspace=0.2)
    
    for col, (title, sig, sr, vad_info, color) in enumerate([
        ("Female Speech (ID: 2277)", sig_f, sr_f, vad_info_f, "#E63946"),
        ("Male Speech (ID: 174)", sig_m, sr_m, vad_info_m, "#1D3557")
    ]):
        time_axis = np.linspace(0, len(sig) / sr, len(sig))
        
        # 1. Raw Waveform
        axes[0, col].plot(time_axis, sig, color=color, alpha=0.8, lw=0.6)
        axes[0, col].set_title(f"{title}\nRaw Time-Domain Waveform x[n]", fontsize=11, fontweight="bold")
        axes[0, col].set_ylabel("Amplitude [-1.0, 1.0]")
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].set_ylim([-1.05, 1.05])
        
        # 2. VAD Detection & Speech Boundaries
        axes[1, col].plot(time_axis, sig, color="gray", alpha=0.4, lw=0.5, label="Waveform")
        axes[1, col].axvspan(vad_info["start_time_sec"], vad_info["end_time_sec"], 
                             color="green", alpha=0.2, label=f"Active Speech ({vad_info['trimmed_duration_sec']:.2f}s)")
        axes[1, col].axvline(vad_info["start_time_sec"], color="green", linestyle="--", lw=1.5)
        axes[1, col].axvline(vad_info["end_time_sec"], color="green", linestyle="--", lw=1.5)
        axes[1, col].set_title("Voice Activity Detection (Energy VAD)", fontsize=11, fontweight="bold")
        axes[1, col].set_ylabel("Amplitude")
        axes[1, col].legend(loc="upper right", fontsize=8)
        axes[1, col].grid(True, alpha=0.3)
        
        # 3. Zoomed 50ms Periodicity (Glottal Pulses & Fundamental Period T0)
        # Select an active vowel segment in the center
        mid_time = (vad_info["start_time_sec"] + vad_info["end_time_sec"]) / 2.0
        zoom_start = int((mid_time) * sr)
        zoom_len = int(0.04 * sr) # 40 ms window
        zoom_sig = sig[zoom_start:zoom_start + zoom_len]
        zoom_t = np.linspace(0, 40, len(zoom_sig)) # in milliseconds
        
        axes[2, col].plot(zoom_t, zoom_sig, color=color, lw=1.5, marker=".", markersize=3)
        axes[2, col].set_title("Vocal Fold Periodicity (Zoomed 40ms Vowel)", fontsize=11, fontweight="bold")
        axes[2, col].set_xlabel("Time (milliseconds)")
        axes[2, col].set_ylabel("Amplitude")
        axes[2, col].grid(True, alpha=0.3)
        
    out_fig = "VoiceGenderClassificationV2/output/figures/01_waveform_and_vad.png"
    plt.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to: {out_fig}")
    print("\nStep 1, 2, 3 completed successfully!")

if __name__ == "__main__":
    main()
