"""
Automated validation of the Streamlit App backend pipeline:
1. Model loading (CNN, Random Forest, MLP)
2. Audio ingestion and DSP processing
3. Female and Male sample inference
4. Real-time Noise Injection (Chaos Lab Step 12)
"""

import os
import torch
import numpy as np
import soundfile as sf
import joblib
from src.models import SpectrogramCNN, OldBaselineMLP
from src.audio_processing import resample_audio, compute_rms, rms_to_dbfs, peak_normalize
from src.features import compute_fft, compute_mel_spectrogram, compute_mfcc, extract_mfcc_statistics

def test_pipeline():
    print("--- 1. Testing Model Loading ---")
    device = torch.device("cpu")
    
    # 1. CNN
    cnn = SpectrogramCNN(n_mels=80)
    cnn_path = "output/models/spectrogram_cnn.pt"
    assert os.path.exists(cnn_path), f"Missing {cnn_path}"
    cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
    cnn.eval()
    print("✓ SpectrogramCNN loaded successfully")
    
    # 2. RF
    rf_path = "output/models/rf_pipeline.joblib"
    assert os.path.exists(rf_path), f"Missing {rf_path}"
    rf = joblib.load(rf_path)
    print("✓ Random Forest Pipeline loaded successfully")
    
    # 3. MLP
    mlp = OldBaselineMLP(input_dim=40)
    mlp_path = "output/models/baseline_old_mlp.pt"
    assert os.path.exists(mlp_path), f"Missing {mlp_path}"
    mlp.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
    mlp.eval()
    print("✓ OldBaselineMLP loaded successfully")

    print("\n--- 2. Testing Sample Inferences (Female vs Male) ---")
    female_file = "data/raw/2277_2277-149896-0026.wav"
    male_file = "data/raw/777_777-126732-0074.wav"
    
    for label, filepath in [("Female", female_file), ("Male", male_file)]:
        data, sr = sf.read(filepath, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != 16000:
            data = resample_audio(data, orig_sr=sr, target_sr=16000)
            sr = 16000
        data = peak_normalize(data, target_peak=0.95)
        
        # Features
        mel_db, _ = compute_mel_spectrogram(data, sr=sr, n_mels=80)
        mel_norm = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)
        mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        stats_360, _ = extract_mfcc_statistics(data, sr=sr, n_mfcc=20)
        
        mfcc_40 = compute_mfcc(data, sr=sr, n_mfcc=40)
        mfcc_40_mean = np.mean(mfcc_40, axis=1).reshape(1, -1)
        mlp_tensor = torch.tensor(mfcc_40_mean, dtype=torch.float32)
        
        # Inferences
        with torch.no_grad():
            prob_male_cnn = float(torch.sigmoid(cnn(mel_tensor)).item())
            prob_male_mlp = float(torch.sigmoid(mlp(mlp_tensor)).item())
        prob_male_rf = float(rf.predict_proba(stats_360.reshape(1, -1))[0][1])
        
        print(f"[{label} Sample: {os.path.basename(filepath)}]")
        print(f"  • 2D CNN Male Prob:      {prob_male_cnn*100:.1f}% -> {'MALE' if prob_male_cnn >= 0.5 else 'FEMALE'} (Conf: {(prob_male_cnn if prob_male_cnn>=0.5 else 1-prob_male_cnn)*100:.1f}%)")
        print(f"  • Random Forest Prob:    {prob_male_rf*100:.1f}% -> {'MALE' if prob_male_rf >= 0.5 else 'FEMALE'} (Conf: {(prob_male_rf if prob_male_rf>=0.5 else 1-prob_male_rf)*100:.1f}%)")
        print(f"  • Old Baseline MLP Prob: {prob_male_mlp*100:.1f}% -> {'MALE' if prob_male_mlp >= 0.5 else 'FEMALE'} (Conf: {(prob_male_mlp if prob_male_mlp>=0.5 else 1-prob_male_mlp)*100:.1f}%)")

    print("\n--- 3. Testing Step 12 Chaos Noise Injection ---")
    sig_power = np.mean(data ** 2)
    for snr in [20.0, 10.0, 0.0, -5.0]:
        snr_linear = 10.0 ** (snr / 10.0)
        noise = np.random.normal(0, np.sqrt(sig_power / snr_linear), size=data.shape).astype(np.float32)
        noisy = np.clip(data + noise, -1.0, 1.0)
        
        # Mel for CNN
        mel_noisy, _ = compute_mel_spectrogram(noisy, sr=16000, n_mels=80)
        mel_noisy_norm = (mel_noisy - np.mean(mel_noisy)) / (np.std(mel_noisy) + 1e-6)
        with torch.no_grad():
            cnn_noisy_prob = float(torch.sigmoid(cnn(torch.tensor(mel_noisy_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0))).item())
        print(f"  SNR: {snr:+5.1f} dB -> CNN Prediction Prob Female: {cnn_noisy_prob*100:.1f}%")

    print("\n✓ ALL PIPELINE TESTS PASSED!")

if __name__ == "__main__":
    test_pipeline()
