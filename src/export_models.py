"""
Export Trained Models for the Streamlit UI
Pre-trains and exports:
1. Modern 2D SpectrogramCNN (mel spectrograms -> binary gender)
2. Random Forest Pipeline (360 multi-moment MFCC features -> binary gender)
3. Old Baseline MLP (40 MFCC mean features -> binary gender)
"""

import os
import torch
import torch.nn as nn
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.models import SpectrogramCNN, OldBaselineMLP

def main():
    os.makedirs("output/models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exporting models using device: {device}...")

    # -------------------------------------------------------------
    # 1. Train and export Random Forest on 360-dim MFCC stats
    # -------------------------------------------------------------
    stats_path = "data/processed/mfcc_stats_dataset.npz"
    if os.path.exists(stats_path):
        data = np.load(stats_path)
        X_stats, y_stats = data["X"], data["y"]
        print(f"Loaded MFCC stats: {X_stats.shape}")

        rf_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
        ])
        rf_pipeline.fit(X_stats, y_stats)
        rf_path = "output/models/rf_pipeline.joblib"
        joblib.dump(rf_pipeline, rf_path)
        print(f"Saved Random Forest Pipeline to {rf_path}")
    else:
        print(f"Warning: {stats_path} not found!")

    # -------------------------------------------------------------
    # 2. Train and export 2D SpectrogramCNN
    # -------------------------------------------------------------
    mel_path = "data/processed/mel_specs_dataset.npz"
    if os.path.exists(mel_path):
        mel_data = np.load(mel_path)
        specs, labels = mel_data["specs"], mel_data["labels"]
        print(f"Loaded Mel Spectrograms: {specs.shape}")

        torch.manual_seed(42)
        cnn = SpectrogramCNN(n_mels=80).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(cnn.parameters(), lr=0.001, weight_decay=1e-4)

        X_tensor = torch.tensor(specs, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        print("Training SpectrogramCNN...")
        cnn.train()
        for epoch in range(25):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(cnn(bx), by)
                loss.backward()
                optimizer.step()

        cnn_path = "output/models/spectrogram_cnn.pt"
        torch.save(cnn.state_dict(), cnn_path)
        print(f"Saved SpectrogramCNN weights to {cnn_path}")
    else:
        print(f"Warning: {mel_path} not found!")

    print("Model export complete!")

if __name__ == "__main__":
    main()
