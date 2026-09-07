"""
Step 8: Reproduce Old MFCC -> Mean -> Dense NN Pipeline
Uses the exact 5,993 pre-extracted samples from voice_gender.csv
Trains the 4-layer Dense MLP architecture and reports full evaluation metrics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import OldBaselineMLP
from src.evaluation import compute_metrics


def parse_feature_string(s: str) -> np.ndarray:
    cleaned = s.replace("[", "").replace("]", "").replace("\n", " ").strip()
    return np.fromstring(cleaned, sep=" ", dtype=np.float32)


def main():
    print("=" * 80)
    print("AUDIO AI V2: STEP 8 - REPRODUCING OLD BASELINE (MFCC MEAN -> DENSE MLP)")
    print("=" * 80)
    
    csv_path = "VoiceBasedGenderDetection-NullClassInternshipTask/voice_gender.csv"
    print(f"\n1. Loading old dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    X = np.array([parse_feature_string(s) for s in df["feature"]], dtype=np.float32)
    y = df["class"].to_numpy(dtype=np.float32)
    
    print(f"   - Dataset Shape: {X.shape} (5993 audio files x 40 mean MFCCs)")
    print(f"   - Labels: {np.sum(y == 1):.0f} Males (1), {np.sum(y == 0):.0f} Females (0)")
    print(f"   - Imbalance: {np.mean(y == 1)*100:.1f}% Male, {np.mean(y == 0)*100:.1f}% Female")
    
    # Exact split from old notebook: test_size=0.2, random_state=0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    print(f"\n2. Train / Test Split (random shuffle, test_size=0.2):")
    print(f"   - Training samples:   {X_train.shape[0]}")
    print(f"   - Testing samples:    {X_test.shape[0]}")
    print(f"   [WARNING] Note: No speaker grouping was performed, risking speaker leakage!")

    # PyTorch DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).unsqueeze(1))
    
    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Model Setup
    torch.manual_seed(42)
    model = OldBaselineMLP(input_dim=40)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n3. Initialized Old Baseline MLP Architecture:")
    print(model)
    
    # Training Loop
    epochs = 40
    best_val_loss = float("inf")
    best_state = None
    
    print(f"\n4. Training for {epochs} epochs (Batch Size = {batch_size})...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        train_loss = total_loss / len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in test_loader:
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(bx)
        val_loss = val_loss / len(test_ds)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"   Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    # Load best weights
    model.load_state_dict(best_state)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_probs = model.predict_proba(torch.tensor(X_test)).numpy().ravel()
        
    metrics = compute_metrics(y_test, test_probs)
    
    print("\n" + "=" * 60)
    print("OLD BASELINE EVALUATION METRICS (Calculated):")
    print("-" * 60)
    print(f"  - Test Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  - Precision:      {metrics['precision']*100:.2f}%")
    print(f"  - Recall:         {metrics['recall']*100:.2f}%")
    print(f"  - F1-Score:       {metrics['f1']*100:.2f}%")
    print(f"  - ROC-AUC:        {metrics['roc_auc']:.4f}")
    print(f"  - Confusion Matrix:")
    print(f"    [[Female Correct (TN): {metrics['true_negatives_female']}, Female False Pos (FP): {metrics['false_positives_female_as_male']}],")
    print(f"     [Male False Neg (FN): {metrics['false_negatives_male_as_female']}, Male Correct (TP): {metrics['true_positives_male']}]]")

    # Save artifacts
    os.makedirs("VoiceGenderClassificationV2/output/models", exist_ok=True)
    metrics_file = "VoiceGenderClassificationV2/output/models/baseline_old_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
        
    torch.save(best_state, "VoiceGenderClassificationV2/output/models/baseline_old_mlp.pt")
    print(f"\nSaved metrics to: {metrics_file}")
    print("Saved model checkpoint to: VoiceGenderClassificationV2/output/models/baseline_old_mlp.pt")
    print("\nStep 8 completed successfully!")

if __name__ == "__main__":
    main()
