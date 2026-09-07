"""
Step 9: Improved Audio Modeling with MFCC Statistics + Classical ML & MLP
Extracts 360 statistical features (Mean, Std, Min, Max, Skew, Kurt for MFCCs + Deltas + Delta-Deltas)
Trains and benchmarks:
- Logistic Regression
- Random Forest Classifier (with Feature Importance analysis)
- Modern Multi-Layer Perceptron (MLP)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.audio_processing import load_audio, peak_normalize, energy_vad
from src.features import extract_mfcc_statistics
from src.models import build_classical_models
from src.evaluation import compute_metrics


def main():
    print("=" * 80)
    print("AUDIO AI V2: STEP 9 - MFCC MULTI-MOMENT STATISTICS & CLASSICAL ML BENCHMARK")
    print("=" * 80)
    
    metadata_path = "VoiceGenderClassificationV2/data/metadata.csv"
    df = pd.read_csv(metadata_path)
    
    features_cache = "VoiceGenderClassificationV2/data/processed/mfcc_stats_dataset.npz"
    os.makedirs("VoiceGenderClassificationV2/data/processed", exist_ok=True)
    
    if os.path.exists(features_cache):
        print(f"\n1. Loading pre-computed statistical features from cache: {features_cache}")
        cached = np.load(features_cache)
        X = cached["X"]
        y = cached["y"]
        feature_names = cached["feature_names"].tolist()
    else:
        print(f"\n1. Extracting MFCC Multi-Moment Statistics from {len(df)} audio files...")
        X_list = []
        y_list = []
        feature_names = None
        
        for idx, row in df.iterrows():
            audio_path = os.path.join("VoiceGenderClassificationV2/data/raw", row["filename"])
            sig, sr = load_audio(audio_path)
            sig = peak_normalize(sig, 0.95)
            sig, _, _ = energy_vad(sig, sr)
            
            # Extract 20 MFCCs + Deltas + Delta-Deltas with 6 moments each (360 features)
            feats, names = extract_mfcc_statistics(sig, sr, n_mfcc=20)
            if feature_names is None:
                feature_names = names
            X_list.append(feats)
            y_list.append(1 if row["gender"] == "male" else 0)
            
            if (idx + 1) % 20 == 0 or (idx + 1) == len(df):
                print(f"   Processed {idx+1}/{len(df)} files...")
                
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        np.savez_compressed(features_cache, X=X, y=y, feature_names=feature_names)
        print(f"   Saved feature matrix to: {features_cache}")
        
    print(f"\n   Extracted Feature Matrix Shape: {X.shape} (120 files x {X.shape[1]} statistical features)")
    print(f"   Classes: {np.sum(y == 1)} Males (1), {np.sum(y == 0)} Females (0) - Perfectly Balanced!")
    
    # Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # Build models
    models = build_classical_models()
    results = {}
    
    print("\n2. Training & Evaluating Models:")
    print("-" * 60)
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        preds_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, preds_proba)
        results[name] = metrics
        
        print(f"\n* Model: {name}")
        print(f"  - Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"  - Precision: {metrics['precision']*100:.2f}%")
        print(f"  - Recall:    {metrics['recall']*100:.2f}%")
        print(f"  - F1-Score:  {metrics['f1']*100:.2f}%")
        print(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  - Confusion: TN={metrics['true_negatives_female']}, FP={metrics['false_positives_female_as_male']}, "
              f"FN={metrics['false_negatives_male_as_female']}, TP={metrics['true_positives_male']}")

    # Save results
    metrics_file = "VoiceGenderClassificationV2/output/models/step9_classical_ml_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to: {metrics_file}")
    
    # Feature Importance Analysis (Random Forest)
    print("\n3. Feature Importance Analysis (Which acoustic statistics separate gender best?):")
    print("-" * 60)
    rf_model = models["RandomForest"].named_steps["clf"]
    importances = rf_model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:15]
    
    print("Top 15 Most Discriminative Features:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. {feature_names[idx]:<20} (Importance: {importances[idx]:.4f})")
        
    # Plot Feature Importances
    plt.figure(figsize=(10, 6))
    top_names = [feature_names[i] for i in top_indices[::-1]]
    top_scores = importances[top_indices[::-1]]
    plt.barh(range(len(top_names)), top_scores, color="#457B9D")
    plt.yticks(range(len(top_names)), top_names, fontsize=9)
    plt.xlabel("Random Forest Gini Feature Importance", fontsize=10, fontweight="bold")
    plt.title("Step 9: Top Discriminative MFCC Statistical Features for Gender", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3)
    
    fig_path = "VoiceGenderClassificationV2/output/figures/04_mfcc_feature_importance.png"
    plt.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nSaved feature importance plot to: {fig_path}")
    print("\nStep 9 completed successfully!")

if __name__ == "__main__":
    main()
