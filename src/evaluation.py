"""
Evaluation & Robustness Module: Voice Gender Classification v2
Implements scientific controls, metrics, and robustness benchmarks:
1. Classification Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
2. Speaker-Independent Data Splitting (Group Splitting to prevent speaker leakage)
3. Noise Robustness Test Harness: Additive noise at calibrated SNR levels (+20dB to -5dB)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def compute_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Computes a comprehensive suite of classification metrics.
    
    WHAT it does:
        Evaluates predictions using Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
    WHY it is needed:
        The old project only reported raw accuracy on an imbalanced dataset (61.4% male).
        Accuracy alone hides asymmetric error rates between genders.
    WHAT the output means:
        Returns dictionary of all metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_probs = np.asarray(y_pred_probs, dtype=float)
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_probs)) if len(np.unique(y_true)) > 1 else 0.5,
        "true_negatives_female": int(tn),
        "false_positives_female_as_male": int(fp),
        "false_negatives_male_as_female": int(fn),
        "true_positives_male": int(tp),
        "confusion_matrix": cm.tolist()
    }
    return metrics


def split_by_speaker(
    df: pd.DataFrame,
    speaker_col: str = "speaker_id",
    test_speakers: list[str] = None,
    val_speakers: list[str] = None,
    test_ratio: float = 0.2,
    val_ratio: float = 0.15,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions dataset into train, validation, and test sets with DISJOINT speakers.
    
    WHAT it does:
        Ensures all utterances from any given speaker appear in ONLY ONE of the splits
        (train, val, or test).
    WHY it is needed:
        Prevents "Speaker Leakage" (Data Snooping).
        When the same speaker's clips are in both train and test sets, models can achieve
        artificially inflated accuracy simply by memorizing vocal timbre, recording environment,
        or microphone acoustics rather than true gender characteristics.
    WHAT the output means:
        train_df, val_df, test_df where set(train.speakers) & set(test.speakers) == empty.
    """
    unique_speakers = df[speaker_col].unique()
    rng = np.random.default_rng(random_state)
    shuffled_speakers = rng.permutation(unique_speakers)
    
    if test_speakers is None:
        n_test = max(1, int(round(len(unique_speakers) * test_ratio)))
        test_speakers = list(shuffled_speakers[:n_test])
        remaining = shuffled_speakers[n_test:]
    else:
        remaining = [s for s in unique_speakers if s not in test_speakers]
        
    if val_speakers is None:
        n_val = max(1, int(round(len(unique_speakers) * val_ratio)))
        val_speakers = list(remaining[:n_val])
        train_speakers = list(remaining[n_val:])
    else:
        train_speakers = [s for s in remaining if s not in val_speakers]
        
    train_df = df[df[speaker_col].isin(train_speakers)].reset_index(drop=True)
    val_df = df[df[speaker_col].isin(val_speakers)].reset_index(drop=True)
    test_df = df[df[speaker_col].isin(test_speakers)].reset_index(drop=True)
    
    return train_df, val_df, test_df


def add_noise_at_snr(signal: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """
    Adds Gaussian white noise to an audio signal at a specified Signal-to-Noise Ratio (SNR).
    
    Mathematical Derivation:
        SNR (dB) = 10 * log10( P_signal / P_noise )
        => P_noise = P_signal / (10^(SNR / 10))
        => noise_std = sqrt( P_noise )
        noisy_signal[n] = signal[n] + Gaussian(0, noise_std)
    
    WHAT it does:
        Injects mathematically calibrated noise into the audio waveform.
    WHY it is needed:
        Tests acoustic robustness. Real-world microphones capture street noise, HVAC hum,
        and reverberation. A fragile model collapses when noise is added; a robust model maintains
        accurate formant and pitch detection.
    WHAT the output means:
        noisy_signal: Audio with the exact specified SNR in dB.
        +20 dB: Barely noticeable background hiss.
        +10 dB: Moderate noise (office/room background).
        0 dB: Noise has equal power to speech!
        -5 dB: Noise is louder than the speaker's voice.
    """
    signal_power = np.mean(signal ** 2)
    if signal_power <= 1e-12:
        return signal.copy()
        
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)
    
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=signal.shape)
    
    noisy_signal = signal + noise
    return noisy_signal.astype(np.float32)
