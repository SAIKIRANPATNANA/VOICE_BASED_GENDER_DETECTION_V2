"""
Models Module: Voice Gender Classification v2
Implements both baseline and modern neural and statistical models:
1. OldBaselineMLP: Replicating the old Keras MLP (40 inputs -> Dense 100/200/100 -> 1 output)
2. ClassicalML: Scikit-learn Logistic Regression, Random Forest, and MLP on MFCC stats
3. SpectrogramCNN: Modern 2D Convolutional Neural Network for Log-Mel Spectrograms
"""

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class OldBaselineMLP(nn.Module):
    """
    PyTorch replica of the old project's Keras Sequential MLP:
    Input: (40,) [Mean MFCCs]
    Dense(100) -> ReLU -> Dropout(0.5)
    Dense(200) -> ReLU -> Dropout(0.5)
    Dense(100) -> ReLU -> Dropout(0.5)
    Dense(1)   -> Sigmoid
    """
    def __init__(self, input_dim: int = 40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw logits (use with BCEWithLogitsLoss for numerical stability)
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def build_classical_models() -> dict[str, Pipeline]:
    """
    Builds enhanced classical ML pipelines with standard feature scaling.
    WHAT it does:
        Creates Logistic Regression, Random Forest, and MLP pipelines with z-score scaling.
    WHY it is needed:
        Comparing against simpler, interpretable classical models proves whether
        deep learning is truly adding value over solid statistical feature engineering.
    """
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1))
        ]),
        "ModernMLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                                 early_stopping=True, max_iter=300, random_state=42))
        ])
    }
    return models


class SpectrogramCNN(nn.Module):
    """
    Modern 2D Convolutional Neural Network for Log-Mel Spectrogram Classification.
    
    WHAT it does:
        Treats the 2D Log-Mel Spectrogram (n_mels x time_frames) as a 1-channel image.
        Uses 2D convolutions to scan localized time-frequency patches.
    WHY it is needed:
        Unlike the old project which collapsed all time frames into one average vector,
        the 2D CNN retains the entire 2D spectrogram and automatically learns
        vowel formant movements, harmonic structures, and time-frequency textures.
    Input Shape:
        (Batch_size, 1, n_mels, time_steps)
    Output:
        Logits of shape (Batch_size, 1)
    """
    def __init__(self, n_mels: int = 80):
        super().__init__()
        
        # Block 1: Feature map extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Halves both freq and time
        )
        
        # Block 2: Higher-level acoustic patterns
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: Deep representations
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Global Average Pooling over remaining time and frequency dimensions
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Linear Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, Mels, Time)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # shape: (B, 128, 1, 1)
        x = torch.flatten(x, 1)  # shape: (B, 128)
        logits = self.classifier(x)  # shape: (B, 1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))
