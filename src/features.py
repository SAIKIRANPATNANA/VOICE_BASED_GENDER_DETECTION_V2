"""
Audio Features Extraction Module: Voice Gender Classification v2
Implements spectral representations and audio feature engineering:
- FFT (Fast Fourier Transform): 1D frequency spectrum & dominant frequency
- STFT (Short-Time Fourier Transform): Time-Frequency analysis & trade-offs
- Mel Spectrogram: Psychoacoustic human pitch perception filter banks
- MFCC (Mel-Frequency Cepstral Coefficients): Full pipeline & Quefrency analysis
- MFCC Multi-Moment Statistics: Mean, Std, Skew, Kurtosis, Deltas (velocity & acceleration)
"""

import numpy as np
import scipy.fft
import scipy.stats
import librosa


def compute_fft(signal: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes Fast Fourier Transform (FFT) on a 1D audio signal.
    
    Mathematical Concept:
        X[k] = sum_{n=0}^{N-1} x[n] * exp(-j * 2 * pi * k * n / N)
    
    WHAT it does:
        Converts the discrete time-domain signal into its frequency components.
        Computes one-sided magnitude spectrum and frequency bins.
    WHY it is needed:
        Time waveforms show "when" energy happens, but FFT reveals "what pitches"
        and frequencies are present. In speech, male voices have a lower fundamental
        frequency (F0 typically 85-180 Hz) than female voices (typically 165-255 Hz).
    WHAT the output means:
        freqs: Frequency bins in Hz (from 0 to sr / 2, Nyquist frequency).
        magnitudes: Magnitude of each frequency component (|X[k]|).
        peak_freq: Dominant frequency in Hz (proxy for fundamental pitch or resonance).
    """
    N = len(signal)
    if N == 0:
        return np.array([]), np.array([]), 0.0
    
    # Compute FFT
    fft_vals = scipy.fft.rfft(signal)
    freqs = scipy.fft.rfftfreq(N, d=1.0 / sr)
    
    # Absolute magnitude scaled by length
    magnitudes = np.abs(fft_vals) / N
    
    # Find dominant peak frequency in speech pitch range (60 Hz to 500 Hz)
    pitch_mask = (freqs >= 60) & (freqs <= 500)
    if np.any(pitch_mask):
        peak_idx = np.argmax(magnitudes[pitch_mask])
        peak_freq = float(freqs[pitch_mask][peak_idx])
    else:
        peak_freq = float(freqs[np.argmax(magnitudes)])
        
    return freqs, magnitudes, peak_freq


def compute_stft(
    signal: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if win_length is None or win_length > n_fft:
        win_length = n_fft
    """
    Computes Short-Time Fourier Transform (STFT) linear spectrogram.
    
    Mathematical Concept:
        STFT{x}(m, k) = sum_{n=0}^{N-1} x[n + m * H] * w[n] * exp(-j * 2 * pi * k * n / N)
    
    WHAT it does:
        Applies a Hann window of size win_length to short overlapping chunks of audio,
        hops by hop_length, and computes the FFT for each frame.
    WHY it is needed:
        Speech is non-stationary: phonemes, vowels, and consonants change over time.
        A single global FFT loses time localization. STFT gives a 2D time-frequency map.
        - Large n_fft: high frequency resolution, poor time resolution.
        - Small n_fft: high time resolution, poor frequency resolution.
    WHAT the output means:
        spec_db: 2D array of shape (n_fft // 2 + 1, time_frames) in decibels (dB).
        freq_bins: Frequency values for rows (0 to sr/2 Hz).
        time_bins: Time values in seconds for columns.
    """
    # Use librosa STFT with Hann window
    stft_matrix = librosa.stft(
        y=signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann"
    )
    # Compute power spectrogram: |STFT|^2
    power_spec = np.abs(stft_matrix) ** 2
    # Convert to decibels
    spec_db = librosa.power_to_db(power_spec, ref=np.max)
    
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    time_bins = librosa.frames_to_time(np.arange(spec_db.shape[1]), sr=sr, hop_length=hop_length)
    
    return spec_db, freq_bins, time_bins


def compute_mel_spectrogram(
    signal: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    fmin: float = 50.0,
    fmax: float = 8000.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Log-Mel Spectrogram using psychoacoustic Mel-scale filter banks.
    
    Mathematical Concept:
        Mel formula: m = 2595 * log10(1 + f / 700)
    
    WHAT it does:
        1. Calculates STFT power spectrogram.
        2. Applies n_mels triangular overlapping filters spaced logarithmically on the Mel scale.
        3. Converts linear acoustic energy to decibels (log-amplitude).
    WHY it is needed:
        Human hearing resolves low frequencies with high precision (to distinguish vowel formants
        and speech pitch), but compresses high frequencies. The Mel scale mirrors human auditory
        perception, making it the standard representation for speech AI and deep learning CNNs.
    WHAT the output means:
        mel_db: 2D array of shape (n_mels, time_frames) representing energy in dB.
        mel_freqs: Center frequencies of the n_mels filter banks in Hz.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    
    return mel_db, mel_freqs


def compute_mfcc(
    signal: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    fmin: float = 50.0,
    fmax: float = 8000.0
) -> np.ndarray:
    """
    Computes Mel-Frequency Cepstral Coefficients (MFCCs).
    
    Mathematical Pipeline:
        Audio -> Framing/Windowing -> FFT Magnitude -> Mel Filterbank -> Log -> DCT-II
    
    WHAT it does:
        Applies Discrete Cosine Transform (DCT) to the Log-Mel Spectrogram:
        C[n] = sum_{m=0}^{M-1} log(S_mel[m]) * cos( pi * n * (m + 0.5) / M )
    WHY it is needed:
        - De-correlates filter bank energies into compact, orthogonal coefficients.
        - Separates the vocal tract filter (smooth spectral envelope, captured in low quefrencies)
          from the vocal cord excitation source (pitch harmonics, high quefrency).
    WHAT the output means:
        2D array of shape (n_mfcc, time_frames).
        - MFCC 0: Overall frame energy / loudness.
        - MFCC 1-13: Vocal tract resonances (formants), highly discriminative for speaker gender.
        - MFCC 14-40: Fine spectral ripples / harmonic detail.
    """
    mel_db, _ = compute_mel_spectrogram(
        signal=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    # DCT-II over mel channels
    mfccs = scipy.fft.dct(mel_db, type=2, axis=0, norm="ortho")[:n_mfcc]
    return mfccs


def extract_mfcc_statistics(
    signal: np.ndarray,
    sr: int,
    n_mfcc: int = 20
) -> tuple[np.ndarray, list[str]]:
    """
    Extracts multi-moment temporal statistics and dynamic derivatives from MFCCs.
    
    WHAT it does:
        1. Computes MFCCs across time: shape (n_mfcc, T).
        2. Computes first temporal derivative (Delta, velocity): d/dt.
        3. Computes second temporal derivative (Delta-Delta, acceleration): d^2/dt^2.
        4. Calculates multi-moment statistics across time for each coefficient:
           - Mean (first moment)
           - Standard Deviation (second moment)
           - Skewness (third moment / asymmetry)
           - Kurtosis (fourth moment / tailedness)
           - Minimum & Maximum (dynamic range)
    WHY it is needed:
        Fixes the "temporal collapse" of the old project! Instead of just taking the mean,
        we capture how MFCCs vary, their dynamic range, and how rapidly pitch/formants change.
    WHAT the output means:
        features: 1D numpy array of concatenated statistical descriptors.
        feature_names: Descriptive names for every feature dimension.
    """
    mfcc = compute_mfcc(signal=signal, sr=sr, n_mfcc=n_mfcc)
    
    # Calculate temporal deltas (velocity) and delta-deltas (acceleration)
    # librosa.feature.delta approximates finite-difference derivative
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    matrices = [("mfcc", mfcc), ("delta", delta_mfcc), ("delta2", delta2_mfcc)]
    
    feature_vals = []
    feature_names = []
    
    for name, mat in matrices:
        # mat shape: (n_mfcc, T)
        # Calculate statistics along axis 1 (across time)
        mean_vals = np.mean(mat, axis=1)
        std_vals = np.std(mat, axis=1)
        min_vals = np.min(mat, axis=1)
        max_vals = np.max(mat, axis=1)
        skew_vals = scipy.stats.skew(mat, axis=1)
        kurt_vals = scipy.stats.kurtosis(mat, axis=1)
        
        for i in range(n_mfcc):
            feature_vals.extend([
                mean_vals[i], std_vals[i], min_vals[i],
                max_vals[i], skew_vals[i], kurt_vals[i]
            ])
            feature_names.extend([
                f"{name}_{i}_mean", f"{name}_{i}_std", f"{name}_{i}_min",
                f"{name}_{i}_max", f"{name}_{i}_skew", f"{name}_{i}_kurt"
            ])
            
    return np.array(feature_vals, dtype=np.float32), feature_names
