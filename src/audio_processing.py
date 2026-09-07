"""
Audio Processing Module: Voice Gender Classification v2
Teaches and implements core digital signal processing (DSP) fundamentals:
- Audio loading, sampling rate, bit depth, channels, duration
- Peak amplitude, RMS energy, and peak normalization
- Mono conversion & Resampling
- Energy-based Voice Activity Detection (VAD) & Silence Trimming
"""

import numpy as np
import soundfile as sf
import scipy.signal


def get_audio_metadata(file_path: str) -> dict:
    """
    Inspects physical properties of an audio file without loading entire array into memory.
    
    WHAT it does:
        Reads the audio file header to extract sampling rate, channels, bit format, and duration.
    WHY it is needed:
        Avoids loading huge uncompressed audio files into RAM when only inspecting properties.
    WHAT the output means:
        Returns basic digital audio descriptors.
    """
    info = sf.info(file_path)
    return {
        "file_path": file_path,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "duration_sec": info.duration,
        "total_samples": info.frames,
        "format": info.format,
        "subtype": info.subtype  # e.g., PCM_16 (16-bit)
    }


def load_audio(file_path: str, target_sr: int = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """
    Loads an audio file into a 32-bit floating point numpy array.
    
    WHAT it does:
        Reads PCM audio, converts multi-channel to mono (if requested),
        and resamples to target_sr (if requested).
    WHY it is needed:
        Neural networks require consistent sample rates, channel counts, and normalized floats.
    WHAT the output means:
        signal: 1D numpy array of shape (samples,) with values in [-1.0, 1.0].
        sr: The effective sampling rate in Hz.
    """
    # sf.read returns float32 in range [-1.0, 1.0] by default
    data, sr = sf.read(file_path, dtype="float32")
    
    # 1. Handle Channels: Convert multi-channel to mono if requested
    if mono and data.ndim > 1:
        # Channel averaging: (left + right) / 2
        data = np.mean(data, axis=1)
    
    # 2. Handle Resampling if target_sr is specified and differs
    if target_sr is not None and target_sr != sr:
        data = resample_audio(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        
    return data, sr


def compute_rms(signal: np.ndarray) -> float:
    """
    Computes Root Mean Square (RMS) amplitude of a time-domain signal.
    
    Formula:
        RMS = sqrt( (1 / N) * sum( x[n]^2 ) )
    
    WHAT it does:
        Calculates the quadratic mean (effective energy) of the waveform.
    WHY it is needed:
        Peak amplitude tells you only the single loudest sample, but RMS measures
        perceived loudness / continuous acoustic energy over the duration.
    WHAT the output means:
        A positive float between 0.0 (total silence) and 1.0 (full-scale square wave).
    """
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2)))


def rms_to_dbfs(rms_value: float, eps: float = 1e-9) -> float:
    """
    Converts linear RMS amplitude to Decibels relative to Full Scale (dBFS).
    
    Formula:
        dBFS = 20 * log10( RMS / 1.0 )
    
    WHAT the output means:
        0 dBFS is the maximum possible digital volume without clipping.
        Values are negative (e.g., -20 dBFS is moderate speech; -60 dBFS is near silence).
    """
    return float(20.0 * np.log10(max(rms_value, eps)))


def peak_normalize(signal: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Applies Peak Normalization to scale signal to target_peak.
    
    Formula:
        x_norm[n] = x[n] * (target_peak / max(|x[n]|))
    
    WHAT it does:
        Linearly scales audio so its maximum absolute value equals target_peak.
    WHY it is needed:
        Ensures consistent loudness scaling without introducing non-linear clipping distortion.
    WHAT the output means:
        Normalized array where max(abs(x_norm)) == target_peak.
    """
    peak = np.max(np.abs(signal))
    if peak <= 1e-8:
        return signal.copy()
    scaling_factor = target_peak / peak
    return signal * scaling_factor


def resample_audio(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    High-quality polyphase audio resampling using scipy.signal.resample_poly.
    
    WHAT it does:
        Converts the discrete time signal from orig_sr to target_sr using rational fraction L/M.
    WHY it is needed:
        Speech models typically operate at a standard sampling rate (e.g. 16,000 Hz).
        Polyphase filtering avoids aliasing artifacts by applying a low-pass anti-aliasing filter.
    WHAT the output means:
        New audio array with length = round(len(signal) * target_sr / orig_sr).
    """
    if orig_sr == target_sr:
        return signal
    
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    
    resampled = scipy.signal.resample_poly(signal, up=up, down=down)
    return resampled.astype(np.float32)


def energy_vad(
    signal: np.ndarray,
    sr: int,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    energy_threshold_db: float = -35.0,
    min_speech_duration_ms: float = 100.0
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Energy-Based Voice Activity Detection (VAD) and Silence Trimming.
    
    WHAT it does:
        Divides signal into short overlapping frames (e.g. 25ms window, 10ms step),
        calculates short-term RMS energy (in dBFS) for each frame,
        and identifies active speech frames vs background silence.
    WHY it is needed:
        In the old project, leading/trailing silence diluted the MFCC average.
        VAD removes inactive regions so models focus exclusively on active speech.
    WHAT the output means:
        trimmed_signal: Audio containing only active speech segments.
        is_speech_mask: Boolean array over frames indicating speech presence.
        vad_info: Metadata including silence percentage and speech duration.
    """
    frame_length = int(round(frame_length_ms * sr / 1000.0))
    hop_length = int(round(hop_length_ms * sr / 1000.0))
    
    if len(signal) < frame_length:
        return signal, np.array([True]), {"silence_ratio": 0.0}
    
    # Extract frames using sliding window
    num_frames = 1 + (len(signal) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        signal,
        shape=(num_frames, frame_length),
        strides=(signal.strides[0] * hop_length, signal.strides[0])
    )
    
    # Calculate RMS energy for each frame in dBFS
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    frame_db = 20.0 * np.log10(frame_rms)
    
    # Compare with adaptive threshold: relative to peak frame energy or absolute
    peak_frame_db = np.max(frame_db)
    # Active if above absolute threshold AND within 30dB of peak voice frame
    is_speech = (frame_db >= energy_threshold_db) & (frame_db >= peak_frame_db - 35.0)
    
    # Smooth VAD decisions using a morphological closing/opening filter (dilation/erosion)
    kernel_size = max(3, int(round(min_speech_duration_ms / hop_length_ms)))
    kernel = np.ones(kernel_size, dtype=bool)
    is_speech_smoothed = scipy.ndimage.binary_closing(is_speech, structure=kernel)
    is_speech_smoothed = scipy.ndimage.binary_opening(is_speech_smoothed, structure=kernel)
    
    # Find speech boundaries (leading and trailing trimming)
    speech_indices = np.where(is_speech_smoothed)[0]
    if len(speech_indices) == 0:
        # Fallback if entirely silent: return original signal
        return signal, is_speech_smoothed, {"silence_ratio": 1.0, "trimmed_samples": len(signal)}
    
    start_frame = speech_indices[0]
    end_frame = speech_indices[-1]
    
    start_sample = max(0, start_frame * hop_length)
    end_sample = min(len(signal), end_frame * hop_length + frame_length)
    
    trimmed_signal = signal[start_sample:end_sample]
    silence_ratio = 1.0 - (len(trimmed_signal) / len(signal))
    
    vad_info = {
        "orig_samples": len(signal),
        "trimmed_samples": len(trimmed_signal),
        "orig_duration_sec": len(signal) / sr,
        "trimmed_duration_sec": len(trimmed_signal) / sr,
        "silence_ratio": silence_ratio,
        "start_time_sec": start_sample / sr,
        "end_time_sec": end_sample / sr
    }
    
    return trimmed_signal, is_speech_smoothed, vad_info
