import numpy as np
import librosa
from pydub import AudioSegment
from typing import List, Dict
import noisereduce as nr
from pydub.effects import normalize

def apply_equalizer(audio_segment: AudioSegment, bands: List[float], gains: List[float]) -> AudioSegment:
    """
    Apply multi-band equalizer to audio using librosa.

    Args:
        audio_segment: Input audio segment
        bands: List of frequency bands in Hz
        gains: List of gain values in dB for each band

    Returns:
        Equalized audio segment
    """
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    sample_rate = audio_segment.frame_rate

    channels = []
    if audio_segment.channels == 2:
        samples = samples.reshape(-1, 2)
        for ch in range(2):
            channel = samples[:, ch]
            eq_channel = _apply_eq_to_channel(channel, sample_rate, bands, gains)
            channels.append(eq_channel)
        eq_samples = np.column_stack(channels).ravel()
    else:
        eq_samples = _apply_eq_to_channel(samples, sample_rate, bands, gains)

    return AudioSegment(
        eq_samples.astype(np.int16).tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=audio_segment.channels
    )

def _apply_eq_to_channel(channel: np.ndarray, sample_rate: int, bands: List[float], gains: List[float]) -> np.ndarray:
    """Apply EQ to a single channel"""
    eq_channel = np.zeros_like(channel)
    for band, gain in zip(bands, gains):
        linear_gain = 10 ** (gain / 20.0)
        filtered = librosa.effects.preemphasis(channel, coef=band/sample_rate)
        eq_channel += filtered * linear_gain
    return eq_channel

def process_audio_track(track: AudioSegment, noise_reduction_threshold: float = 0.1,
                       normalization_level: float = -3.0, eq_settings: Dict = None) -> AudioSegment:
    """Process a single audio track with noise reduction, normalization, and EQ"""
    # Noise reduction
    samples = np.array(track.get_array_of_samples()).astype(np.float32)
    reduced_noise = nr.reduce_noise(
        y=samples,
        sr=track.frame_rate,
        stationary=True,
        prop_decrease=noise_reduction_threshold
    )
    track = AudioSegment(
        reduced_noise.astype(np.int16).tobytes(),
        frame_rate=track.frame_rate,
        sample_width=track.sample_width,
        channels=track.channels
    )

    # Normalization
    track = normalize(track, headroom=normalization_level)

    # EQ if settings provided
    if eq_settings and 'bands' in eq_settings and 'gains' in eq_settings:
        track = apply_equalizer(
            track,
            bands=eq_settings['bands'],
            gains=eq_settings['gains']
        )

    return track
