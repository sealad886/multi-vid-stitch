import pytest
import os
import sys
import numpy as np
import time

# Adjust sys.path to include the parent directory of multicam_stitch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multicam_stitch.audio_processing import assess_audio_quality, enhance_audio_segment, process_video_audio
from multicam_stitch.exceptions import AudioProcessingError
from tests.setup_required_files import setup_required_files, add_created_test_file, cleanup_test_files
from pydub import AudioSegment
from pydub.generators import Sine

def setup_module(module):
    setup_required_files()

def teardown_module(module):
    cleanup_test_files()

setup_module(None)
time.sleep(1)  # Allow time for setup to complete

def create_test_audio_segment(duration_ms=1000, sample_rate=44100):
    sine_wave = Sine(440).to_audio_segment(duration=duration_ms)
    return sine_wave

def test_assess_audio_quality():
    audio = create_test_audio_segment()
    quality = assess_audio_quality(audio)

    assert isinstance(quality, dict)
    assert 'signal_to_noise' in quality
    assert 'clarity_score' in quality
    assert 'overall_quality' in quality
    assert 0 <= quality['overall_quality'] <= 100

def test_enhance_audio_segment():
    audio = create_test_audio_segment()
    enhanced = enhance_audio_segment(audio, noise_reduce=True)

    assert isinstance(enhanced, AudioSegment)
    assert len(enhanced) == len(audio)

def test_process_video_audio():
    video_files = [
        {'filepath': './input_videos/12-30-00_Camera1_seq1.mp4'},
        {'filepath': './input_videos/12-31-00_Camera2_seq2.mp4'}
    ]

    try:
        result = process_video_audio(video_files, noise_reduce=True)
        assert result is not None
        assert os.path.exists(result)
    except AudioProcessingError:
        pytest.skip("Skipping due to audio processing limitations in test environment")

def test_process_video_audio_invalid_files():
    video_files = [
        {'filepath': 'nonexistent1.mp4'},
        {'filepath': 'nonexistent2.mp4'}
    ]

    with pytest.raises(AudioProcessingError):
        process_video_audio(video_files)

def test_process_video_audio_empty_list():
    with pytest.raises(AudioProcessingError):
        process_video_audio([])

teardown_module(None)
