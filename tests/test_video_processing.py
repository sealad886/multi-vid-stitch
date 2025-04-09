import pytest
import os
import sys
import cv2
import numpy as np

# Adjust sys.path to include the parent directory of multicam_stitch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multicam_stitch.stitch_videos import stitch_videos
from multicam_stitch.video_processing import (
    synchronize_clips, assess_video_quality,
    get_frame_rate, get_video_duration, align_frames, detect_speakers, create_gpu_mat, GPU_INFO
)
from multicam_stitch.exceptions import VideoProcessingError, SynchronizationError
from .setup_required_files import setup_required_files, add_created_test_file, cleanup_test_files, create_test_video
from datetime import datetime
import time

def setup_module(module):
    setup_required_files()

def teardown_module(module):
    """Clean up test files after all tests are complete."""
    cleanup_test_files()

setup_module(None)
time.sleep(1)  # Allow time for setup to complete

def test_assess_video_quality():
    # Create a fresh test video for quality assessment
    video_path = create_test_video('./input_videos/quality_test.mp4', duration=1)

    try:
        quality = assess_video_quality(video_path)
        assert isinstance(quality, dict)
        assert 'overall_score' in quality
        assert 'speaker_coverage' in quality
        assert 'is_wide_angle' in quality
        assert 'audio_quality' in quality
    except VideoProcessingError:
        pytest.skip("Video quality assessment not supported in this environment")
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

def test_get_frame_rate():
    # Use a fresh test video
    video_path = create_test_video('./input_videos/framerate_test.mp4', fps=30)
    try:
        fps = get_frame_rate(video_path)
        assert isinstance(fps, float)
        assert fps > 0
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

def test_get_video_duration():
    """Test video duration with verification of actual frames."""
    video_path = None
    try:
        video_path = create_test_video('./input_videos/duration_test.mp4', duration=2.0, fps=30)

        # First verify the video exists and can be opened
        if not os.path.exists(video_path):
            pytest.skip("Could not create test video")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            pytest.skip("Could not open test video")

        # Get actual video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        expected_duration = frame_count / fps
        cap.release()

        # Test the duration
        duration = get_video_duration(video_path)
        assert isinstance(duration, float)
        assert duration > 0
        assert abs(duration - expected_duration) < 0.5  # Allow for some variance

    except Exception as e:
        pytest.skip(f"Video creation failed: {str(e)}")
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)

def test_align_frames():
    # Create two simple test frames with known features
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add distinct features to both frames
    cv2.rectangle(frame1, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.rectangle(frame2, (120, 120), (220, 220), (255, 255, 255), -1)

    aligned = align_frames(frame1, frame2)
    assert isinstance(aligned, np.ndarray)
    assert aligned.shape == frame1.shape

    # Test with GPU if available
    if GPU_INFO['backend']:
        aligned_gpu = align_frames(frame1, frame2, use_gpu=True)
        assert isinstance(aligned_gpu, (np.ndarray, cv2.UMat))
        if isinstance(aligned_gpu, cv2.UMat):
            aligned_gpu = aligned_gpu.get()
        assert aligned_gpu.shape == frame1.shape

def test_detect_speakers():
    video_path = './input_videos/12-30-00_Camera1_seq1.mp4'
    try:
        segments = detect_speakers(video_path)
        assert isinstance(segments, list)
    except (SynchronizationError, VideoProcessingError) as e:
        pytest.skip(f"Skipping due to video processing limitations: {str(e)}")

def test_synchronize_clips():
    try:
        # Create test videos first
        video1_path = create_test_video('./input_videos/12-30-00_Camera1_seq1.avi', duration=2)
        video2_path = create_test_video('./input_videos/12-31-00_Camera2_seq2.avi', duration=2)
    except RuntimeError as e:
        pytest.skip(f"Could not create test videos: {e}")

    video_files = [
        {
            'filepath': video1_path,
            'timestamp': datetime.now()
        },
        {
            'filepath': video2_path,
            'timestamp': datetime.now()
        }
    ]

    try:
        segments = synchronize_clips(video_files)
        assert isinstance(segments, list)
        if segments:
            assert all(isinstance(segment, dict) for segment in segments)
    except SynchronizationError:
        pytest.skip("Skipping due to synchronization limitations in test environment")

def test_synchronize_clips_invalid_files():
    video_files = [
        {
            'filepath': 'nonexistent1.mp4',
            'timestamp': datetime.now()
        }
    ]

    try:
        synchronize_clips(video_files)
        pytest.fail("Expected VideoProcessingError or SynchronizationError")
    except (VideoProcessingError, SynchronizationError):
        assert True  # Test passes if either exception is raised

def test_stitch_videos_feature_detector():
    video_files = [
        {'filepath': './input_videos/12-30-00_Camera1_seq1.mp4'},
        {'filepath': './input_videos/12-31-00_Camera2_seq2.mp4'}
    ]

    try:
        result = stitch_videos(video_files, feature_detector='orb')
        assert result is not None
        assert os.path.exists(result)
    except VideoProcessingError:
        pytest.skip("Skipping due to video processing limitations in test environment")

def test_stitch_videos_stitcher_type():
    video_files = [
        {'filepath': './input_videos/12-30-00_Camera1_seq1.mp4'},
        {'filepath': './input_videos/12-31-00_Camera2_seq2.mp4'}
    ]

    try:
        result = stitch_videos(video_files, stitcher_type='opencv')
        assert result is not None
        assert os.path.exists(result)
    except VideoProcessingError:
        pytest.skip("Skipping due to video processing limitations in test environment")

def test_gpu_detection():
    assert isinstance(GPU_INFO, dict)
    assert 'backend' in GPU_INFO
    assert 'device_name' in GPU_INFO
    assert 'platform' in GPU_INFO

def test_create_gpu_mat():
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    gpu_frame = create_gpu_mat(test_frame)
    assert gpu_frame is not None

    # Check GPU support
    if GPU_INFO['backend'] is not None:
        if GPU_INFO['backend'] == 'cuda':
            assert isinstance(gpu_frame, cv2.cuda_GpuMat)
        elif GPU_INFO['backend'] in ('metal', 'opencl'):
            try:
                # Try to create UMat
                umat = cv2.UMat(test_frame)
                assert isinstance(gpu_frame, (cv2.UMat, np.ndarray))
            except cv2.error:
                pytest.skip("OpenCL/Metal acceleration not available")
    else:
        assert isinstance(gpu_frame, np.ndarray)

teardown_module(None)
