import os
import cv2
import numpy as np
import platform
import time
import subprocess

_created_test_files = set()  # Track created files globally

def create_test_video(filename, duration=2, fps=30, size=(640, 480)):
    """Creates a test video file using FFmpeg."""
    output_file = filename
    try:
        command = [
            'ffmpeg',
            '-f', 'lavfi',
            '-i', f'testsrc=duration={duration}:size={size[0]}x{size[1]}:rate={fps}',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_file
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully created video: {output_file}")
        add_created_test_file(output_file)
        return output_file
    except Exception as e:
        print(f"Failed to create video: {e}")
        if os.path.exists(output_file):
            os.remove(output_file)
        raise RuntimeError(f"Failed to create video: {str(e)}")

def add_created_test_file(filepath):
    _created_test_files.add(filepath)
    return

def get_created_test_files():
    """Returns set of all test files created during testing."""
    return _created_test_files

def cleanup_test_files():
    """Removes all test files created during testing."""
    for file in get_created_test_files():
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed test file: {file}")
        except OSError as e:
            print(f"Error removing test file {file}: {e}")
    _created_test_files.clear()

def setup_required_files():
    """Sets up all required test files and directories."""
    # Create necessary directories
    dirs = ['input_videos', 'output']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    # Create test videos with proper timestamps for testing
    test_videos = [
        ('12-30-00_Camera1_seq1', 2),  # 2 seconds duration
        ('12-31-00_Camera2_seq2', 2),   # 2 seconds duration
    ]

    for base_name, duration in test_videos:
        input_path = os.path.join('input_videos', f"{base_name}.mp4")
        try:
            output_file = create_test_video(input_path, duration=duration)
        except Exception as e:
            print(f"Failed to create {input_path}: {e}")

    return

if __name__ == "__main__":
    setup_required_files()
