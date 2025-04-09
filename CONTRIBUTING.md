# Contributing Developer Documentation

## Overview

This document provides detailed information for developers contributing to the project.

## Code Structure

The project is organized into several modules within the `multicam_stitch` package:

- `__main__.py`: Main entry point for the application
- `audio_processing.py`: Handles audio-related operations
- `video_processing.py`: Contains video processing functionality
- `file_parser.py`: Responsible for parsing input video filenames
- `stitch_videos.py`: Core logic for stitching videos together
- `exceptions.py`: Custom exception classes used throughout the project

## Contribution Guidelines

1. Fork the repository and create a new branch for your feature or bug fix.
2. Ensure all tests pass before submitting a pull request.
3. Follow PEP 8 guidelines for code style.
4. Document your changes in this file and update inline docstrings as necessary.

## Testing

The project uses pytest for testing. Tests are located in the `tests/` directory.

### Running Tests

To run tests:

```bash
python -m pytest tests
```

### Test Files

Tests create temporary video and audio files that are automatically cleaned up after testing.
The cleanup process:

1. Creates test files in `input_videos/` and `output/` directories
2. Removes all generated test files after tests complete
3. Removes empty test directories if no other files exist

### Test Environment Requirements

- OpenCV (cv2) with video codecs support
- FFmpeg for video encoding/decoding
- Sufficient disk space for temporary test files (~100MB recommended)
- Write permissions in the test directories

### Test File Naming Convention

Test files follow these patterns:

- Video tests: `HH-MM-SS_CameraX_seqY.mp4`
- Quality tests: `quality_test.mp4`
- Frame rate tests: `framerate_test.mp4`
- Duration tests: `duration_test.mp4`

### Adding New Tests

When adding new tests:

1. Use the `create_test_video()` helper function for video files
2. Add cleanup steps in `teardown_module()`
3. Handle codec compatibility issues
4. Use try/except blocks with appropriate skip conditions

## GPU Acceleration

### GPU Backend Detection

The application uses a priority-based system for GPU backend selection:

1. CUDA (NVIDIA GPUs)
2. Metal (Apple Silicon/AMD on macOS)
3. OpenCL (Cross-platform)
4. CPU fallback

### Testing GPU Features

When writing tests involving GPU operations:

1. Always handle both GPU and CPU cases
2. Use `GPU_INFO` to check available backends
3. Test with `cv2.UMat` and `np.ndarray` types
4. Include proper cleanup of GPU resources
5. Use `pytest.skip()` for unavailable GPU features

Example:

```python
def test_gpu_feature():
    if not GPU_INFO['backend']:
        pytest.skip("No GPU acceleration available")
    # Test GPU-specific functionality
```

### GPU Memory Management

- Use `create_gpu_mat()` for GPU memory allocation
- Always release GPU resources in `finally` blocks
- Handle both `cv2.UMat` and `cv2.cuda_GpuMat` types
- Check `isinstance()` before GPU operations

## Logging

The application uses Python's logging module. Log levels can be adjusted using the `--log-level` command-line argument.

## Future Enhancements

- Consider adding type hints for function parameters and return types.
- Explore using a more advanced video stitching library for improved performance.
