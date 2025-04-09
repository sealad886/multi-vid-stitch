# Multi-Camera Security Video and Audio Stitching Application

This Python-based project intelligently combines short, event-triggered MP4 video clips from multiple Amazon Blink security cameras into a cohesive, high-quality video with enhanced audio clarity. It addresses common issues like synchronization, overlapping video segments, and background audio interference.

## Features

- **Automated File Parsing:** Extract and sort video clips by camera and timestamp from filenames.
- **Intelligent Clip Synchronization:** Align clips chronologically, handling gaps and overlaps smoothly (relies on video synchronization for audio alignment).
- **Improved Video Synchronization:** Implemented feature detection and matching for more robust video synchronization, handling varying frame rates and camera movements.
- **Quality-based Video Stitching:** Select the best-quality clips for seamless visual continuity.
- **Advanced Audio Enhancement:** Applies normalization and noise reduction (using `noisereduce`) to improve clarity.
- **Volume Control:** Allows overall volume adjustment and track-specific volume adjustments via configuration file.
- **Human Speech Optimization:** Enhance speech clarity and normalize audio levels across the final video.

## Hardware Acceleration Support

The application automatically detects and utilizes available GPU acceleration:

- **CUDA**: Automatically used on NVIDIA GPUs
- **Metal**: Utilized on Apple Silicon and compatible AMD GPUs on macOS
- **OpenCL**: Used as fallback on other supported GPUs
- **CPU**: Automatic fallback when no GPU acceleration is available

The GPU backend is automatically selected in this order of preference:

1. CUDA (highest performance if available)
2. Metal (on macOS systems)
3. OpenCL (cross-platform fallback)
4. CPU (if no GPU acceleration is available)

To check which GPU backend is being used, check the application logs or use the `--log-level DEBUG` option.

## Requirements

- Python 3.8+
- OpenCV
- Librosa
- PyDub
- noisereduce
- FFmpeg
- NVIDIA NeMo (optional for advanced speech enhancement)

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/multicam-stitch.git
cd multicam-stitch
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure FFmpeg is installed on your system:

```bash
# On Ubuntu
sudo apt-get install ffmpeg

# On macOS
brew install ffmpeg
```

## Usage

Place your video clips in the `input_videos` directory, ensuring filenames follow the naming convention:

```text
HH-MM-ss_CameraName_seqID.mp4
```

Run the stitching application using the main module entry point:

```bash
python -m multicam_stitch [OPTIONS]
```

**Command-Line Options:**

- `-i, --input-dir DIR`: Specify the input directory (default: `input_videos`).
- `-o, --output-dir DIR`: Specify the output directory (default: `output`).
- `--log-level LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL; default: INFO).
- `-c, --config FILE`: Path to a JSON configuration file.
- `--no-noise-reduction`: Disable audio noise reduction (it's enabled by default).
- `--overall-volume DB`: Adjust overall audio volume in dB (e.g., -3 for quieter, 6 for louder; default: 0).
- `--feature-detector ALGORITHM`: Specify the feature detection algorithm to use for video synchronization (default: `orb`). Available algorithms: `orb`, `sift`.
- `--stitcher ALGORITHM`: Specify the stitching algorithm to use (default: `feature`). Available algorithms: `feature`, `opencv`.
- `--overlap PERCENTAGE`: Specify the overlap percentage between videos (default: 0.5).
- `--blend MODE`: Specify the blending mode to use for stitching (default: `feather`). Available modes: `feather`, `multi-band`.
- `--perspective-correction`: Enable perspective correction (disabled by default).

**Configuration File (`config.json`):**

You can use a `config.yaml` file to specify settings instead of command-line arguments. CLI arguments override config file settings.

Example `config.yaml`:

```yaml
# Directory paths
input_dir: "input_videos"
output_dir: "output"

# Logging configuration
log_level: "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Audio processing options
noise_reduction: true
overall_volume: -3.0  # Volume adjustment in dB (negative = quieter, positive = louder)
track_volumes:  # Per-track volume adjustments in dB
  "video1.mp4": 0.0
  "video2.mp4": -2.0
  "video3.mp4": 2.0

# Video processing options
feature_detector: "orb"  # Options: orb, sift
stitcher_type: "feature"  # Options: feature, opencv
overlap_percentage: 0.5  # Overlap between videos (0.0 to 1.0)
blending_mode: "feather"  # Options: feather, multi-band
perspective_correction: true

# Output options
output_format: "mp4"
codec: "H264"
encoding_params:
  preset: "medium"
  crf: 23
  max_bitrate: "5000k"
```

## Examples

To demonstrate the usage of the application, let's consider the following example:

1. Place your video clips in the `input_videos` directory. For instance:

   ```text
   input_videos/
       10-30-00_Camera1_01.mp4
       10-30-05_Camera2_01.mp4
       10-30-10_Camera1_02.mp4
   ```

2. Run the application with default settings:

   ```bash
   python -m multicam_stitch
   ```

3. The output will be saved in the `output` directory:

   ```text
   output/
       stitched_video.mp4
   ```

You can also customize the input and output directories, logging level, and other settings using command-line options or a configuration file.

**Stitching Algorithms and Options:**

The application offers two video stitching algorithms:

- `feature`: This algorithm aligns frames using feature detection and matching. It uses the specified feature detector (`--feature-detector`) to find features in the video frames and then aligns the frames based on these features.
- `opencv`: This algorithm uses OpenCV's `Stitcher` class to stitch the videos. It is a more automated approach that may produce better results in some cases.

The following options can be used to adjust the stitching process:

- `--overlap`: This option specifies the overlap percentage between videos. It controls how much the videos should overlap when they are stitched together. A higher overlap percentage may result in a smoother transition between videos, but it may also require more processing power.
- `--blend`: This option specifies the blending mode to use for stitching. The available blending modes are:
- `feather`: This mode uses feathering to blend the videos together. Feathering is a technique that gradually fades the edges of the videos to create a smoother transition.

## Troubleshooting

Common issues and their solutions:

1. **FFmpeg not found**: Ensure FFmpeg is installed and added to your system's PATH.
2. **Video synchronization issues**: Try adjusting the `--overlap` parameter or using a different `--feature-detector` algorithm.
3. **Audio distortion**: Check if the `--overall-volume` adjustment is too high, causing clipping.
4. **Missing dependencies**: Run `pip install -r requirements.txt` to ensure all dependencies are installed.

If you encounter other issues, please check the log files for more detailed error messages.

- `multi-band`: This mode uses multi-band blending to blend the videos together. Multi-band blending is a more advanced technique that may produce better results than feathering in some cases.
- `--perspective-correction`: This option enables perspective correction. Perspective correction is a technique that corrects for perspective distortions in the videos. This can be useful if the videos were taken from different angles.
- `track_volumes`: A dictionary where keys are the relative paths to video files (from the project root) and values are the volume adjustments in dB for that specific track.

Output files will be saved to the specified `output` directory (default: `output`).

## Project Structure

```text
.
├── input_videos
├── output/
├── multicam_stitch/
│   ├── __init__.py
│   ├── __main__.py         # Main executable script
│   ├── audio_processing.py
│   ├── video_processing.py
│   ├── file_parser.py
│   ├── stitch_videos.py    # Core stitching logic
│   └── exceptions.py
│   └── stitch_videos.py
├── requirements.txt
└── README.md
```

## Future Enhancements

- Automated metadata tagging for easier review.
- User interface integration for manual control.

## License

This project is licensed under the GNU General Public License version 3 (GPL-3.0). See below for the full license text.
