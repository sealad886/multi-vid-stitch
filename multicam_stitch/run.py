#!/usr/bin/env python3

# Standard library imports
import os
import sys
import json
import signal
from threading import Lock

# GUI and interface imports
import tkinter as tk
from tkinter import ttk
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn
)

# Audio processing imports
import pyaudio
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import normalize

# Local module imports
from .exceptions import (
    InvalidInputError, AudioProcessingError, AudioVideoSyncError,
    VideoProcessingError, FrameRateMismatchError,
    TimestampMismatchError, FrameSizeMismatchError,
    CodecUnsupportedError, GPUMemoryError,
    SynchronizationError, StitchingError,
    InvalidColorSpaceError, StitchingBoundaryError,
    OutputWriteError, ProcessingError,
    FileParsingError
)
from . import file_parser
from . import audio_utils
from . import audio_processing
from . import video_processing
from .stitch_videos import stitch_videos
from .logger_manager import get_logger

logger = get_logger()
progress = None  # Initialize global progress variable
app = None  # Add global app variable

# Create a single Console instance for reuse
console = Console()

class InteractiveQualityAssessment:
    """Interactive GUI for assessing audio quality and managing tracks."""

    def __init__(self, root):
        import contextlib
        self.root = root
        self.root.title("Multi-Cam Audio Quality Assessment")
        self.pyaudio_instance = pyaudio.PyAudio()  # Add this line
        self.setup_gui()

    def setup_gui(self):
        # Create main frames
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(pady=5)

        # Track selection
        self.track_var = tk.StringVar()
        ttk.Label(self.control_frame, text="Track:").pack(side=tk.LEFT)
        self.track_dropdown = ttk.Combobox(self.control_frame, textvariable=self.track_var)
        self.track_dropdown.pack(side=tk.LEFT, padx=5)

        # Playback controls
        self.play_btn = ttk.Button(self.control_frame, text="Play", command=self.start_playback)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(self.control_frame, text="Stop", command=self.stop_playback)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Segment selection
        self.start_time = tk.StringVar()
        self.end_time = tk.StringVar()
        ttk.Label(self.control_frame, text="Start Time:").pack(side=tk.LEFT)
        ttk.Entry(self.control_frame, textvariable=self.start_time).pack(side=tk.LEFT)
        ttk.Label(self.control_frame, text="End Time:").pack(side=tk.LEFT)
        ttk.Entry(self.control_frame, textvariable=self.end_time).pack(side=tk.LEFT)
        self.select_btn = ttk.Button(self.control_frame, text="Select Segment", command=self.mark_segment)
        self.select_btn.pack(side=tk.LEFT, padx=5)

        # Metadata entry
        self.metadata_var = tk.StringVar()
        ttk.Label(self.control_frame, text="Metadata:").pack(side=tk.LEFT)
        ttk.Entry(self.control_frame, textvariable=self.metadata_var).pack(side=tk.LEFT)

        # Audio visualization
        self.canvas = tk.Canvas(self.root, width=800, height=200)
        self.canvas.pack(pady=10)

        # Initialize audio playback
        self.audio = None
        self.stream = None

        # Load available tracks
        self.load_available_tracks()

    def load_available_tracks(self, tracks=None):
        """Load audio tracks into the interface.

        Args:
            tracks: List of (name, audio_data) tuples or None to scan input dir
        """
        if tracks is None:
            # Scan input directory for audio files
            input_dir = os.path.join(os.path.dirname(__file__), '../input_videos')
            tracks = []
            for f in os.listdir(input_dir):
                if f.endswith(('.wav', '.mp3', '.aac')):
                    tracks.append((f, None))  # Will load on demand

        self.tracks = tracks
        self.track_dropdown['values'] = [name for name, _ in tracks]
        if tracks:
            self.track_var.set(tracks[0][0])

    def start_playback(self):
        """Start playing the selected audio track."""
        if hasattr(self, 'stream') and self.stream:
            self.stop_playback()

        track_name = self.track_var.get()
        track_data = next((data for name, data in self.tracks if name == track_name), None)

        if not track_data:
            logger.error(f"Track {track_name} not found")
            return

        self.stream = self.pyaudio_instance.open(  # Use class instance
            format=pyaudio.paFloat32,
            channels=1,
            rate=44100,
            output=True)
        self.stream.write(track_data)

    def stop_playback(self):
        """Stop audio playback."""
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def mark_segment(self):
        """Mark the current time segment for inclusion in final output."""
        start = float(self.start_time.get())
        end = float(self.end_time.get())
        metadata = self.metadata_var.get()

        if not hasattr(self, 'selected_segments'):
            self.selected_segments = []

        self.selected_segments.append({
            'start': start,
            'end': end,
            'track': self.track_var.get(),
            'metadata': metadata
        })

    def get_selected_segments(self):
        """Return all marked segments with their metadata."""
        return getattr(self, 'selected_segments', [])

    def __del__(self):
        """Cleanup when instance is destroyed"""
        if hasattr(self, 'pyaudio_instance'):
            self.pyaudio_instance.terminate()

def create_progress_display() -> Progress:
    """Creates a rich progress display with multiple bars.

    Returns:
        Progress: A Progress object configured for displaying progress bars.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("'>"),
        TimeRemainingColumn(compact=True,),
        expand=True,
        redirect_stderr=True,
        redirect_stdout=True,
        refresh_per_second=2,
    )

def cleanup():
    """Perform cleanup of resources before exiting.

    Releases all system resources including:
    - Audio streams and PyAudio instances
    - GUI windows
    - GPU resources
    - Running threads
    - File handles
    """
    global console, progress, app  # Add app to globals
    console.print("[yellow]Cleaning up resources...[/yellow]")

    # Clean up any running GUI instances
    try:
        if app is not None and hasattr(app, 'root'):
            app.root.destroy()
            if hasattr(app, 'pyaudio_instance'):
                app.pyaudio_instance.terminate()
    except Exception as e:
        logger.error(f"Error closing GUI: {e}")

    # Clean up audio resources
    try:
        if app is not None and hasattr(app, 'stream'):
            if app.stream.is_active():
                app.stream.stop_stream()
            app.stream.close()
            app.stream = None
    except Exception as e:
        logger.error(f"Error closing audio stream: {e}")

    # Clean up GPU resources (if used)
    try:
        from .gpu_setup import release_gpu_resources
        release_gpu_resources()
    except ImportError:
        pass  # GPU not available
    except Exception as e:
        logger.error(f"Error releasing GPU resources: {e}")

    # Clean up progress display
    try:
        if progress is not None:
            progress.stop()
    except Exception as e:
        logger.error(f"Error stopping progress display: {e}")

    logger.info("Cleanup completed")

def signal_handler(signum: int, frame) -> None:
    """Handle interrupt signals gracefully.

    Args:
        signum (int): Signal number.
        frame: Current stack frame.
    """
    global console
    console.print("\n[yellow]Received interrupt signal. Cleaning up...[/yellow]")
    cleanup()
    sys.exit(0)

def main(input_dir: str, output_dir: str, noise_reduction: bool = True, overall_volume: float = 0.0,
         track_volumes: dict = None, feature_detector: str = 'orb', stitcher_type: str = 'feature',
         overlap_percentage: float = 0.5, blending_mode: str = 'feather', perspective_correction: bool = True,
         output_format: str = 'mp4', codec: str = 'mp4v', encoding_params: str = None,
         manual_stitching_order: list = None) -> str:
    """Main function to stitch multiple video files into a single output video with enhanced audio.

    Args:
        input_dir (str): Directory containing input video files.
        output_dir (str): Directory where the output video will be saved.
        noise_reduction (bool, optional): Whether to apply noise reduction to audio. Defaults to True.
        overall_volume (float, optional): Overall volume adjustment in dB. Defaults to 0.0.
        track_volumes (dict, optional): Dictionary of volume adjustments for specific tracks. Defaults to None.
        feature_detector (str, optional): Feature detection algorithm to use. Defaults to 'orb'.
        stitcher_type (str, optional): Type of stitcher to use. Defaults to 'feature'.
        overlap_percentage (float, optional): Percentage of overlap between videos. Defaults to 0.5.
        blending_mode (str, optional): Blending mode for stitching. Defaults to 'feather'.
        perspective_correction (bool, optional): Whether to apply perspective correction. Defaults to True.
        output_format (str, optional): Output video format. Defaults to 'mp4'.
        codec (str, optional): Codec to use for output video. Defaults to 'mp4v'.
        encoding_params (str, optional): Additional encoding parameters. Defaults to None.
        manual_stitching_order (list, optional): Manual order for stitching videos. Defaults to None.

    Returns:
        str: Path to the output stitched video file.
    """
    global progress  # Add global declaration

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    console.print(Panel("Multi-Camera Video Stitching", style="bold blue"))

    # Create progress display
    progress = create_progress_display()
    progress_lock = Lock()

    def update_progress(task_id, completed):
        with progress_lock:
            progress.update(task_id, completed=min(completed, progress.tasks[task_id].total))

    with Live(progress, refresh_per_second=10, transient=True) as live:
        try:
            # 1. Parameter validation and initialization
            try:
                if not os.path.exists(input_dir):
                    raise InvalidInputError(f"Input directory does not exist: {input_dir}")
                if not os.path.exists(output_dir):
                    try:
                        os.makedirs(output_dir)
                    except OSError as e:
                        raise OutputWriteError(f"Failed to create output directory: {e}")

                # 2. Parse input files
                try:
                    video_files = file_parser.get_video_files(input_dir)
                    if not video_files:
                        raise InvalidInputError("No valid video files found in input directory")
                except FileParsingError as e:
                    logger.error(f"File parsing failed: {e}")
                    raise

                # 3. Audio processing pipeline
                try:
                    audio_tracks = audio_processing.process_video_audio(
                        video_files,
                        track_volumes=track_volumes,
                        overall_volume=overall_volume,
                        noise_reduce=noise_reduction,
                        progress_callback=lambda p: update_progress('audio', p)
                    )

                    processed_tracks, sync_markers = audio_processing.parallel_audio_processing(
                        video_files,
                        noise_reduction_threshold=0.1 if noise_reduction else 0,
                        normalization_level=-3.0,
                        progress_callback=lambda p: update_progress('audio', p)
                    )
                except AudioVideoSyncError as e:
                    logger.error(f"Audio-video synchronization failed: {e}")
                    raise AudioProcessingError(f"Audio processing failed: {str(e)}")
                except Exception as e:
                    logger.error(f"Audio processing error: {e}")
                    raise AudioProcessingError(f"Audio processing failed: {str(e)}")

                # 4. Interactive quality assessment
                selected_segments = interactive_quality_assessment()

                # 5. Video processing and stitching
                try:
                    synchronized_segments = video_processing.synchronize_clips(
                        video_files,
                        progress_callback=lambda p: update_progress('video', p),
                        manual_offsets=None
                    )
                except FrameRateMismatchError as e:
                    logger.error(f"Frame rate mismatch: {e}")
                    raise VideoProcessingError(f"Video synchronization failed: {str(e)}")
                except TimestampMismatchError as e:
                    logger.error(f"Timestamp mismatch: {e}")
                    raise VideoProcessingError(f"Video synchronization failed: {str(e)}")
                except Exception as e:
                    logger.error(f"Video synchronization error: {e}")
                    raise VideoProcessingError(f"Video synchronization failed: {str(e)}")

                # 6. Stitch videos with enhanced error handling
                try:
                    output_path = stitch_videos(
                        video_files,
                        stitcher_type=stitcher_type,
                        feature_detector=feature_detector,
                        progress_callback=lambda p: update_progress('stitch', p),
                        overlap_percentage=overlap_percentage,
                        blending_mode=blending_mode,
                        perspective_correction=perspective_correction,
                        output_format=output_format,
                        codec=codec,
                        encoding_params=encoding_params,
                        manual_stitching_order=manual_stitching_order
                    )
                except CodecUnsupportedError as e:
                    logger.error(f"Unsupported codec: {e}")
                    raise VideoProcessingError(f"Video stitching failed: {str(e)}")
                except FrameSizeMismatchError as e:
                    logger.error(f"Frame size mismatch: {e}")
                    raise VideoProcessingError(f"Video stitching failed: {str(e)}")
                except GPUMemoryError as e:
                    logger.error(f"GPU memory error: {e}")
                    raise VideoProcessingError(f"Video stitching failed: {str(e)}")
                except StitchingBoundaryError as e:
                    logger.error(f"Stitching boundary exceeded: {e}")
                    raise VideoProcessingError(f"Video stitching failed: {str(e)}")
                except InvalidColorSpaceError as e:
                    logger.error(f"Invalid color space: {e}")
                    raise VideoProcessingError(f"Video stitching failed: {str(e)}")
                except VideoProcessingError as e:
                    logger.error(f"Video stitching failed: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error during stitching: {e}")
                    raise StitchingError(f"Stitching failed: {str(e)}")

                # 7. Generate final output
                try:
                    final_output = os.path.join(output_dir, f"stitched_output.{output_format}")
                    os.rename(output_path, final_output)
                    logger.info(f"Final output generated at: {final_output}")
                    return final_output
                except OSError as e:
                    raise OutputWriteError(f"Failed to write output file: {e}")

            except InvalidInputError as e:
                logger.error(f"Input validation failed: {e}")
                raise
            except AudioProcessingError as e:
                logger.error(f"Audio processing failed: {e}")
                raise
            except VideoProcessingError as e:
                logger.error(f"Video processing failed: {e}")
                raise
            except KeyboardInterrupt:
                logger.info("Processing interrupted by user")
                raise
            except OutputWriteError as e:
                logger.error(f"Output write error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise ProcessingError(f"Processing failed: {str(e)}")
        finally:
            # Ensure cleanup runs on all exit paths
            cleanup()

def interactive_quality_assessment():
    global app  # Add global declaration
    root = tk.Tk()
    app = InteractiveQualityAssessment(root)
    root.mainloop()
    return app.get_selected_segments()  # Returns metadata about selected segments
