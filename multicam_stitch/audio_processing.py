# multicam_stitch/audio_processing.py
from pydub.exceptions import CouldntDecodeError
from pydub import AudioSegment
import numpy as np
import os
import multiprocessing

import pydub.exceptions
from . import audio_utils
from .exceptions import AudioProcessingError
from .logger_manager import get_logger
logger = get_logger()

def load_audio_track(filepath: str, start_time: float = None, end_time: float = None) -> AudioSegment:
    """
    Load an audio track from a file with optional time slicing.

    Args:
        filepath (str): Path to the audio file
        start_time (float, optional): Start time in seconds. Defaults to None.
        end_time (float, optional): End time in seconds. Defaults to None.

    Returns:
        AudioSegment: Loaded audio segment

    Raises:
        AudioProcessingError: If file cannot be loaded or is invalid
    """
    try:
        audio = AudioSegment.from_file(filepath)

        if start_time is not None and end_time is not None:
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            audio = audio[start_ms:end_ms]

        return audio
    except Exception as e:
        logger.error(f"Failed to load audio file {filepath}: {str(e)}")
        raise AudioProcessingError(f"Failed to load audio file: {str(e)}")

def assess_audio_quality(audio_segment):
    """
    Analyzes audio quality using various metrics.

    Args:
        audio_segment (AudioSegment): PyDub audio segment to analyze

    Returns:
        dict: Quality metrics including:
            - signal_to_noise: float
            - clarity_score: float
            - overall_quality: float
    """
    # Convert PyDub audio segment to numpy array for analysis
    samples = np.array(audio_segment.get_array_of_samples())

    # Calculate signal-to-noise ratio
    noise_floor = np.percentile(np.abs(samples), 10)
    signal_peak = np.percentile(np.abs(samples), 90)
    snr = 20 * np.log10(signal_peak / noise_floor) if noise_floor > 0 else 0

    # Calculate clarity score based on audio dynamics
    rms = np.sqrt(np.mean(samples**2))
    peak = np.max(np.abs(samples))
    clarity = rms / peak if peak > 0 else 0

    # Overall quality score (0-100)
    quality_score = min(100, (snr * 0.6 + clarity * 40))

    return {
        'signal_to_noise': snr,
        'clarity_score': clarity * 100,
        'overall_quality': quality_score
    }

def _process_single_video_audio(args):
    """Worker function to process audio for a single video."""
    video_dict, track_volume, noise_reduce_flag = args
    filepath = video_dict['filepath']
    try:
        audio = AudioSegment.from_file(filepath)
        duration = len(audio) / 1000.0
        quality = assess_audio_quality(audio)
        quality_score = quality['overall_quality']

        # Use audio_utils instead of local implementation
        enhanced_audio = audio_utils.process_audio_track(
            audio,
            noise_reduction_threshold=0.1 if noise_reduce_flag else 0,
            normalization_level=-3.0
        )

        if enhanced_audio and track_volume != 0:
            enhanced_audio = enhanced_audio + track_volume

        return (filepath, enhanced_audio, quality_score, duration)
    except Exception as e:
        logger.error(f"Error processing audio for {filepath}: {str(e)}")
        return (filepath, None, 0, 0)

def process_video_audio(video_files, track_volumes=None, overall_volume=0, noise_reduce=True, progress_callback=None):
    """
    Processes audio from multiple video sources in parallel, applies enhancements,
    adjusts volumes, and combines them.

    Args:
        video_files (list): List of video file dictionaries.
        track_volumes (dict, optional): Dictionary mapping filepath to volume adjustment in dB. Defaults to None.
        overall_volume (float, optional): Overall volume adjustment in dB. Defaults to 0.
        noise_reduce (bool, optional): Whether to apply noise reduction. Defaults to True.
        progress_callback (function, optional): Callback for progress updates. Defaults to None.

    Returns:
        str: Path to the final combined and enhanced audio file, or None if processing fails.
    """
    try:
        if not video_files:
            raise AudioProcessingError("No video files provided")

        total_videos = len(video_files)
        if total_videos == 0:
            logger.info("No video files to process.")
            return None

        sorted_videos = sorted(video_files, key=lambda x: x['timestamp'])
        processed_count = 0
        results_map = {}

        # Prepare arguments for the worker function, including volume settings
        process_args = []
        for video in sorted_videos:
            filepath = video['filepath']
            # Get specific track volume or default to 0 dB change
            track_vol = track_volumes.get(filepath, 0) if track_volumes else 0
            process_args.append((video, track_vol, noise_reduce))

        total_files = len(sorted_videos)
        processed_files = 0

        with multiprocessing.Pool() as pool:
            future_results = pool.imap_unordered(_process_single_video_audio, process_args)

            for result in future_results:
                filepath, enhanced_audio, quality_score, duration = result
                processed_files += 1

                if enhanced_audio is not None:
                    results_map[filepath] = (enhanced_audio, quality_score, duration)
                else:
                    logger.warning(f"Failed to process audio for {filepath}")
                    results_map[filepath] = (None, 0, 0)

                if progress_callback:
                    # First 80% for processing individual files
                    progress = (processed_files / total_files) * 80
                    progress_callback(progress)

        # Combine audio segments with remaining 20% progress
        if progress_callback:
            progress_callback(80)  # Signal start of combination phase

        final_audio = AudioSegment.empty()
        combined_files = 0

        for video in sorted_videos:
            filepath = video['filepath']
            if filepath in results_map:
                enhanced_audio, quality_score, _ = results_map[filepath]
                video['audio_quality'] = quality_score

                if enhanced_audio:
                    final_audio += enhanced_audio
                    combined_files += 1
                    if progress_callback:
                        # Last 20% for combining files
                        progress = 80 + (combined_files / total_files) * 20
                        progress_callback(min(progress, 100))

        # Export final audio
        output_dir = "output"       # TODO: Use the output directory from config
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "final_enhanced_audio.wav")

        if len(final_audio) > 0:
            # Apply overall volume adjustment
            if overall_volume != 0:
                final_audio = final_audio + overall_volume

            final_audio.export(output_path, format="wav")
            logger.info(f"Final enhanced audio exported to {output_path}")
            return output_path
        else:
            logger.warning("No audio data processed successfully, final audio is empty.")
            return None

    except CouldntDecodeError as e:
        raise AudioProcessingError(f"Failed to decode audio: {str(e)}")
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        raise AudioProcessingError(str(e))

def parallel_audio_processing(video_files, noise_reduction_threshold=0.1, normalization_level=-3.0,
                            eq_settings=None, progress_callback=None):
    """Process audio tracks in parallel with synchronization markers."""
    processed_tracks = []
    sync_markers = []

    def process_track(filepath):
        try:
            track = AudioSegment.from_file(filepath)
            duration_ms = len(track)

            # Use audio_utils for processing
            track = audio_utils.process_audio_track(
                track,
                noise_reduction_threshold=noise_reduction_threshold,
                normalization_level=normalization_level,
                eq_settings=eq_settings
            )

            return (filepath, track, duration_ms)
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return (filepath, None, 0)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    # Process tracks in parallel
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {executor.submit(process_track, v['filepath']): v for v in video_files}

        for i, future in enumerate(as_completed(futures)):
            filepath, track, duration_ms = future.result()

            if track:
                # Create sync marker with original timestamp and duration
                video = futures[future]
                sync_marker = {
                    'filepath': filepath,
                    'timestamp': video['timestamp'],
                    'duration_ms': duration_ms,
                    'start_frame': 0,  # Will be updated during alignment
                    'end_frame': int(duration_ms * track.frame_rate / 1000)
                }

                processed_tracks.append(track)
                sync_markers.append(sync_marker)

            if progress_callback:
                progress_callback((i + 1) / len(video_files) * 100)

    return processed_tracks, sync_markers
