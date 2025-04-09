# multicam_stitch/video_processing.py
import multiprocessing
from multiprocessing import Queue, Pipe, Manager
import concurrent.futures
import threading
import queue
import time
import cv2
import datetime
import os
from .exceptions import VideoProcessingError, SynchronizationError
import platform
import subprocess
import numpy as np
import asyncio
from concurrent.futures import ProcessPoolExecutor
import tracemalloc
import psutil
import contextlib
import threading
import functools

from .exceptions import (
    FrameProcessingError, FrameSizeMismatchError,
    InvalidInputError, CodecUnsupportedError,
    StitchingBoundaryError,
)

# Global process pool executor
process_pool = None

from .logger_manager import get_logger
logger = get_logger()

# Optimization configuration defaults
CONFIG = {
    'batch_size': 16,
    'num_workers': max(1, (os.cpu_count() or 4) - 1),
    'prefetch_batches': 4,
    'max_queue_size': 64,
    'enable_async': True,
    'enable_process_pool': True,
    'metrics_logging': True
}

CONFIG.update({
    'queue_timeout': 5,
    'max_retries': 5,
    'backoff_base': 0.1,
    'watchdog_timeout': 30,
    'global_stage_timeout': 300
})


def exponential_backoff_retry(func, *args, max_retries=None, timeout=None, backoff_base=None, **kwargs):
    max_retries = max_retries if max_retries is not None else CONFIG['max_retries']
    timeout = timeout if timeout is not None else CONFIG['queue_timeout']
    backoff_base = backoff_base if backoff_base is not None else CONFIG['backoff_base']
    attempt = 0
    while attempt < max_retries:
        try:
            return func(*args, timeout=timeout, **kwargs)
        except (queue.Empty, queue.Full):
            sleep_time = backoff_base * (2 ** attempt)
            logger.warning(f"Retry {attempt + 1}/{max_retries} after {sleep_time:.2f}s due to queue timeout")
            time.sleep(sleep_time)
            attempt += 1
    logger.error(f"Operation {func.__name__} failed after {max_retries} retries")
    raise


@contextlib.contextmanager
def acquire_locks_ordered(*locks, timeout=CONFIG['queue_timeout']):
    locks = sorted(locks, key=lambda x: id(x))  # ordered acquisition
    acquired = []
    try:
        for lock in locks:
            acquired_success = lock.acquire(timeout=timeout)
            if not acquired_success:
                logger.warning("Failed to acquire lock within timeout")
                raise TimeoutError("Lock acquisition timed out")
            acquired.append(lock)
        yield
    finally:
        for lock in reversed(acquired):
            try:
                lock.release()
            except RuntimeError:
                pass


@contextlib.contextmanager
def watchdog_timer(name, timeout=CONFIG['watchdog_timeout']):
    timer = threading.Timer(timeout, lambda: logger.error(f"Watchdog timeout in {name} after {timeout}s"))
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


async def async_run_subprocess(cmd):
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)
    return stdout, stderr


def log_metrics(label):
    if not CONFIG.get('metrics_logging'):
        return
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    logger.info(f"[METRICS] {label}: Memory usage = {mem:.2f} MB")

    current, peak = tracemalloc.get_traced_memory()
    logger.info(f"[METRICS] {label}: Tracemalloc current={current/1024/1024:.2f} MB, peak={peak/1024/1024:.2f} MB")


def align_frames(reference_frame, frame_to_align, feature_detector='orb', use_gpu=False):
    """Align frames using feature detection and matching with enhanced error handling."""
    try:
        if reference_frame is None or frame_to_align is None:
            raise FrameProcessingError("Invalid input frames (None)")

        if reference_frame.shape != frame_to_align.shape:
            raise FrameSizeMismatchError(
                f"Frame size mismatch: ref {reference_frame.shape} vs {frame_to_align.shape}"
            )

        # Initialize feature detector with error handling
        try:
            if feature_detector == 'orb':
                detector = cv2.ORB.create(
                    nfeatures=2000,
                    scaleFactor=1.2,
                    nlevels=8,
                    edgeThreshold=31,
                    firstLevel=0,
                    WTA_K=2,
                    patchSize=31,
                    fastThreshold=20
                )
            elif feature_detector == 'sift':
                detector = cv2.SIFT.create(
                    nfeatures=2000,
                    nOctaveLayers=3,
                    contrastThreshold=0.04,
                    edgeThreshold=10,
                    sigma=1.6
                )
            else:
                raise InvalidInputError(f"Invalid feature detector: {feature_detector}")
        except cv2.error as e:
            raise CodecUnsupportedError(f"Feature detector initialization failed: {str(e)}")

        # Detect keypoints and compute descriptors
        try:
            kp1, des1 = detector.detectAndCompute(reference_frame, None)
            kp2, des2 = detector.detectAndCompute(frame_to_align, None)
        except cv2.error as e:
            raise FrameProcessingError(f"Feature detection failed: {str(e)}")

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            logger.warning("Insufficient features for alignment, returning original frame")
            return frame_to_align

        # Create matcher
        try:
            if feature_detector == 'orb':
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        except cv2.error as e:
            raise FrameProcessingError(f"Matcher creation failed: {str(e)}")

        # Match descriptors
        try:
            matches = matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
        except cv2.error as e:
            raise FrameProcessingError(f"Feature matching failed: {str(e)}")

        # Extract location of good matches
        try:
            points1 = np.zeros((len(matches), 2), dtype=np.float32)
            points2 = np.zeros((len(matches), 2), dtype=np.float32)

            for i, match in enumerate(matches):
                points1[i, :] = kp1[match.queryIdx].pt
                points2[i, :] = kp2[match.trainIdx].pt
        except Exception as e:
            raise FrameProcessingError(f"Match point extraction failed: {str(e)}")

        # Find homography
        try:
            h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
        except cv2.error as e:
            raise StitchingBoundaryError(f"Homography calculation failed: {str(e)}")

        # Use homography
        try:
            height, width = reference_frame.shape[:2]
            aligned = cv2.warpPerspective(frame_to_align, h, (width, height))
            return aligned
        except cv2.error as e:
            raise FrameProcessingError(f"Frame warping failed: {str(e)}")

    except FrameProcessingError as e:
        logger.error(f"Frame alignment failed: {e}")
        raise
    except FrameSizeMismatchError as e:
        logger.error(f"Frame size mismatch: {e}")
        raise
    except StitchingBoundaryError as e:
        logger.error(f"Stitching boundary error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in frame alignment: {e}")
        raise FrameProcessingError(f"Frame alignment failed: {str(e)}")

def assess_video_quality(video_path, desired_speaker_count=1):
    """
    Enhanced quality assessment that considers multiple speakers and view types.
    Returns a dictionary with quality metrics.
    """
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise VideoProcessingError("Face cascade classifier not loaded")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoProcessingError(f"Cannot open video file: {video_path}")

        # Sample frames for quality assessment
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = min(100, total_frames)
        step = max(1, total_frames // sample_frames)

        sharpness_scores = []
        brightness_scores = []
        contrast_scores = []
        face_counts = []

        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate sharpness (variance of Laplacian)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_scores.append(sharpness)

            # Calculate brightness and contrast
            brightness = np.mean(gray)
            contrast = gray.std()
            brightness_scores.append(brightness)
            contrast_scores.append(contrast)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            face_counts.append(len(faces))

        cap.release()

        # Calculate overall quality score (0-100)
        avg_sharpness = np.mean(sharpness_scores) if sharpness_scores else 0
        avg_brightness = np.mean(brightness_scores) if brightness_scores else 0
        avg_contrast = np.mean(contrast_scores) if contrast_scores else 0
        avg_faces = np.mean(face_counts) if face_counts else 0

        # Normalize and weight metrics
        sharpness_score = min(avg_sharpness / 100, 1) * 40  # Max 40 points
        brightness_score = (1 - abs(avg_brightness - 128)/128) * 30  # Max 30 points
        contrast_score = min(avg_contrast / 64, 1) * 20  # Max 20 points
        face_score = min(avg_faces / desired_speaker_count, 1) * 10  # Max 10 points

        overall_score = sharpness_score + brightness_score + contrast_score + face_score

        return {
            'overall_score': overall_score,
            'sharpness': avg_sharpness,
            'brightness': avg_brightness,
            'contrast': avg_contrast,
            'face_count': avg_faces
        }

    except Exception as e:
        logger.error(f"Error in assess_video_quality: {e}")
        raise VideoProcessingError(f"Failed to assess video quality: {str(e)}")

def get_frame_rate(filepath):
    """Get the frame rate of a video file."""
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise VideoProcessingError(f"Cannot open video file: {filepath}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps
    except Exception as e:
        logger.error(f"Error getting frame rate for {filepath}: {e}")
        raise VideoProcessingError(f"Failed to get frame rate: {str(e)}")

def get_video_duration(filepath):
    """Get the duration of a video file in seconds."""
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise VideoProcessingError(f"Cannot open video file: {filepath}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
        return duration
    except Exception as e:
        logger.error(f"Error getting video duration for {filepath}: {e}")
        raise VideoProcessingError(f"Failed to get video duration: {str(e)}")

def detect_speakers(video_path, desired_speaker_count=1):
    try:
        # Extract audio from video
        audio_path = video_path.replace('.mp4', '.wav')
        command = f"ffmpeg -y -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {audio_path}"
        try:
            # Check if video has an audio stream
            probe_command = f"ffprobe -v error -show_streams -select_streams a:0 {video_path}"
            probe_result = subprocess.run(probe_command, shell=True, capture_output=True, text=True)
            if probe_result.returncode != 0 or not probe_result.stdout:
                logger.warning(f"No audio stream found in {video_path}, skipping speaker detection")
                return []
        except Exception as e:
            logger.error(f"Error checking audio stream for {video_path}: {e}")
        try:
            if CONFIG.get('enable_async'):
                import asyncio
                asyncio.run(async_run_subprocess(command))
            elif CONFIG.get('enable_process_pool') and process_pool is not None:
                future = process_pool.submit(subprocess.run, command, shell=True, check=True, capture_output=True)
                result = future.result()
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout, stderr=result.stderr)
            else:
                subprocess.run(command, shell=True, check=True)
        except Exception as e:
            logger.error(f"Audio extraction failed for {video_path}: {e}")
            raise VideoProcessingError(f"Audio extraction failed: {str(e)}")
    except Exception as e:
        logger.error(f"Audio extraction failed for {video_path}: {e}")
        raise VideoProcessingError(f"Audio extraction failed: {str(e)}")

def synchronize_clips(video_files, time_window_minutes=2, progress_callback=None, manual_offsets=None):
    """
    Enhanced synchronization with parallel analysis, progress reporting, and manual offset adjustments.
    """
    if not video_files:
        raise SynchronizationError("No video files provided")

    try:
        # Validate all files exist before processing
        missing_files = [vf['filepath'] for vf in video_files if not os.path.exists(vf['filepath'])]
        if missing_files:
            raise VideoProcessingError(f"Missing video files: {', '.join(missing_files)}")

        import time
        start_time = time.time()

        # Sort clips by timestamp
        sorted_clips = sorted(video_files, key=lambda x: x['timestamp'])
        total_clips = len(sorted_clips)
        analysis_results = {}

        logger.info(f"Starting parallel analysis of {total_clips} video clips...")
        def _analyze_single_clip(clip):
            try:
                filepath = clip['filepath']
                speaker_segments = detect_speakers(filepath)
                quality_assessment = assess_video_quality(filepath)
                return filepath, speaker_segments, quality_assessment
            except Exception as e:
                logger.error(f"Error analyzing clip {clip['filepath']}: {e}")
                return clip['filepath'], None, None

        with multiprocessing.Pool() as pool:
            future_results = pool.imap_unordered(_analyze_single_clip, sorted_clips)
            processed_count = 0

            for result in future_results:
                filepath, speaker_segments, quality_assessment = result
                processed_count += 1

                if speaker_segments is not None and quality_assessment is not None:
                    analysis_results[filepath] = {
                        'speaker_segments': speaker_segments,
                        'quality_assessment': quality_assessment
                    }
                else:
                    logger.warning(f"Analysis failed for clip: {filepath}")
                    analysis_results[filepath] = None

                if progress_callback:
                    # First 50% for analysis
                    progress = (processed_count / total_clips) * 50
                    progress_callback(progress)

        logger.info("Parallel analysis complete.")

        updated_clips = []
        for clip in sorted_clips:
            filepath = clip['filepath']
            if filepath in analysis_results and analysis_results[filepath] is not None:
                clip.update(analysis_results[filepath])
                if manual_offsets and filepath in manual_offsets:
                    clip['timestamp'] += datetime.timedelta(seconds=manual_offsets[filepath])
                updated_clips.append(clip)
            else:
                logger.warning(f"Skipping clip {filepath} due to analysis failure.")

        sorted_clips = updated_clips
        if not sorted_clips:
            logger.error("No clips were successfully analyzed. Cannot proceed with synchronization.")
            return []

        # Determine the most common frame rate using parallel processing
        frame_rates = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(get_frame_rate, clip['filepath']) for clip in sorted_clips]
            for future in concurrent.futures.as_completed(futures):
                frame_rates.append(future.result())

        fps_counts = {}
        for fps in frame_rates:
            if fps in fps_counts:
                fps_counts[fps] += 1
            else:
                fps_counts[fps] = 1
        most_common_fps = max(fps_counts, key=fps_counts.get)

        logger.info("Starting second pass: Creating synchronized segments...")

        # Calculate end_time using parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(get_video_duration, clip['filepath']) for clip in sorted_clips]
            durations = [future.result() for future in futures]

        end_time = max(clip['timestamp'] + datetime.timedelta(seconds=duration) for clip, duration in zip(sorted_clips, durations))
        current_time = min(clip['timestamp'] for clip in sorted_clips)

        segment_duration = datetime.timedelta(seconds=1.0 / most_common_fps)
        segments_processed = 0
        total_segments = int((end_time - current_time).total_seconds() * most_common_fps)

        # Cache all durations to avoid redundant calls
        durations_dict = {}
        for clip, duration in zip(sorted_clips, durations):
            durations_dict[clip['filepath']] = duration

        def process_segment_chunk(chunk_start, chunk_end):
            chunk_segments = []
            current_time = chunk_start
            while current_time < chunk_end:
                best_clip_segment = None
                best_quality_score = -1

                for clip in sorted_clips:
                    clip_start_time = clip['timestamp']
                    clip_end_time = clip_start_time + datetime.timedelta(seconds=durations_dict[clip['filepath']])

                    if clip_start_time - datetime.timedelta(minutes=time_window_minutes) <= current_time <= clip_end_time + datetime.timedelta(minutes=time_window_minutes):
                        quality_score = clip['quality_assessment']['overall_score']
                        speaker_segments = clip['speaker_segments']

                        has_speakers_in_segment = any(segment['start_time'] <= current_time.timestamp() <= segment['end_time'] for segment in speaker_segments)

                        if has_speakers_in_segment:
                            if isinstance(quality_score, cv2.UMat):
                                quality_score = quality_score.get()
                            if isinstance(best_quality_score, cv2.UMat):
                                best_quality_score = best_quality_score.get()
                            if quality_score > best_quality_score:
                                best_quality_score = quality_score
                                best_clip_segment = clip
                        elif best_clip_segment is None:
                            if isinstance(quality_score, cv2.UMat):
                                quality_score = quality_score.get()
                            if isinstance(best_quality_score, cv2.UMat):
                                best_quality_score = best_quality_score.get()
                            if quality_score > best_quality_score:
                                best_quality_score = quality_score
                                best_clip_segment = clip

                if best_clip_segment:
                    chunk_segments.append({
                        'start_time': current_time,
                        'end_time': current_time + segment_duration,
                        'filepath': best_clip_segment['filepath'],
                        'quality_score': best_quality_score
                    })
                else:
                    print(f"Warning: No suitable clip found for time {current_time}")

                current_time += segment_duration

            return chunk_segments

        # Divide timeline into chunks for parallel processing
        num_workers = multiprocessing.cpu_count()
        total_seconds = (end_time - current_time).total_seconds()
        chunk_size_seconds = total_seconds / num_workers
        chunk_ranges = []
        for i in range(num_workers):
            chunk_start = current_time + datetime.timedelta(seconds=i * chunk_size_seconds)
            chunk_end = current_time + datetime.timedelta(seconds=(i + 1) * chunk_size_seconds)
            if chunk_end > end_time:
                chunk_end = end_time
            chunk_ranges.append((chunk_start, chunk_end))

        synchronized_segments = []
        segments_processed = 0

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = []
            for chunk_start, chunk_end in chunk_ranges:
                results.append(pool.apply_async(process_segment_chunk, args=(chunk_start, chunk_end)))
            for r in results:
                chunk_segments = r.get()
                synchronized_segments.extend(chunk_segments)
                segments_processed += len(chunk_segments)
                if progress_callback:
                    progress = 50 + (segments_processed / total_segments) * 50
                    progress = min(progress, 100)
                    logger.info(f"segments_processed: {segments_processed}, total_segments: {total_segments}, progress: {progress}")
                    progress_callback(progress)

        synchronized_segments.sort(key=lambda seg: seg['start_time'])

        print(f"Second pass complete. Created {len(synchronized_segments)} synchronized segments.")

    except Exception as e:
        logger.error(f"Unexpected error during synchronization: {e}")
        raise SynchronizationError(f"Error synchronizing clips: {str(e)}")
    return synchronized_segments
