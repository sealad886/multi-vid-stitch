from .exceptions import (
    VideoProcessingError, GPUInitializationError, FileParsingError,
    GPUMemoryError, InvalidInputError, FrameProcessingError,
    CodecUnsupportedError, OutputWriteError, FrameRateMismatchError,
    FrameSizeMismatchError
)
from .gpu_setup import create_gpu_mat, get_gpu_device
from .video_processing import CONFIG, align_frames, exponential_backoff_retry, logger, watchdog_timer
import cv2
import concurrent.futures
import heapq
import json
import multiprocessing
import os
import queue
import threading
import time


def stitch_videos(video_files, stitcher_type='feature', feature_detector='orb', progress_callback=None, overlap_percentage=0.5, blending_mode='feather', perspective_correction=True, output_format='mp4', codec='mp4v', encoding_params=None, manual_stitching_order=None, num_threads=1):
    """
    Enhanced with multiprocessing-based priority queues and metadata IPC.
    """
    class PriorityQueueMP:
        def __init__(self, maxsize=100):
            self.maxsize = maxsize
            self.queue = multiprocessing.Queue(maxsize)
            self.heap = []
            self.lock = multiprocessing.Lock()

        def _put_once(self, item, priority=1, timeout=5):
            with self.lock:
                if len(self.heap) < self.maxsize:
                    heapq.heappush(self.heap, (priority, time.time(), item))
                    return True
                else:
                    raise queue.Full

        def put(self, item, priority=1, block=True, timeout=None):
            timeout = timeout if timeout is not None else CONFIG['queue_timeout']
            with watchdog_timer("PriorityQueueMP.put", timeout=CONFIG['watchdog_timeout']):
                try:
                    return exponential_backoff_retry(self._put_once, item, priority, timeout=timeout)
                except Exception as e:
                    logger.error(f"PriorityQueueMP.put failed after retries: {e}")
                    raise

        def _get_once(self, timeout=5):
            with self.lock:
                if self.heap:
                    return heapq.heappop(self.heap)
                else:
                    raise queue.Empty

        def get(self, block=True, timeout=None):
            timeout = timeout if timeout is not None else CONFIG['queue_timeout']
            with watchdog_timer("PriorityQueueMP.get", timeout=CONFIG['watchdog_timeout']):
                try:
                    return exponential_backoff_retry(self._get_once, timeout=timeout)
                except Exception as e:
                    logger.error(f"PriorityQueueMP.get failed after retries: {e}")
                    raise
        def qsize(self):
            with self.lock:
                return len(self.heap)

        def empty(self):
            with self.lock:
                return len(self.heap) == 0

        def full(self):
            with self.lock:
                return len(self.heap) >= self.maxsize

        def close(self):
            pass  # No explicit close needed

    # Metadata channel: high-priority, small queue
    metadata_queue = multiprocessing.Queue(maxsize=20)
    # Duplex pipe for bidirectional metadata/control
    meta_parent_conn, meta_child_conn = multiprocessing.Pipe(duplex=True)
    """
    GPU-accelerated video stitching using OpenCV.
    """
    try:
        # GPU initialization with error handling
        try:
            use_gpu = get_gpu_device()
        except GPUInitializationError as e:
            logger.warning(f"GPU initialization failed, falling back to CPU: {e}")
            use_gpu = False
        except Exception as e:
            logger.error(f"Unexpected GPU initialization error: {e}")
            raise GPUMemoryError(f"GPU initialization failed: {str(e)}")

        if manual_stitching_order:
            video_files = sorted(video_files, key=lambda x: manual_stitching_order.index(x['filepath']))
        video_paths = [vf['filepath'] for vf in video_files]
        if not video_paths:
            raise InvalidInputError("No video paths provided")

        total_frames = 0
        processed_frames = 0

        # First pass: count total frames with enhanced error handling
        for path in video_paths:
            cap = None
            try:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise FileParsingError(f"Cannot open video file: {path}")

                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count <= 0:
                    raise FrameProcessingError(f"Invalid frame count in video file: {path}")

                total_frames += frame_count
            except cv2.error as e:
                logger.error(f"OpenCV error while processing {path}: {e}")
                raise CodecUnsupportedError(f"Video codec error: {str(e)}")
            except FrameProcessingError as e:
                logger.error(f"Frame processing error: {e}")
                raise
            except Exception as e:
                logger.error(f"Error reading video file {path}: {e}")
                raise FileParsingError(f"Error reading video file: {str(e)}")
            finally:
                if cap and cap.isOpened():
                    cap.release()

        # Output directory handling
        try:
            if not os.path.exists("output"):
                os.makedirs("output")
        except OSError as e:
            logger.error(f"Failed to create output directory: {e}")
            raise OutputWriteError(f"Failed to create output directory: {str(e)}")

        output_path = os.path.join("output", f"stitched_video.{output_format}")
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
        except cv2.error as e:
            raise CodecUnsupportedError(f"Unsupported codec '{codec}': {str(e)}")

        # Determine optimal resolution and frame rate with error handling
        frame_widths = []
        frame_heights = []
        frame_rates = []
        for path in video_paths:
            cap = None
            try:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise FileParsingError(f"Cannot open video file: {path}")

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                if width <= 0 or height <= 0:
                    raise FrameSizeMismatchError(f"Invalid frame dimensions in {path}")
                if fps <= 0:
                    raise FrameRateMismatchError(f"Invalid frame rate in {path}")

                frame_widths.append(width)
                frame_heights.append(height)
                frame_rates.append(fps)
            except FrameSizeMismatchError as e:
                logger.error(f"Frame size error: {e}")
                raise
            except FrameRateMismatchError as e:
                logger.error(f"Frame rate error: {e}")
                raise
            except Exception as e:
                logger.error(f"Error reading video properties from {path}: {e}")
                raise VideoProcessingError(f"Error reading video properties: {str(e)}")
            finally:
                if cap and cap.isOpened():
                    cap.release()

        max_width = max(frame_widths) if frame_widths else 1280
        max_height = max(frame_heights) if frame_heights else 720

        # Find most common frame rate with validation
        if not frame_rates:
            raise FrameRateMismatchError("No valid frame rates found in input videos")

        fps_counts = {}
        for fps in frame_rates:
            if fps in fps_counts:
                fps_counts[fps] += 1
            else:
                fps_counts[fps] = 1
        most_common_fps = max(fps_counts, key=fps_counts.get)

        # Load encoding parameters from JSON if provided
        encoding_parameters = {}
        if encoding_params:
            try:
                if os.path.isfile(encoding_params):
                    with open(encoding_params, 'r') as f:
                        encoding_parameters = json.load(f)
                else:
                    encoding_parameters = json.loads(encoding_params)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in encoding parameters: {e}")
                raise InvalidInputError(f"Invalid encoding parameters JSON: {str(e)}")
            except FileNotFoundError as e:
                logger.error(f"Encoding parameters file not found: {e}")
                raise FileParsingError(f"Encoding parameters file not found: {str(e)}")
            except Exception as e:
                logger.error(f"Error loading encoding parameters: {e}")
                raise VideoProcessingError(f"Error loading encoding parameters: {str(e)}")

        # Second pass: process frames with accurate progress
        if stitcher_type == 'opencv':
            try:
                # Prepare images for stitching
                images = []
                for path in video_paths:
                    cap = cv2.VideoCapture(path)
                    if not cap.isOpened():
                        raise VideoProcessingError(f"Cannot open video file: {path}")
                    ret, frame = cap.read()
                    if not ret:
                        raise VideoProcessingError(f"Cannot read frames from video file: {path}")
                    if use_gpu:
                        frame = create_gpu_mat(frame)
                    images.append(frame)
                    cap.release()

                # Initialize Stitcher
                stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
                if blending_mode == 'multi-band':
                    stitcher.setBlendingMode(cv2.Stitcher_BlendingMode_MULTI_BAND)
                else:
                    stitcher.setBlendingMode(cv2.Stitcher_BlendingMode_FEATHER)

                try:
                    status, stitched = stitcher.stitch(images)
                finally:
                    # Clear GPU memory
                    images.clear()

                if status != cv2.Stitcher_OK:
                    raise VideoProcessingError(f"Stitching failed with status code: {status}")

                if isinstance(stitched, cv2.UMat):
                    stitched = stitched.get()  # Get back to CPU if needed for saving

                # Save the stitched image to a video using determined parameters
                frame_height, frame_width = stitched.shape[:2]
                fps = most_common_fps
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                out.write(stitched)
                out.release()
                return output_path

            except cv2.error as e:
                logger.error(f"OpenCV error while stitching with Stitcher class: {e}")
                raise VideoProcessingError(f"OpenCV error: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing video files for stitching: {e}")
                raise VideoProcessingError(f"Error processing video files for stitching: {str(e)}")
        elif stitcher_type == 'feature':
            import concurrent.futures
            # Feature-based stitching with GPU acceleration
            reference_frame = None
            out = None  # Initialize out variable
            try:
                frame_width = max_width
                frame_height = max_height
                fps = most_common_fps
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                if not out.isOpened():
                    raise VideoProcessingError(f"Cannot open video writer for output path: {output_path}")

                task_queue = PriorityQueueMP(maxsize=100)
                result_queue = multiprocessing.Queue(maxsize=100)
                counter_lock = threading.RLock()
                processed_frames_counter = {'count': 0}

                def worker():
                    try:
                        while True:
                            try:
                                with watchdog_timer("task_queue.get", timeout=CONFIG['watchdog_timeout']):
                                    try:
                                        (priority, _, task_data) = exponential_backoff_retry(task_queue.get, timeout=CONFIG['queue_timeout'])
                                    except Exception as e:
                                        logger.error(f"task_queue.get failed after retries: {e}")
                                        break
                            except queue.Empty:
                                break  # Exit if no more tasks
                            ref_frame, frame, path = task_data
                            start_time = time.time()
                            try:
                                aligned = align_frames(ref_frame, frame, feature_detector, use_gpu)
                            except Exception as e:
                                logger.error(f"Error aligning frame from {path}: {e}")
                                aligned = frame
                            elapsed = time.time() - start_time
                            logger.debug(f"Aligned frame from {path} in {elapsed:.3f}s")

                            with counter_lock:
                                processed_frames_counter['count'] += 1
                                count = processed_frames_counter['count']
                            put_success = False
                            while not put_success:
                                try:
                                    with watchdog_timer("async_result_queue.get", timeout=CONFIG['watchdog_timeout']):
                                        try:
                                            exponential_backoff_retry(result_queue.get, timeout=CONFIG['queue_timeout'])
                                        except Exception as e:
                                            logger.error(f"async_result_queue.get failed after retries: {e}")
                                            break
                                    put_success = True
                                except queue.Full:
                                    logger.warning("Result queue full, retrying...")

                            # Check metadata/control channel
                            if meta_child_conn.poll():
                                meta_msg = meta_child_conn.recv()
                                logger.debug(f"Worker received metadata/control: {meta_msg}")

                    finally:
                        pass  # Cleanup if needed

                # Enqueue tasks with prioritization
                for path in video_paths:
                    cap = cv2.VideoCapture(path)
                    if not cap.isOpened():
                        raise VideoProcessingError(f"Cannot open video file: {path}")

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if use_gpu:
                            frame = create_gpu_mat(frame)
                        if reference_frame is None and path == video_paths[0]:
                            reference_frame = frame
                        else:
                            # Prioritize critical frames
                            priority = 0 if 'critical' in path else 1
                            try:
                                task_queue.put((priority, (reference_frame, frame, path)), priority=priority, timeout=5)
                            except queue.Full:
                                logger.warning("Task queue full, skipping frame from %s", path)

                            # Example: send metadata for first frame
                            if reference_frame is frame:
                                try:
                                    metadata_queue.put({'type': 'metadata', 'info': 'Reference frame set'}, timeout=5)
                                    meta_parent_conn.send({'status': 'ref_frame_set'})
                                except:
                                    pass

                    cap.release()

                # Start worker threads
                max_workers = num_threads or min(8, os.cpu_count() or 4)
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(worker) for _ in range(max_workers)]

                    # Collect results as they become available
                    frames_written = 0
                    # Wait for all tasks to be marked done
                    task_queue.join()

                    # Drain the result queue
                    while True:
                        try:
                            aligned_frame, path, count = result_queue.get(timeout=5)
                        except queue.Empty:
                            # If empty after timeout, assume all results processed
                            break
                        if isinstance(aligned_frame, cv2.UMat):
                            aligned_frame = aligned_frame.get()
                        out.write(aligned_frame)
                        frames_written += 1
                        if progress_callback:
                            progress_callback((count / total_frames) * 100)
                        logger.debug(f"Written aligned frame {count}/{total_frames} from {path}")

                logger.info("All frames aligned and written successfully.")
                return output_path
            except cv2.error as e:
                logger.error(f"OpenCV error while stitching {path}: {e}")
                raise VideoProcessingError(f"OpenCV error: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing video file {path} for stitching: {e}")
                raise VideoProcessingError(f"Error processing video file for stitching: {str(e)}")
            finally:
                if out:
                    out.release()
        else:
            raise VideoProcessingError(f"Invalid stitcher_type: {stitcher_type}. Choose 'feature' or 'opencv'.")

    except VideoProcessingError as e:
        logger.error(f"Video processing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in stitch_videos: {e}")
        raise
