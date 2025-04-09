# multicam_stitch/file_parser.py
import os
import re
from datetime import datetime

from . import exceptions # Ensure exceptions are imported if needed for error handling
from .logger_manager import get_logger
logger = get_logger()

def parse_video_filename(filename):
    """
    Parses the video filename to extract timestamp, camera name, and sequence ID.

    Filename format: HH-MM-ss_CameraName_seqID.mp4

    Args:
        filename (str): The filename of the video clip.

    Returns:
        tuple: A tuple containing datetime object, camera name, and sequence ID, or None if parsing fails.
    """
    pattern = re.compile(r"(\d{2})-(\d{2})-(\d{2})_([a-zA-Z0-9]+)_(\w+)\.mp4")
    match = pattern.match(filename)
    if match:
        try:
            hour_str, minute_str, second_str, camera_name, seq_id = match.groups()
            # Combine date and time (assuming current date for simplicity)
            current_date = datetime.now().date()
            time_object = datetime.strptime(f"{hour_str}:{minute_str}:{second_str}", "%H:%M:%S").time()
            timestamp = datetime.combine(current_date, time_object)
            return timestamp, camera_name, seq_id
        except ValueError as e:
            logger.error(f"Filename parsing error for '{filename}': {e}")
            return None
    return None

def get_video_files(input_dir):
    """
    Lists all video files in the input directory and parses their filenames.

    Args:
        input_dir (str): Path to the input directory containing video files.

    Returns:
        list: A list of tuples, each containing filename, timestamp, camera name, and sequence ID.
    """
    # Check if input directory exists
    if not os.path.isdir(input_dir):
        raise exceptions.FileParsingError(f"Input directory not found: {input_dir}")

    video_files = []
    invalid_formats = [filename for filename in os.listdir(input_dir) if not filename.lower().endswith('.mp4')]
    try:
        for filename in os.listdir(input_dir):
            filepath = os.path.join(input_dir, filename)

            # Validate file extension - only .mp4 files are supported
            if not filename.lower().endswith(".mp4"):
                logger.warning(f"Skipping non-MP4 file: {filename}")
                continue

            # Check if file exists and has read permissions
            if not os.path.exists(filepath):
                raise exceptions.FileParsingError(f"Video file not found: {filepath}")

            if not os.access(filepath, os.R_OK):
                raise exceptions.FileParsingError(f"No read permission for video file: {filepath}")

            parsed_info = parse_video_filename(filename)
            if parsed_info:
                timestamp, camera_name, seq_id = parsed_info
                video_files.append({
                    'filepath': filepath,
                    'filename': filename,
                    'timestamp': timestamp,
                    'camera_name': camera_name,
                    'seq_id': seq_id
                })
    except FileNotFoundError:
        raise exceptions.FileParsingError(f"Input directory not found: {input_dir}")
    except PermissionError:
        raise exceptions.FileParsingError(f"Permission denied to access directory: {input_dir}")

    if len(video_files) < 2:
        raise exceptions.FileParsingError("At least two video files are required for stitching.")

    # Check if all files have valid MP4 format
    if invalid_formats:
        logger.error(f"Invalid file formats: {invalid_formats}")
        raise exceptions.FileParsingError("All input files must be in .mp4 format.")

    return video_files
