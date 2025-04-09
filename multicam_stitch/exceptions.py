class VideoProcessingError(Exception):
    """Base exception for video processing errors"""
    pass

class AudioProcessingError(Exception):
    """Base exception for audio processing errors"""
    pass

class FileParsingError(Exception):
    """Base exception for file parsing errors"""
    pass

class SynchronizationError(Exception):
    """Base exception for synchronization errors"""
    pass

class InvalidInputError(Exception):
    """Base exception for invalid input errors"""
    pass

class ConfigurationError(Exception):
    """Base exception for configuration errors"""
    pass

class GPUInitializationError(Exception):
    """Base exception for GPU initialization errors"""
    pass

class FrameProcessingError(VideoProcessingError):
    """Exception raised when frame processing fails"""
    pass

class CodecUnsupportedError(VideoProcessingError):
    """Exception raised when video codec is not supported"""
    pass

class InvalidColorSpaceError(VideoProcessingError):
    """Exception raised for invalid or unsupported color space"""
    pass

class StitchingError(VideoProcessingError):
    """Exception raised when video stitching fails"""
    pass

class StitchingBoundaryError(VideoProcessingError):
    """Exception raised when video stitching exceeds boundaries"""
    pass

class FrameSizeMismatchError(SynchronizationError):
    """Exception raised when frame sizes don't match"""
    pass

class FrameRateMismatchError(SynchronizationError):
    """Exception raised when frame rates don't match"""
    pass

class AudioVideoSyncError(SynchronizationError):
    """Exception raised when audio and video are out of sync"""
    pass

class TimestampMismatchError(SynchronizationError):
    """Exception raised when timestamps don't match"""
    pass

class GPUMemoryError(GPUInitializationError):
    """Exception raised when GPU memory is insufficient"""
    pass

class OutputWriteError(FileParsingError):
    """Exception raised when output file cannot be written"""
    pass

class ProcessingError(Exception):
    """General exception for processing errors"""
    pass
