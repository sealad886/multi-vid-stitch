import logging

_logger = None

def setup_logger(name=__name__, log_level="INFO", log_file_path=None):
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if log_file_path is not None:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def set_logger(logger_instance=setup_logger(log_file_path='multicam_stitch.log')):
    global _logger
    _logger = logger_instance

def get_logger():
    return _logger

set_logger()  # Initialize the logger with default settings
