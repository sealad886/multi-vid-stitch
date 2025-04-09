from .logger_manager import get_logger
logger = get_logger()
from .config_parser import parse_config
config = parse_config()

import sys
import argparse
import os
from .run import main

if __name__ == "__main__":
    # Log the final configuration being used (from config_parser)
    logger.info(f"Using Input Directory: {config['input_dir']}")
    logger.info(f"Using Output Directory: {config['output_dir']}")
    logger.info(f"Using Log Level: {config['log_level']}")
    logger.info(f"Noise Reduction Enabled: {config['noise_reduction']}")
    logger.info(f"Overall Volume Adjustment: {config['overall_volume']} dB")
    if config['track_volumes']:
        logger.info(f"Track-specific volumes loaded: {len(config['track_volumes'])} entries")
    config_file_path = config.get('config_file_path') # Might be None
    if config_file_path and os.path.exists(config_file_path):
        logger.info(f"Loaded configuration from: {config_file_path}")
    elif config.get('config'): # Raw argparser value
        logger.warning(f"Specified config file '{config['config']}' not found.")


    try:
        logger.info("Starting video stitching process...")
        # Pass the determined settings to the main function
        main(
            input_dir=config['input_dir'],
            output_dir=config['output_dir'],
            noise_reduction=config['noise_reduction'],
            overall_volume=config['overall_volume'],
            track_volumes=config['track_volumes'],
            feature_detector=config['feature_detector'],
            stitcher_type=config['stitcher_type'],
            overlap_percentage=config['overlap_percentage'],
            blending_mode=config['blending_mode'],
            perspective_correction=config['perspective_correction'],
            output_format=config['output_format'],
            codec=config['codec'],
            encoding_params=config['encoding_params']
        )
        logger.info("Video stitching process completed successfully.")
    except Exception as e:
        logger.critical(f"Unhandled exception during main execution: {e}", exc_info=True) # Log traceback
        sys.exit(1)
