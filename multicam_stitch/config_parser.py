import argparse
import yaml
import os
import sys
import logging
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from shutil import get_terminal_size


def parse_config():
    parser = argparse.ArgumentParser(
        description="Stitch multiple video files based on timestamps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--input-dir',
        default='input_videos',
        help='Directory containing input video files.'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='output',
        help='Directory to save the stitched output video.'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level.'
    )
    parser.add_argument(
        '--log-file',
        default='multicam_stitch.log',
        help='Path to the log file.',
        dest='log_file'
    )
    parser.add_argument(
        '-c', '--config',
        default='config.yaml',
        help=f'Path to a YAML configuration file (default: \'config.yaml\' if it exists). Can specify input/output dirs, log level, overall volume, noise reduction, and track-specific volumes (see README).'
    )
    parser.add_argument(
        '--no-noise-reduction',
        action='store_false',
        dest='noise_reduction',
        help='Disable audio noise reduction.'
    )
    parser.add_argument(
        '--overall-volume',
        type=float,
        help='Overall volume adjustment in dB (e.g., -3 for quieter, 6 for louder).'
    )
    parser.add_argument(
        '--feature-detector',
        default='orb',
        choices=['orb', 'sift'],
        help='Feature detection algorithm to use for video synchronization.'
    )
    parser.add_argument(
        '--stitcher',
        default='feature',
        choices=['feature', 'opencv'],
        help='Stitching algorithm to use.'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.5,
        help='Overlap percentage between videos.'
    )
    parser.add_argument(
        '--blend',
        default='feather',
        choices=['feather', 'multi-band'],
        help='Blending mode to use for stitching.'
    )
    parser.add_argument(
        '--perspective-correction',
        action='store_true',
        help='Enable perspective correction.'
    )
    parser.add_argument(
        '--output-format',
        default='mp4',
        help='Output file format (e.g., mp4, mov, avi).'
    )
    parser.add_argument(
        '--codec',
        default='mp4v',
        help='Video codec to use (e.g., H264, H265, mp4v).'
    )
    parser.add_argument(
        '--encoding-params',
        help='Encoding parameters as a JSON string or path to a JSON file.'
    )

    args = parser.parse_args()

    config = {}
    config_file_path = args.config

    if config_file_path and os.path.exists(config_file_path):
        try:
            with open(config_file_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded configuration from: {config_file_path}")
        except FileNotFoundError:
            if args.config:
                print(f"Error: Configuration file not found: {config_file_path}", file=sys.stderr)
                sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error: Could not decode YAML from configuration file: {config_file_path}\n{e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error loading configuration file {config_file_path}: {e}", file=sys.stderr)
            sys.exit(1)

    input_dir = args.input_dir or config.get('input_dir', args.input_dir)
    output_dir = args.output_dir or config.get('output_dir', args.output_dir)
    log_level = config.get('log_level', args.log_level)
    noise_reduction = args.noise_reduction if args.noise_reduction is not None else config.get('noise_reduction', True)
    overall_volume = args.overall_volume if args.overall_volume is not None else config.get('overall_volume', 0.0)
    track_volumes = config.get('track_volumes', None)
    log_file = config.get('log_file', args.log_file)

    return {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'log_level': log_level,
        'noise_reduction': noise_reduction,
        'overall_volume': overall_volume,
        'track_volumes': track_volumes,
        'feature_detector': args.feature_detector,
        'stitcher_type': args.stitcher,
        'overlap_percentage': args.overlap,
        'blending_mode': args.blend,
        'perspective_correction': args.perspective_correction,
        'output_format': args.output_format,
        'codec': args.codec,
        'encoding_params': args.encoding_params,
        'config_file_path': config_file_path, # To log which config file was loaded, if any
        'config': args.config, # Raw argparser value, to warn if specified file not found
        'log_file': log_file
    }

config = parse_config()
# Create a columns container to hold multiple tables
tables = []
current_table = None

# Get terminal width
terminal_width = get_terminal_size().columns
max_setting_width = max(len(key.replace('_', ' ').title()) for key in config.keys() if key != 'config')
max_value_width = max(len(str(value)) for value in config.values())

# Calculate how many column pairs can fit
pair_width = max_setting_width + max_value_width + 4  # +4 for padding
num_columns = max(1, terminal_width // pair_width)

# Create tables
items = [(key, value) for key, value in config.items() if key != 'config']
for i in range(0, len(items), (len(items) + num_columns - 1) // num_columns):
    table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0,1))
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    chunk = items[i:i + (len(items) + num_columns - 1) // num_columns]
    for key, value in chunk:
        table.add_row(key.replace('_', ' ').title(), str(value))
    tables.append(table)

# Create a columns layout with the tables
table = Columns(tables, expand=True, equal=True)

# Create a panel containing the table
panel = Panel(table, title="Multicam Stitch Configuration", border_style="blue")
rprint(panel)
