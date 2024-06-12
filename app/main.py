#!/usr/bin/env python
"""
This script tracks rectangles and circles in provided sample video.

Usage: main.py -h

© Jakub Maly 2024
"""

import sys
import logging
import argparse

from pathlib import Path

from tracker import ShapeTracker


def main(sample_file_name: str, log_file_name: str, verbosity_level: int) -> None:
    """
    The main function of program.
    """

    # Set logging.
    logger = logging.getLogger()
    logger.setLevel(verbosity_level * 10)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler = logging.FileHandler(Path('app') / 'logs' / log_file_name)
    file_handler.setLevel(verbosity_level * 10)
    file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(verbosity_level * 10)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    # Check sample path.
    sample = Path('app') / 'samples' / sample_file_name
    assert(sample.exists()), f'{sample} - Path to the sample video is incorrect.'

    # Check sample extension.
    sample_extension = sample.suffix
    assert(sample_extension == '.mp4'), f'{sample_extension} - Sample video is in wrong format.'

    # Process sample.
    tracker = ShapeTracker(logger)
    tracker.process_video_file(sample)
    output_filename = Path(sample_file_name).stem + '.png'
    output = Path('app') / 'output' / output_filename
    tracker.save_trajectory_window(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        '--sample',
                        help='name of sample video file',
                        default='luxonis_task_video.mp4')
    parser.add_argument('-l',
                        '--log',
                        help='name of log file',
                        default='main.log')
    parser.add_argument('-v',
                        '--verbosity',
                        help='level of verbosity (0 = NOTSET, 1 = DEBUG, \
                              2 = INFO, 3 = WARNMING, 4 = ERROR, 5 = CRITICAL)',
                        type=int,
                        default=2,
                        choices=range(0, 6))
    args = parser.parse_args()

    main(args.sample, args.log, args.verbosity)
