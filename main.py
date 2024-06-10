import argparse
import logging
import sys

from pathlib import Path


def main(sample_path: str, log_file_path: str, verbosity_level: int) -> None:
    # Set logging.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler = logging.StreamHandler(log_file_path)
    file_handler.setLevel(verbosity_level)
    file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(verbosity_level)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    # Check sample path
    sample = Path(sample_path)
    assert(sample.exists()), f'{sample} - Path to the sample video is incorrect.'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        '--sample',
                        help='path to a sample video file',
                        default='samples/luxonis_task_video.mp4')
    parser.add_argument('-l',
                        '--log',
                        help='path to a log file',
                        default='logs/main.log')
    parser.add_argument('-v',
                        '--verbosity',
                        help='level of verbosity (0 = NOTSET, 1 = DEBUG, 2 = INFO, 3 = WARNMING, 4 = ERROR, 5 = CRITICAL)',
                        type=int,
                        default=3,
                        choices=range(0, 6))
    args = parser.parse_args()

    main(args.sample, args.log, args.verbosity)
