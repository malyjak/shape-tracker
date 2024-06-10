import argparse
import logging
import sys
import cv2 as cv
import numpy as np


from time import sleep

from pathlib import Path
from logging import Logger

BORDER_COLOR = (0, 255, 0)
POINT_COLOR = (0, 0, 255)


def process_video_file(sample: Path, logger: Logger) -> None:
    cap = cv.VideoCapture(sample)

    while cap.isOpened():
        ret, frame = cap.read()

        # Frame read is not correct.
        if not ret:
            logger.warning('Cannot receive frame - stream end?')
            cv.waitKey(0)
            break

        # Perform operations on the frame.
        # Convert to grayscale.
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Compute corners.
        # gray_np = np.float32(gray)
        # dst = cv.cornerHarris(gray_np, 2, 3, 0.04)
        # # Dilatate to make markers more visible.
        # dst = cv.dilate(dst, None)
        # # Threshold for an optimal value, it may vary depending on the image.
        # frame[dst>0.1 * dst.max()] = [0, 255, 0]

        # Compute circles.
        blurred = cv.medianBlur(gray, 3)
        # param1: the higher threshold for the edge detection and divided by 2 is the lower threshold.
        # param2: accumulator threshold - the lower the value the more circles will be returned.
        circles = cv.HoughCircles(blurred, method=cv.HOUGH_GRADIENT, dp=1, minDist=20, param1=100, param2=20, minRadius=20, maxRadius=70)
        cnt = 0
        if circles is not None:
            circles_np = np.uint16(np.around(circles))
            for pt in circles_np[0, :]:
                a, b, r = pt[0], pt[1], pt[2] 
                cv.circle(frame, (a, b), r, BORDER_COLOR, 2)
                cv.circle(frame, (a, b), 1, POINT_COLOR, 3)
                cv.circle(blurred, (a, b), r, BORDER_COLOR, 2)
                cv.circle(blurred, (a, b), 1, POINT_COLOR, 3)
                cnt = cnt + 1

            logger.debug(f'Detected {cnt} circles')
            # logger.debug(f{})

        # Display the result.
        cv.imshow('frame', frame)

        if cv.waitKey(1) == ord('q'):
            break

    # Release the capture in the end.
    cap.release()
    cv.destroyAllWindows()


def main(sample_path: str, log_file_name: str, verbosity_level: int) -> None:
    # Set logging.
    logger = logging.getLogger()
    logger.setLevel(verbosity_level * 10)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler = logging.FileHandler(Path('logs') / log_file_name)
    file_handler.setLevel(verbosity_level * 10)
    file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(verbosity_level * 10)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    # Check sample path.
    sample = Path(sample_path)
    assert(sample.exists()), f'{sample} - Path to the sample video is incorrect.'

    # Check sample extension.
    sample_extension = sample.suffix
    assert(sample_extension == '.mp4'), f'{sample_extension} - Sample video is in wrong format.'

    # Process sample.
    process_video_file(sample, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        '--sample',
                        help='path to a sample video file',
                        default='samples/luxonis_task_video.mp4')
    parser.add_argument('-l',
                        '--log',
                        help='name of log file',
                        default='main.log')
    parser.add_argument('-v',
                        '--verbosity',
                        help='level of verbosity (0 = NOTSET, 1 = DEBUG, 2 = INFO, 3 = WARNMING, 4 = ERROR, 5 = CRITICAL)',
                        type=int,
                        default=3,
                        choices=range(0, 6))
    args = parser.parse_args()

    main(args.sample, args.log, args.verbosity)
