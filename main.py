import argparse
import logging
import sys
import cv2 as cv
import numpy as np

from pathlib import Path
from logging import Logger
from math import sqrt


BORDER_COLOR = (0, 255, 0)
POINT_COLOR = (0, 0, 255)

BORDER_SIZE = 2
POINT_SIZE = 3
LINE_SIZE = 3


COLOR_DISTANCE_THRESHOLD = 10


class DataPoint(object):
    def __init__(self, a: int, b: int, color_list: list) -> None:
        self.a = a
        self.b = b
        self.color_list = color_list

    def get_color_list(self) -> list:
        return self.color_list

    def get_coordinates(self) -> list:
        return self.a, self.b


def check_for_similar_key(data: dict, color_list) -> str | None:
    for key in data.keys():
        key_color_list = data[key][-1].get_color_list()
        # Naive implementation.
        distance = sqrt(pow(color_list[0] - key_color_list[0], 2) + pow(color_list[1] - key_color_list[1], 2) + pow(color_list[2] - key_color_list[2], 2))
        if distance < COLOR_DISTANCE_THRESHOLD:
            return key

    return None

def rgb_to_hex(color: list) -> str:
    return '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])


def process_video_file(sample: Path, logger: Logger) -> None:
    cap = cv.VideoCapture(sample)
    traj = None
    data = {}

    while cap.isOpened():
        ret, frame = cap.read()

        # Frame read is not correct.
        if not ret:
            logger.warning('Cannot receive frame - stream end?')
            logger.info(f'Detected {len(data)} distinct colors of circles')
            cv.waitKey(0)
            break

        # Create trajectories window (obtain dimensions from frame).
        if traj is None:
            traj = frame
            # Set black for background.
            traj[:] = (0, 0, 0)

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
        # Blurr shapes.
        blurred = cv.medianBlur(gray, 3)
        # param1: the higher threshold for the edge detection and divided by 2 is the lower threshold.
        # param2: accumulator threshold - the lower the value the more circles will be returned.
        circles = cv.HoughCircles(blurred, method=cv.HOUGH_GRADIENT, dp=1, minDist=20, param1=100, param2=20, minRadius=20, maxRadius=70)
        cnt = 0
        if circles is not None:
            circles_np = np.uint16(np.around(circles))
            for pt in circles_np[0, :]:
                a, b, r = pt[0], pt[1], pt[2]

                # Get color of the detected circle.
                color = frame[b, a]
                color_list = list(map(int, color))
                color_hex = rgb_to_hex(color_list)
                if not color_hex in data:
                    similar_hex = check_for_similar_key(data, color_list)
                    if similar_hex is None:
                        data[color_hex] = list(tuple())
                    else:
                        color_hex = similar_hex
                        color_list = data[similar_hex][-1].get_color_list()

                # Draw the new point and a line from the last point if needed.
                cv.circle(traj, (a, b), 1, color_list, POINT_SIZE)
                if len(data[color_hex]):
                    cv.line(traj, data[color_hex][-1].get_coordinates(), (a, b), color_list, LINE_SIZE)

                # Add the new point.
                data_point = DataPoint(a, b, color_list)
                data[color_hex].append(data_point)
                logger.debug(f'Detected {color_hex} circle')

                cv.circle(frame, (a, b), r, BORDER_COLOR, BORDER_SIZE)
                cv.circle(frame, (a, b), 1, POINT_COLOR, POINT_SIZE)

                cnt = cnt + 1

            logger.debug(f'Detected {cnt} circles in one frame')

        # Display the result.
        cv.imshow('frame', frame)
        cv.imshow('trajectories', traj)

        if cv.waitKey(1) == ord('q'):
            break

    # Release the capture in the end.
    cap.release()
    cv.destroyAllWindows()


def main(sample_file_name: str, log_file_name: str, verbosity_level: int) -> None:
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
    sample = Path('samples') / sample_file_name
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
                        help='name of sample video file',
                        default='luxonis_task_video.mp4')
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
