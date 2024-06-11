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
from logging import Logger
from math import sqrt

import cv2 as cv
import numpy as np


BORDER_COLOR = (0, 255, 0)
POINT_COLOR = (0, 0, 255)
FONT_COLOR = (255, 255, 255)

BORDER_SIZE = 2
POINT_SIZE = 3
LINE_SIZE = 3
FONT_SIZE = 0.5

FONT_OFFSET = 10
FONT_TYPE = cv.FONT_HERSHEY_SIMPLEX

COLOR_DISTANCE_THRESHOLD = 30


class DataPoint():
    """
    Class for storing coordinates and color lists.
    """

    def __init__(self, a: int, b: int, color_list: list[int]) -> None:
        self.a = a
        self.b = b
        self.color_list = color_list

    def get_color_list(self) -> list:
        """
        Returns the color list.
        """

        return self.color_list

    def get_coordinates(self) -> list[int]:
        """
        Returns the coordinates.
        """

        return [self.a, self.b]


def check_for_similar_key(data: dict, color_list: list[int], logger: Logger) -> str | None:
    """
    Checks for a similar key name (color in hex format) in a given dict.
    """

    for key in data.keys():
        key_color_list = data[key][-1].get_color_list()
        # Naive implementation.
        distance = sqrt(pow(color_list[0] - key_color_list[0], 2) +
                        pow(color_list[1] - key_color_list[1], 2) +
                        pow(color_list[2] - key_color_list[2], 2))
        if distance < COLOR_DISTANCE_THRESHOLD:
            logger.debug(f'Detected color similarity for {color_list} and {key_color_list}')
            return key

    return None

def rgb_to_hex(color: list) -> str:
    """
    Converts rgb color to hex string.
    """

    return '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])

def process_video_file(sample: Path, logger: Logger) -> None:
    """
    Process a given sample video.
    """

    cap = cv.VideoCapture(str(sample))
    traj = None
    data_rectangles = {}
    data_circles = {}

    while cap.isOpened():
        cnt_rectangles = 0
        cnt_circles = 0
        ret, frame = cap.read()

        # Incorrect frame read.
        if not ret:
            logger.warning('Cannot receive frame - stream end?')
            logger.info(f'Detected {len(data_rectangles)} distinct colors of rectangles and {len(data_circles)} distinct colors of circles')
            cv.waitKey(0)
            break

        # Create trajectories window (obtain dimensions from frame).
        if traj is None:
            traj = frame
            # Set black for background.
            traj[:] = (0, 0, 0)

        # Convert to grayscale.
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Blurr shapes.
        blurred = cv.medianBlur(gray, 3)

        # Compute rectangles.
        # Perform Canny edge detection.
        edges = cv.Canny(blurred, 50, 150)
        # Find contours in the edges image.
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Approximate the contour to get polygon.
            polygon = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
            # Check if the polygon is rectangle (it has 4 sides).
            if len(polygon) == 4:
                x, y, w, h = cv.boundingRect(polygon)

                cx = int(x + w/2)
                cy = int(y + h/2)

                # Get color of the detected rectangle.
                color = frame[cy, cx]
                color_list = list(map(int, color))
                color_hex = rgb_to_hex(color_list)
                if not color_hex in data_rectangles:
                    similar_hex = check_for_similar_key(data_rectangles, color_list, logger)
                    if similar_hex is None:
                        data_rectangles[color_hex] = list(tuple())
                    else:
                        color_hex = similar_hex
                        color_list = data_rectangles[similar_hex][-1].get_color_list()

                 # Draw a new point and a line from the last point if needed.
                cv.circle(traj, (cx, cy), 1, color_list, POINT_SIZE)
                if len(data_rectangles[color_hex]):
                    cv.line(traj,
                            data_rectangles[color_hex][-1].get_coordinates(),
                            (cx, cy), color_list, LINE_SIZE)
                else:
                    cv.putText(traj, color_hex + ' [rectangle]',
                               (cx + FONT_OFFSET, cy + FONT_OFFSET), FONT_TYPE,
                               FONT_SIZE, FONT_COLOR)

                # Add the new point to dictionary.
                data_point = DataPoint(cx, cy, color_list)
                data_rectangles[color_hex].append(data_point)
                logger.debug(f'Detected {color_hex} rectangle')

                cv.rectangle(frame, (x, y), (x + w, y + h), BORDER_COLOR,
                             BORDER_SIZE)
                cv.circle(frame, (cx, cy), 1, POINT_COLOR, POINT_SIZE)

                cnt_rectangles = cnt_rectangles + 1

        # Compute circles.
        # param1: the higher threshold for the edge detection and divided by 2
        #         is the lower threshold.
        # param2: accumulator threshold - the lower the value the more circles
        #         will be returned.
        circles = cv.HoughCircles(blurred, method=cv.HOUGH_GRADIENT, dp=1,
                                  minDist=20, param1=100, param2=20,
                                  minRadius=20, maxRadius=70)
        if circles is not None:
            circles_np = np.uint16(np.around(circles))
            for pt in circles_np[0, :]:
                x, y, r = pt[0], pt[1], pt[2]

                # Get color of the detected circle.
                color = frame[y, x]
                color_list = list(map(int, color))
                color_hex = rgb_to_hex(color_list)
                if not color_hex in data_circles:
                    similar_hex = check_for_similar_key(data_circles,
                                                        color_list, logger)
                    if similar_hex is None:
                        data_circles[color_hex] = list(tuple())
                    else:
                        color_hex = similar_hex
                        color_list = data_circles[similar_hex][-1].get_color_list()

                # Draw a new point and a line from the last point if needed.
                cv.circle(traj, (x, y), 1, color_list, POINT_SIZE)
                if len(data_circles[color_hex]):
                    cv.line(traj, data_circles[color_hex][-1].get_coordinates(),
                            (x, y), color_list, LINE_SIZE)
                else:
                    cv.putText(traj, color_hex + ' (circle)',
                               (x + FONT_OFFSET, y + FONT_OFFSET), FONT_TYPE,
                               FONT_SIZE, FONT_COLOR)

                # Add the new point to dictionary.
                data_point = DataPoint(x, y, color_list)
                data_circles[color_hex].append(data_point)
                logger.debug(f'Detected {color_hex} circle')

                cv.circle(frame, (x, y), r, BORDER_COLOR, BORDER_SIZE)
                cv.circle(frame, (x, y), 1, POINT_COLOR, POINT_SIZE)

                cnt_circles = cnt_circles + 1

        logger.debug(f'Detected {cnt_rectangles} rectangles and {cnt_circles} circles in this frame')

        # Display the result.
        cv.imshow('frame', frame)
        cv.imshow('trajectories', traj)

        if cv.waitKey(1) == ord('q'):
            break

    # Release the capture in the end.
    cap.release()
    cv.destroyAllWindows()

def main(sample_file_name: str, log_file_name: str, verbosity_level: int) -> None:
    """
    The main function of program.
    """

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
                        help='level of verbosity (0 = NOTSET, 1 = DEBUG, \
                              2 = INFO, 3 = WARNMING, 4 = ERROR, 5 = CRITICAL)',
                        type=int,
                        default=2,
                        choices=range(0, 6))
    args = parser.parse_args()

    main(args.sample, args.log, args.verbosity)
