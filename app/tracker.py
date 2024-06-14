"""
Tracking classes and functions.

© Jakub Maly 2024
"""

from logging import Logger
from pathlib import Path

from math import sqrt
import cv2 as cv
import numpy as np

from helpers import DataPoint, rgb_to_hex


# Hough circles algorithm params - modify this to boost circle detection.
# param1: the higher threshold for the edge detection and divided by 2 is the lower threshold.
# param2: accumulator threshold - the lower the value the more circles will be returned.
# minRadius: the minimum radius of circle to be detected
# maxRadius: the maximum radius of circle to be detected
HOUGH_PARAM_1 = 100
HOUGH_PARAM_2 = 20
HOUGH_MIN_RADIUS = 20
HOUGH_MAX_RADIUS = 70

# Element colors.
BORDER_COLOR = (0, 255, 0)
POINT_COLOR = (0, 0, 255)
FONT_COLOR = (255, 255, 255)

# Element sizes.
BORDER_SIZE = 2
POINT_SIZE = 3
LINE_SIZE = 3
FONT_SIZE = 0.5

# Font settings.
FONT_OFFSET = 10
FONT_TYPE = cv.FONT_HERSHEY_SIMPLEX

# Maximaum distance between colors to be considered similar.
COLOR_DISTANCE_THRESHOLD = 30


class ShapeTracker():
    """
    Class for tracking shapes.
    """

    def __init__(self, logger: Logger) -> None:
        self.logger = logger
        self.data_rectangles = {}
        self.data_circles = {}
        self.traj_window: cv.typing.MatLike | None = None

    def __check_for_similar_key(self, data: dict, color_list: list[int]) -> str | None:
        """
        Checks for a similar key name (color in hex format) in a given dict
        by comparing distance of color lists.
        """

        for key in data.keys():
            key_color_list = data[key][-1].get_color_list()
            # Naive implementation.
            distance = sqrt(pow(color_list[0] - key_color_list[0], 2) +
                            pow(color_list[1] - key_color_list[1], 2) +
                            pow(color_list[2] - key_color_list[2], 2))
            if distance < COLOR_DISTANCE_THRESHOLD:
                self.logger.debug((f'Detected color similarity for {color_list} '
                                   f'and {key_color_list}'))
                return key

        return None

    def get_data(self) -> list[dict]:
        """
        Returns trajectory data.
        """

        return [self.data_rectangles, self.data_circles]

    def save_trajectory_window(self, path: Path) -> None:
        """
        Saves trajectory window.
        """

        if self.traj_window is not None:
            cv.imwrite(str(path), self.traj_window)
            self.logger.debug(f'Saved trajectories window as {path}')

    def process_video_file(self, sample: Path) -> None:
        """
        Process a given sample video.
        """

        # Cleanup.
        self.data_rectangles = {}
        self.data_circles = {}
        self.traj_window = None

        cap = cv.VideoCapture(str(sample))

        while cap.isOpened():
            cnt_rectangles = 0
            cnt_circles = 0
            ret, frame = cap.read()

            # Incorrect frame read.
            if not ret:
                self.logger.warning('Cannot receive frame - stream end?')
                self.logger.info((f'Detected {len(self.data_rectangles)} distinct colors of '
                                  f'rectangles and {len(self.data_circles)} distinct colors of '
                                  'circles'))
                cv.waitKey(0)
                break

            # Create trajectories window (obtain dimensions from frame).
            if self.traj_window is None:
                self.traj_window = frame
                # Set black for background.
                self.traj_window[:] = (0, 0, 0)

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
                    if not color_hex in self.data_rectangles:
                        similar_hex = self.__check_for_similar_key(self.data_rectangles, color_list)
                        if similar_hex is None:
                            self.data_rectangles[color_hex] = list(tuple())
                        else:
                            color_hex = similar_hex
                            color_list = self.data_rectangles[similar_hex][-1].get_color_list()

                    # Draw a new point and a line from the last point if needed.
                    cv.circle(self.traj_window, (cx, cy), 1, color_list, POINT_SIZE)
                    if len(self.data_rectangles[color_hex]):
                        cv.line(self.traj_window,
                                self.data_rectangles[color_hex][-1].get_coordinates(), (cx, cy),
                                color_list, LINE_SIZE)
                    else:
                        cv.putText(self.traj_window, color_hex + ' [rectangle]',
                                   (cx + FONT_OFFSET, cy + FONT_OFFSET), FONT_TYPE, FONT_SIZE,
                                   FONT_COLOR)

                    # Add the new point to dictionary.
                    data_point = DataPoint(cx, cy, color_list)
                    self.data_rectangles[color_hex].append(data_point)
                    self.logger.debug(f'Detected {color_hex} rectangle')

                    cv.rectangle(frame, (x, y), (x + w, y + h), BORDER_COLOR, BORDER_SIZE)
                    cv.circle(frame, (cx, cy), 1, POINT_COLOR, POINT_SIZE)

                    cnt_rectangles = cnt_rectangles + 1

            # Compute circles.
            circles = cv.HoughCircles(blurred, method=cv.HOUGH_GRADIENT, dp=1, minDist=20,
                                      param1=HOUGH_PARAM_1, param2=HOUGH_PARAM_2,
                                      minRadius=HOUGH_MIN_RADIUS, maxRadius=HOUGH_MAX_RADIUS)
            if circles is not None:
                circles_np = np.uint16(np.around(circles))
                for pt in circles_np[0, :]:
                    x, y, r = pt[0], pt[1], pt[2]

                    # Get color of the detected circle.
                    color = frame[y, x]
                    color_list = list(map(int, color))
                    color_hex = rgb_to_hex(color_list)
                    if not color_hex in self.data_circles:
                        similar_hex = self.__check_for_similar_key(self.data_circles, color_list)
                        if similar_hex is None:
                            self. data_circles[color_hex] = list(tuple())
                        else:
                            color_hex = similar_hex
                            color_list = self. data_circles[similar_hex][-1].get_color_list()

                    # Draw a new point and a line from the last point if needed.
                    cv.circle(self.traj_window, (x, y), 1, color_list, POINT_SIZE)
                    if len(self.data_circles[color_hex]):
                        cv.line(self.traj_window,
                                self.data_circles[color_hex][-1].get_coordinates(), (x, y),
                                color_list, LINE_SIZE)
                    else:
                        cv.putText(self.traj_window, color_hex + ' (circle)',
                                   (x + FONT_OFFSET, y + FONT_OFFSET), FONT_TYPE, FONT_SIZE,
                                   FONT_COLOR)

                    # Add the new point to dictionary.
                    data_point = DataPoint(x, y, color_list)
                    self.data_circles[color_hex].append(data_point)
                    self.logger.debug(f'Detected {color_hex} circle')

                    cv.circle(frame, (x, y), r, BORDER_COLOR, BORDER_SIZE)
                    cv.circle(frame, (x, y), 1, POINT_COLOR, POINT_SIZE)

                    cnt_circles = cnt_circles + 1

            self.logger.debug((f'Detected {cnt_rectangles} rectangles '
                               f'and {cnt_circles} circles in this frame'))

            # Display the result.
            cv.imshow('frame', frame)
            cv.imshow('trajectories', self.traj_window)

            if cv.waitKey(1) == ord('q'):
                break

        # Release the capture in the end.
        cap.release()
        cv.destroyAllWindows()
