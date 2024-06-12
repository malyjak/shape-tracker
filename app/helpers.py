"""
Helper classes and functions.

© Jakub Maly 2024
"""


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

def rgb_to_hex(color: list) -> str:
    """
    Converts rgb color to hex string.
    """

    return '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])
