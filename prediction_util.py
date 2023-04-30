import numpy as np
from typing import Tuple, List


class ObjectPosition:
    def __init__(self, coordinates: List, coordinates_2d: Tuple):
        self.x, self.y, self.z = coordinates
        self.x_2d, self.y_2d = coordinates_2d

    def calculate_distance(self, x, y, z):
        if z:
            return np.sqrt(
                np.power(self.x - x, 2) + np.power(self.y - y, 2) + np.power(self.z - z, 2)
            )
        return np.sqrt(np.power(self.x_2d - x, 2) + np.power(self.y_2d - y, 2))


class Prediction:
    def __init__(self, frame_number: int, certainties: np.ndarray):
        self._frame_number = frame_number
        self._certainties = certainties
        self._object_id = np.argmax(certainties)
        self._certainty = np.max(certainties)

    @property
    def frame_number(self):
        return self._frame_number

    @property
    def certainties(self):
        return self._certainties

    @property
    def object_id(self):
        return self._object_id

    @property
    def certainty(self):
        return self._certainty
