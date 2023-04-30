import numpy as np
from enum import Enum
from pandas import Series
from typing import List

from util import softmax
from prediction_util import Prediction, ObjectPosition


class JointType(str, Enum):
    right_elbow = 'right_elbow'
    right_hand = 'right_hand'
    torso = 'torso'


def joint_distance(
        joint_type: JointType,
        df_row: Series,
        object_position: ObjectPosition
        ) -> float:
    """
    Returns distance (in meters) between specified joint and given object.
    """
    x = df_row[joint_type + '_x'] / 1000.0
    y = df_row[joint_type + '_y'] / 1000.0
    z = df_row[joint_type + '_z'] / 1000.0

    return object_position.calculate_distance(x, y, z)


class SkeletonPredictor:
    def __init__(self):
        self._frame_predictions = dict()
        
    @property
    def frame_predictions(self):
        return self._frame_predictions

    @frame_predictions.setter
    def frame_predictions(self, frame_prediction: Prediction):
        self._frame_predictions[frame_prediction.frame_number] = frame_prediction

    def sliding_window_prediction(self, frame_number, probabilities):
        """
        Calculates weighted prediction in current frame based on last #k frames and the current one.
        :param frame_number: current frame number (id)
        :param probabilities: current probabilities for all objects based on joint distance
        :return: averaged prediction and its probability
        """
        k = 6
        frame_range = range(max(0, frame_number - k), frame_number)
        frames_len = len(frame_range)

        weights_0 = []  # weights of the last 0 frames
        current_weight_0 = 1.0

        weights_1 = [0.45]  # weights of the last 1 frame
        current_weight_1 = 0.55

        weights_2 = [0.20, 0.25]  # weights of the last 2 frames
        current_weight_2 = 0.55

        weights_3 = [0.10, 0.15, 0.25]  # weights of the last 3 frames
        current_weight_3 = 0.50

        weights_4 = [0.10, 0.10, 0.15, 0.20]  # weights of the last 4 frames
        current_weight_4 = 0.45

        weights_5 = [0.10, 0.10, 0.15, 0.15, 0.20]  # weights of the last 5 frames
        current_weight_5 = 0.30

        weights_6 = [0.10, 0.10, 0.10, 0.15, 0.15, 0.20]    # weights of the last 6 frames
        current_weight_6 = 0.20

        parameters = [
            (weights_0, current_weight_0),
            (weights_1, current_weight_1),
            (weights_2, current_weight_2),
            (weights_3, current_weight_3),
            (weights_4, current_weight_4),
            (weights_5, current_weight_5),
            (weights_6, current_weight_6)
        ]
        k_weights, current_weight = parameters[frames_len][0], parameters[frames_len][1]

        weighted_probabilities = np.zeros(len(probabilities))
        f = 0

        for frame in frame_range:
            prediction = self.frame_predictions[frame]
            object_id, certainties = prediction.object_id, prediction.certainties
            weighted_probabilities += (k_weights[f] * certainties)
            f += 1

        weighted_probabilities += (current_weight * probabilities)

        return weighted_probabilities

    def predict(self, frame_number: int, df_row: Series, object_positions: List[ObjectPosition]) -> Prediction:
        number_of_objects = len(object_positions)
        distances = np.zeros(number_of_objects)

        torso_joint_w = 0.1
        elbow_joint_w = 0.1
        hand_joint_w = 0.8

        for idx in range(0, number_of_objects):
            hand_distance = joint_distance(JointType.right_hand, df_row, object_positions[idx])
            torso_distance = joint_distance(JointType.torso, df_row, object_positions[idx])
            elbow_distance = joint_distance(JointType.right_elbow, df_row, object_positions[idx])

            distances[idx] = (
                    torso_joint_w * torso_distance + hand_joint_w * hand_distance + elbow_joint_w * elbow_distance
            )

        current_probabilities = softmax(distances, beta=-40.0)

        probabilities = self.sliding_window_prediction(frame_number, current_probabilities)

        prediction = Prediction(frame_number, probabilities)
        self.frame_predictions = prediction

        return prediction, probabilities
