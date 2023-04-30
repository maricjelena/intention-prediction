import numpy as np
import pandas as pd
import os
import json
import matplotlib.pylab as plt

from pandas import Series
from PIL import Image, ImageDraw
from pandas.core.frame import DataFrame
from typing import List

from prediction_util import Prediction, ObjectPosition


IMAGE_PREFIX = 'rgb_image'
IMAGE_EXTENSION = '.jpg'
SKELETON_FILENAME = 'skeleton.csv'
OBJECTS_FILE = 'objects_list.txt'
DIVIDER = 1000.0
RESULTS_DIR = "results//skeleton_results_14//"


def softmax(x, beta: float = -1.0):
    return np.exp(beta * x) / sum(np.exp(beta * x))


def load_data(data_dir) -> DataFrame:
    skeleton_path = os.path.join(data_dir, SKELETON_FILENAME)

    with open('columns.txt') as f:
        columns = f.read().splitlines()

    types_dict = {'user_id': np.int32, 'timestamp': str}
    types_dict.update({column: np.float64 for column in columns if column not in types_dict})
    print(types_dict)

    df = pd.read_csv(skeleton_path, index_col=False, header=None, names=columns, dtype=types_dict)
    return df


def draw_skeleton(df_row: Series, image_dir: str, objects: List[ObjectPosition],
                  prediction: Prediction, index):

    certainties = np.around(prediction.certainties, decimals=4)

    head = (df_row['head_x_2d'], df_row['head_y_2d'])
    neck = (df_row['neck_x_2d'], df_row['neck_y_2d'])
    left_elbow = (df_row['left_elbow_x_2d'], df_row['left_elbow_y_2d'])
    right_elbow = (df_row['right_elbow_x_2d'], df_row['right_elbow_y_2d'])
    left_shoulder = (df_row['left_shoulder_x_2d'], df_row['left_shoulder_y_2d'])
    right_shoulder = (df_row['right_shoulder_x_2d'], df_row['right_shoulder_y_2d'])
    left_hand = (df_row['left_hand_x_2d'], df_row['left_hand_y_2d'])
    right_hand = (df_row['right_hand_x_2d'], df_row['right_hand_y_2d'])
    torso = (df_row['torso_x_2d'], df_row['torso_y_2d'])

    joints = [head, neck, left_elbow, right_elbow, left_shoulder, right_shoulder, left_hand, torso]
    image_path = image_dir + IMAGE_PREFIX + df_row['timestamp'] + IMAGE_EXTENSION

    if not os.path.exists(image_path):
        print(f'Image rgb_image{str(df_row["timestamp"])}.jpg missing.')
        return

    with Image.open(image_path) as image:

        draw = ImageDraw.Draw(image)

        draw.line([head, neck], fill='blue', width=2)
        draw.line([neck, torso], fill='blue', width=2)
        draw.line([left_shoulder, neck], fill='blue', width=2)
        draw.line([neck, right_shoulder], fill='blue', width=2)
        draw.line([left_shoulder, left_elbow], fill='blue', width=2)
        draw.line([left_elbow, left_hand], fill='blue', width=2)
        draw.line([right_shoulder, right_elbow], fill='blue', width=2)
        draw.line([right_elbow, right_hand], fill='blue', width=2)

        for joint in joints:
            draw.ellipse(
                [joint[0] - 5, joint[1] - 5, joint[0] + 5, joint[1] + 5],
                fill=(0, 0, 255), outline=(0, 0, 255), width=5
            )

        # draw right hand joint with pink
        draw.ellipse(
            [right_hand[0] - 5, right_hand[1] - 5, right_hand[0] + 5, right_hand[1] + 5],
            fill=(255, 90, 180	), outline=(255, 90, 180), width=8
        )

        # write probabilities of each object
        for (obj, conf) in zip(objects, certainties):
            draw.text((obj.x_2d - 15, obj.y_2d + 10), str(conf), fill=(0, 0, 255))
            draw.ellipse(
                [obj.x_2d - 10, obj.y_2d - 10, obj.x_2d + 10, obj.y_2d + 10],
                fill=None, outline=(0, 154, 23), width=5
            )

        # draw location of the goal object with green
        draw.ellipse(
            [objects[prediction.object_id].x_2d - 12, objects[prediction.object_id].y_2d - 12,
             objects[prediction.object_id].x_2d + 12, objects[prediction.object_id].y_2d + 12],
            fill=(0, 255, 0)
        )

        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        image.save(RESULTS_DIR + "drawn" + str(index) + ".jpg")


def load_object_location(filepath: str):
    """
    Reads from specified file and returns object's 3D world coordinates and 2D image coordinates.
    Image and points are mirrored but do not need to be transformed since they correspond.
    """
    file = open(filepath, 'r')
    lines = file.readlines()

    world_location = lines[0].strip().split(',')
    image_location = lines[1].strip().split(',')

    world_location = [float(loc) / DIVIDER for loc in world_location]
    image_location = [float(pixel) for pixel in image_location]

    return world_location, image_location


def load_objects_list(filepath) -> List[str]:
    """
    Reads 'objects_list.txt' file from data folder.
    :param filepath: Path to data folder.
    :return: List of all objects info files in current scene.
    """
    with open(os.path.join(filepath, OBJECTS_FILE), 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]

    return lines


def load_objects(object_paths: list) -> List[ObjectPosition]:
    """
    :return: List of objects represented as ObjectPosition objects.
    """
    object_positions = []

    for obj_path in object_paths:
        object_3d, object_2d = load_object_location(obj_path)
        obj = ObjectPosition(object_3d, object_2d)
        object_positions.append(obj)

    return object_positions


def plot_probabilities_graph(data_dir: str, results_dir: str, seq_start: float = None, seq_end: float = None) -> None:
    """
    Plots graph of every objects reaching probability in time interval [seq_start, seq_end]
    :param data_dir: Path to directory containing reaching_moments.json file.
    :param results_dir: Path to directory containing probabilities.json file
    :param seq_start: Timestamp of the first frame of interest (optional).
    :param seq_end: Timestamp of the last frame of interest (optional).
    :return: None
    """

    styles = ['#DC143C', '#3CB371', '#BA55D3', '#1E90FF']
    markers = ["X", "X", "X", "X"]

    with open(os.path.join(results_dir, "probabilities.json"), "r") as probs_file:
        probs = json.load(probs_file)
        num_objects = 0
        dicts = []

        if seq_start is None:
            seq_start = float(min(probs.keys()))
        if seq_end is None:
            seq_end = float(max(probs.keys()))

        # create #numberOfObjects dictionaries
        for k, v in probs.items():
            num_objects = len(v)
            for _ in range(num_objects):
                dicts.append(dict())
            break

        # create dictionary for each object, Dict[timestamp] = probability
        for timestamp, probabilities in probs.items():
            if seq_start <= float(timestamp) <= seq_end:
                for obj in range(num_objects):
                    dicts[obj][float(timestamp) - seq_start] = probabilities[obj]

        for o in range(num_objects):
            x, y = zip(*dicts[o].items())  # unpack a list of pairs into two tuples
            plt.plot(x, y, color=styles[o], label='predmet ' + str(o))

        with open(os.path.join(data_dir, "reaching_moments.json"), "r") as moments_file:
            moments = json.load(moments_file)

            for u, v in moments.items():
                plt.plot([float(u) - seq_start], [dicts[int(v)][float(u) - seq_start]],
                         marker=markers[int(v)], color=styles[int(v)], markersize=10)

        plt.xlim(0, seq_end - seq_start)
        plt.xlabel("t [s]")
        plt.ylabel("p(o)")
        plt.grid()
        plt.legend()
        plt.show()
