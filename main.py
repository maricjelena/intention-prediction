import json
import os

from util import load_data, draw_skeleton, load_objects, load_objects_list, plot_probabilities_graph
from skeleton_prediction import SkeletonPredictor


DATA_PATH = "datasets//dataset_14//"
GRAPHS_DIR = "graph_results//results_14//"


def main():

    objects_list = load_objects_list(DATA_PATH)
    objects = load_objects(objects_list)

    dataframe = load_data(DATA_PATH)
    skeleton_predictor = SkeletonPredictor()

    probs = dict()

    for index, data_row in dataframe.iloc[:].iterrows():

        skeleton_prediction, probabilities = skeleton_predictor.predict(index, data_row, objects)
        probs[data_row['timestamp']] = list(probabilities)

        draw_skeleton(data_row, DATA_PATH, objects, skeleton_prediction, index)

        print(f'User is reaching for the object with index {skeleton_prediction.object_id} in the input list.')

    # save probabilities for each object in each frame
    if not os.path.exists(GRAPHS_DIR):
        os.makedirs(GRAPHS_DIR)
    with open(os.path.join(GRAPHS_DIR, "probabilities.json"), "w") as json_file:
        json.dump(probs, json_file, indent=4)

    plot_probabilities_graph(DATA_PATH, GRAPHS_DIR)


if __name__ == "__main__":
    main()
