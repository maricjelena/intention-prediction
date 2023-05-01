# intention-prediction

Usage of motion intention estimator:

1. In "main.py" module, change DATA_PATH to be the path to directory containing images, skeleton data in csv format and list of objects in this scenario (datasets are available in "datasets" directory of the repository)
2. In "utils.py" module, change RESULTS_DIR to be the path to existing folder for storing images with drawn predictions.
3. run intention estimation with: python3 main.py